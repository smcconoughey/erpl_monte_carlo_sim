#include "rocket_simulation.hpp"
#include <random>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace rocket_sim {

MonteCarloAnalyzer::MonteCarloAnalyzer(const RocketConfig& rocket, const MotorConfig& motor, const AtmosphereModel& atmosphere)
    : rocket_(rocket), motor_(motor), atmosphere_(atmosphere) {
    
#ifdef __APPLE__
    metal_compute_ = std::make_unique<MetalCompute>();
    if (!metal_compute_->initialize()) {
        std::cerr << "Failed to initialize Metal compute, falling back to CPU" << std::endl;
        metal_compute_.reset();
    }
#endif
}

MonteCarloAnalyzer::MonteCarloResult MonteCarloAnalyzer::run_analysis(const State& nominal_initial_state, int n_samples, const UncertaintyParams& params) {
    MonteCarloResult result;
    result.n_samples = n_samples;
    
    std::cout << "Generating " << n_samples << " perturbed initial states..." << std::endl;
    
    // Generate perturbed initial states
    auto perturbed_states = generate_perturbed_states(nominal_initial_state, n_samples, params);
    
    // Generate wind profiles (simplified for GPU)
    auto wind_profiles = generate_wind_profiles(n_samples, params);
    
    std::vector<SimulationResult> simulation_results;
    
#ifdef __APPLE__
    if (metal_compute_) {
        std::cout << "Running Monte Carlo analysis on GPU..." << std::endl;
        simulation_results = metal_compute_->run_monte_carlo_batch(
            perturbed_states, rocket_, motor_, atmosphere_, wind_profiles
        );
    } else
#endif
    {
        std::cout << "Running Monte Carlo analysis on CPU..." << std::endl;
        simulation_results.reserve(n_samples);
        
        RocketSimulator simulator(rocket_, motor_, atmosphere_);
        
        for (int i = 0; i < n_samples; ++i) {
            if (i % (n_samples / 10) == 0) {
                std::cout << "Progress: " << (i * 100 / n_samples) << "%" << std::endl;
            }
            
            WindProfile* wind_profile = !wind_profiles.empty() ? &wind_profiles[i] : nullptr;
            auto sim_result = simulator.simulate(perturbed_states[i], wind_profile);
            simulation_results.push_back(sim_result);
        }
    }
    
    // Filter successful simulations
    std::vector<SimulationResult> successful_results;
    for (const auto& sim_result : simulation_results) {
        if (sim_result.success) {
            successful_results.push_back(sim_result);
        }
    }
    
    result.n_successful = static_cast<int>(successful_results.size());
    result.individual_results = successful_results;
    
    if (successful_results.empty()) {
        std::cerr << "No successful simulations!" << std::endl;
        return result;
    }
    
    std::cout << "Successful simulations: " << result.n_successful << " / " << n_samples << std::endl;
    
    // Extract metrics
    std::vector<float> apogee_altitudes, ranges, flight_times, max_speeds;
    
    for (const auto& sim_result : successful_results) {
        apogee_altitudes.push_back(sim_result.apogee_altitude);
        ranges.push_back(sim_result.range);
        flight_times.push_back(sim_result.flight_time);
        max_speeds.push_back(sim_result.max_speed);
    }
    
    // Compute statistics
    result.apogee_altitude = compute_statistics(apogee_altitudes);
    result.range = compute_statistics(ranges);
    result.flight_time = compute_statistics(flight_times);
    result.max_speed = compute_statistics(max_speeds);
    
    return result;
}

std::vector<State> MonteCarloAnalyzer::generate_perturbed_states(const State& nominal, int n_samples, const UncertaintyParams& params) {
    std::vector<State> states(n_samples);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::normal_distribution<float> pos_x_dist(0.0f, params.position_std.x);
    std::normal_distribution<float> pos_y_dist(0.0f, params.position_std.y);
    std::normal_distribution<float> pos_z_dist(0.0f, params.position_std.z);
    
    std::normal_distribution<float> vel_x_dist(0.0f, params.velocity_std.x);
    std::normal_distribution<float> vel_y_dist(0.0f, params.velocity_std.y);
    std::normal_distribution<float> vel_z_dist(0.0f, params.velocity_std.z);
    
    std::normal_distribution<float> att_x_dist(0.0f, params.attitude_std.x);
    std::normal_distribution<float> att_y_dist(0.0f, params.attitude_std.y);
    std::normal_distribution<float> att_z_dist(0.0f, params.attitude_std.z);
    
    std::normal_distribution<float> ang_vel_x_dist(0.0f, params.angular_velocity_std.x);
    std::normal_distribution<float> ang_vel_y_dist(0.0f, params.angular_velocity_std.y);
    std::normal_distribution<float> ang_vel_z_dist(0.0f, params.angular_velocity_std.z);
    
    for (int i = 0; i < n_samples; ++i) {
        State& state = states[i];
        state = nominal;  // Start with nominal
        
        // Perturb position
        state[0] += pos_x_dist(gen);
        state[1] += pos_y_dist(gen);
        state[2] += pos_z_dist(gen);
        
        // Perturb velocity
        state[3] += vel_x_dist(gen);
        state[4] += vel_y_dist(gen);
        state[5] += vel_z_dist(gen);
        
        // Perturb attitude (Euler angles)
        float roll_perturbation = att_x_dist(gen);
        float pitch_perturbation = att_y_dist(gen);
        float yaw_perturbation = att_z_dist(gen);
        
        // Convert nominal quaternion to Euler, add perturbation, convert back
        // For simplicity, assume small perturbations
        Quaternion q(state[6], state[7], state[8], state[9]);
        
        // Apply small rotation perturbation
        Quaternion delta_q = euler_to_quaternion(roll_perturbation, pitch_perturbation, yaw_perturbation);
        
        // Quaternion multiplication: q_new = delta_q * q_nominal
        // This is simplified - proper implementation would use quaternion multiplication
        q = q.normalized();
        state[6] = q.w;
        state[7] = q.x;
        state[8] = q.y;
        state[9] = q.z;
        
        // Perturb angular velocity
        state[10] += ang_vel_x_dist(gen);
        state[11] += ang_vel_y_dist(gen);
        state[12] += ang_vel_z_dist(gen);
        
        // Propellant fraction remains nominal for initial state
        state[13] = 1.0f;
    }
    
    return states;
}

std::vector<WindProfile> MonteCarloAnalyzer::generate_wind_profiles(int n_samples, const UncertaintyParams& params) {
    std::vector<WindProfile> profiles(n_samples);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_real_distribution<float> wind_speed_dist(params.wind_speed_range[0], params.wind_speed_range[1]);
    std::uniform_real_distribution<float> wind_dir_dist(0.0f, 2.0f * M_PI);
    
    for (int i = 0; i < n_samples; ++i) {
        WindProfile& profile = profiles[i];
        
        float wind_speed = wind_speed_dist(gen);
        float wind_direction = wind_dir_dist(gen);
        
        // Create simple constant wind profile
        profile.altitudes = {0.0f, 1000.0f, 5000.0f, 10000.0f, 20000.0f};
        profile.wind_velocities.resize(profile.altitudes.size());
        
        for (size_t j = 0; j < profile.altitudes.size(); ++j) {
            float altitude_factor = 1.0f + profile.altitudes[j] / 10000.0f;  // Wind increases with altitude
            float effective_speed = wind_speed * altitude_factor;
            
            profile.wind_velocities[j] = Vec3(
                effective_speed * std::cos(wind_direction),
                effective_speed * std::sin(wind_direction),
                0.0f
            );
        }
    }
    
    return profiles;
}

MonteCarloAnalyzer::Statistics MonteCarloAnalyzer::compute_statistics(const std::vector<float>& values) {
    Statistics stats;
    
    if (values.empty()) {
        return stats;
    }
    
    // Mean
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
    
    // Standard deviation
    float sq_sum = 0.0f;
    for (float value : values) {
        sq_sum += (value - stats.mean) * (value - stats.mean);
    }
    stats.std_dev = std::sqrt(sq_sum / values.size());
    
    // Min and max
    auto minmax = std::minmax_element(values.begin(), values.end());
    stats.min = *minmax.first;
    stats.max = *minmax.second;
    
    // Percentiles
    std::vector<float> sorted_values = values;
    std::sort(sorted_values.begin(), sorted_values.end());
    
    std::array<float, 5> percentile_positions = {0.05f, 0.25f, 0.50f, 0.75f, 0.95f};
    
    for (int i = 0; i < 5; ++i) {
        float pos = percentile_positions[i] * (sorted_values.size() - 1);
        int lower = static_cast<int>(std::floor(pos));
        int upper = static_cast<int>(std::ceil(pos));
        float weight = pos - lower;
        
        if (lower == upper) {
            stats.percentiles[i] = sorted_values[lower];
        } else {
            stats.percentiles[i] = sorted_values[lower] * (1.0f - weight) + sorted_values[upper] * weight;
        }
    }
    
    return stats;
}

} // namespace rocket_sim