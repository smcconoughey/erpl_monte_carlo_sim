#include "rocket_simulation_cpp.hpp"
#include <iostream>
#include <chrono>

using namespace rocket_sim;

void print_statistics(const std::string& name, const MonteCarloAnalyzer::Statistics& stats) {
    std::cout << name << ":" << std::endl;
    std::cout << "  Mean: " << stats.mean << std::endl;
    std::cout << "  Std Dev: " << stats.std_dev << std::endl;
    std::cout << "  Min: " << stats.min << std::endl;
    std::cout << "  Max: " << stats.max << std::endl;
    std::cout << "  Percentiles [5%, 25%, 50%, 75%, 95%]: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << stats.percentiles[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << std::endl << std::endl;
}

int main() {
    std::cout << "C++ Rocket Simulation (CPU-only)" << std::endl;
    std::cout << "=================================" << std::endl << std::endl;
    
    // Create rocket configuration
    RocketConfig rocket;
    MotorConfig motor;
    AtmosphereModel atmosphere;
    
    std::cout << "Rocket Configuration:" << std::endl;
    std::cout << "  Total mass: " << (rocket.dry_mass + rocket.propellant_mass) << " kg" << std::endl;
    std::cout << "  Length: " << rocket.length << " m" << std::endl;
    std::cout << "  Diameter: " << rocket.diameter << " m" << std::endl;
    std::cout << "  Center of pressure: " << rocket.cp_location << " m" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Motor Configuration:" << std::endl;
    std::cout << "  Total impulse: " << motor.total_impulse << " N-s" << std::endl;
    std::cout << "  Burn time: " << motor.burn_time << " s" << std::endl;
    std::cout << "  Average thrust: " << motor.average_thrust << " N" << std::endl;
    std::cout << std::endl;
    
    // Define initial conditions
    State initial_state = {};
    initial_state[0] = 0.0f;    // x position
    initial_state[1] = 0.0f;    // y position
    initial_state[2] = 10.0f;   // z position (altitude)
    initial_state[3] = 0.0f;    // x velocity
    initial_state[4] = 0.0f;    // y velocity
    initial_state[5] = 0.0f;    // z velocity
    
    // Initial attitude (nearly vertical with small perturbation)
    Quaternion initial_attitude = euler_to_quaternion(0.0f, -M_PI/2 + 0.02f, 0.0f);
    initial_state[6] = initial_attitude.w;
    initial_state[7] = initial_attitude.x;
    initial_state[8] = initial_attitude.y;
    initial_state[9] = initial_attitude.z;
    
    initial_state[10] = 0.0f;   // x angular velocity
    initial_state[11] = 0.0f;   // y angular velocity
    initial_state[12] = 0.0f;   // z angular velocity
    initial_state[13] = 1.0f;   // propellant fraction
    
    // Run single simulation first
    std::cout << "Running single simulation..." << std::endl;
    RocketSimulator simulator(rocket, motor, atmosphere);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    SimulationResult single_result = simulator.simulate(initial_state);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Single Simulation Results:" << std::endl;
    std::cout << "  Simulation time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Apogee altitude: " << single_result.apogee_altitude << " m (" 
              << (single_result.apogee_altitude * 3.28084f) << " ft)" << std::endl;
    std::cout << "  Range: " << single_result.range << " m" << std::endl;
    std::cout << "  Flight time: " << single_result.flight_time << " s" << std::endl;
    std::cout << "  Max speed: " << single_result.max_speed << " m/s" << std::endl;
    std::cout << "  Success: " << (single_result.success ? "Yes" : "No") << std::endl;
    std::cout << std::endl;
    
    // Run Monte Carlo analysis
    std::cout << "Running Monte Carlo analysis..." << std::endl;
    
    MonteCarloAnalyzer monte_carlo(rocket, motor, atmosphere);
    
    MonteCarloAnalyzer::UncertaintyParams uncertainty;
    uncertainty.position_std = Vec3(1.0f, 1.0f, 1.0f);        // 1m std dev in position
    uncertainty.velocity_std = Vec3(0.5f, 0.5f, 0.5f);        // 0.5 m/s std dev in velocity
    uncertainty.attitude_std = Vec3(0.02f, 0.02f, 0.02f);     // ~1 degree std dev in attitude
    uncertainty.angular_velocity_std = Vec3(0.01f, 0.01f, 0.01f);  // Small angular velocity uncertainty
    uncertainty.mass_uncertainty = 0.02f;                      // 2% mass uncertainty
    uncertainty.thrust_uncertainty = 0.05f;                    // 5% thrust uncertainty
    uncertainty.wind_speed_range[0] = 0.0f;                    // Min wind speed
    uncertainty.wind_speed_range[1] = 15.0f;                   // Max wind speed
    uncertainty.density_uncertainty = 0.1f;                    // 10% atmospheric density uncertainty
    
    const int n_samples = 1000;  // Moderate sample for demonstration
    
    start_time = std::chrono::high_resolution_clock::now();
    auto mc_result = monte_carlo.run_analysis(initial_state, n_samples, uncertainty);
    end_time = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << std::endl << "Monte Carlo Analysis Results:" << std::endl;
    std::cout << "  Total samples: " << mc_result.n_samples << std::endl;
    std::cout << "  Successful simulations: " << mc_result.n_successful << std::endl;
    std::cout << "  Success rate: " << (100.0f * mc_result.n_successful / mc_result.n_samples) << "%" << std::endl;
    std::cout << "  Total computation time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Time per simulation: " << (duration.count() / static_cast<float>(n_samples)) << " ms" << std::endl;
    std::cout << std::endl;
    
    print_statistics("Apogee Altitude (m)", mc_result.apogee_altitude);
    print_statistics("Range (m)", mc_result.range);
    print_statistics("Flight Time (s)", mc_result.flight_time);
    print_statistics("Max Speed (m/s)", mc_result.max_speed);
    
    // Performance comparison
    float python_time_estimate = n_samples * 100.0f;  // Assume ~100ms per simulation in Python
    float speedup = python_time_estimate / duration.count();
    
    std::cout << "Performance Comparison:" << std::endl;
    std::cout << "  Estimated Python time: " << python_time_estimate << " ms" << std::endl;
    std::cout << "  C++ (multithreaded) time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Estimated speedup: " << speedup << "x" << std::endl;
    
    return 0;
}