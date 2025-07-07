#pragma once

#include <array>
#include <vector>
#include <memory>
#include <functional>

#ifdef __APPLE__
#include <Metal/Metal.h>
#include <MetalKit/MetalKit.h>
#endif

namespace rocket_sim {

// Core data structures
struct Vec3 {
    float x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    Vec3 operator+(const Vec3& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Vec3 operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar}; }
    float dot(const Vec3& other) const { return x * other.x + y * other.y + z * other.z; }
    float magnitude() const { return std::sqrt(x*x + y*y + z*z); }
    Vec3 normalized() const { float mag = magnitude(); return mag > 0 ? *this * (1.0f / mag) : Vec3(); }
};

struct Quaternion {
    float w, x, y, z;
    
    Quaternion() : w(1), x(0), y(0), z(0) {}
    Quaternion(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_) {}
    
    Quaternion normalized() const;
    std::array<std::array<float, 3>, 3> to_rotation_matrix() const;
};

// 14-element state vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, propellant_fraction]
using State = std::array<float, 14>;

// Rocket configuration
struct RocketConfig {
    float dry_mass = 113.4f;           // kg
    float propellant_mass = 63.5f;     // kg
    float length = 7.62f;              // m
    float diameter = 0.219f;           // m
    float reference_area;              // m^2
    float cp_location;                 // m from nose
    float center_of_mass_dry = 1.8f;   // m from nose
    
    // Moments of inertia
    float Ixx_dry = 10.9f;             // kg*m^2
    float Iyy_dry = 36.3f;             // kg*m^2
    float Izz_dry = 36.3f;             // kg*m^2
    
    // Aerodynamic data
    std::vector<float> mach_data;
    std::vector<float> cd0_data;
    std::vector<float> cda_data;
    
    RocketConfig();
    void calculate_derived_properties();
};

// Motor configuration
struct MotorConfig {
    float total_impulse = 156297.0f;   // N-s
    float burn_time = 15.0f;           // s
    float propellant_mass = 63.5f;     // kg
    float average_thrust;              // N
    float mass_flow_rate = 4.26f;      // kg/s
    
    // Thrust curve
    std::vector<float> time_data;
    std::vector<float> thrust_data;
    
    MotorConfig();
    float get_thrust(float time) const;
    float get_mass_flow_rate(float time) const;
};

// Atmosphere model
struct AtmosphereModel {
    float sea_level_pressure = 101325.0f;      // Pa
    float sea_level_temperature = 288.15f;     // K
    float sea_level_density = 1.225f;          // kg/m^3
    float temperature_lapse_rate = 0.0065f;    // K/m
    float gas_constant = 287.053f;             // J/(kg*K)
    float gravity = 9.80665f;                  // m/s^2
    
    struct AtmosphereProperties {
        float temperature;
        float pressure;
        float density;
        float speed_of_sound;
    };
    
    AtmosphereProperties get_properties(float altitude) const;
    float get_gravity(float altitude) const;
};

// Wind model
struct WindProfile {
    std::vector<float> altitudes;
    std::vector<Vec3> wind_velocities;
    
    Vec3 get_wind_at_altitude(float altitude) const;
};

// Integration result
struct SimulationResult {
    std::vector<float> time;
    std::vector<Vec3> position;
    std::vector<Vec3> velocity;
    std::vector<Quaternion> attitude;
    std::vector<Vec3> angular_velocity;
    std::vector<float> propellant_fraction;
    
    float apogee_altitude;
    float apogee_time;
    float range;
    float flight_time;
    float max_speed;
    
    bool success = false;
};

// Core simulator
class RocketSimulator {
public:
    RocketSimulator(const RocketConfig& rocket, const MotorConfig& motor, const AtmosphereModel& atmosphere);
    
    SimulationResult simulate(const State& initial_state, const WindProfile* wind_profile = nullptr);
    
private:
    RocketConfig rocket_;
    MotorConfig motor_;
    AtmosphereModel atmosphere_;
    
    // RK45 integration
    State compute_derivatives(float time, const State& state, const WindProfile* wind_profile);
    SimulationResult integrate_rk45(const State& initial_state, float max_time, const WindProfile* wind_profile);
    
    // Helper functions
    float interpolate(float x, const std::vector<float>& x_data, const std::vector<float>& y_data) const;
    float mach_number(const Vec3& velocity, float temperature) const;
    float angle_of_attack(const Vec3& velocity_body) const;
};

#ifdef __APPLE__
// Metal compute interface
class MetalCompute {
public:
    MetalCompute();
    ~MetalCompute();
    
    bool initialize();
    void shutdown();
    
    // Monte Carlo simulation on GPU
    std::vector<SimulationResult> run_monte_carlo_batch(
        const std::vector<State>& initial_states,
        const RocketConfig& rocket,
        const MotorConfig& motor,
        const AtmosphereModel& atmosphere,
        const std::vector<WindProfile>& wind_profiles
    );
    
private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLComputePipelineState> monte_carlo_pipeline_;
    id<MTLLibrary> library_;
    
    bool create_compute_pipeline();
};
#endif

// Monte Carlo analyzer
class MonteCarloAnalyzer {
public:
    struct UncertaintyParams {
        Vec3 position_std = {0.0f, 0.0f, 0.0f};
        Vec3 velocity_std = {0.1f, 0.1f, 0.1f};
        Vec3 attitude_std = {0.01f, 0.01f, 0.01f};
        Vec3 angular_velocity_std = {0.01f, 0.01f, 0.01f};
        float mass_uncertainty = 0.02f;
        float thrust_uncertainty = 0.05f;
        float wind_speed_range[2] = {0.0f, 15.0f};
        float density_uncertainty = 0.1f;
    };
    
    struct Statistics {
        float mean;
        float std_dev;
        float min;
        float max;
        std::array<float, 5> percentiles; // 5%, 25%, 50%, 75%, 95%
    };
    
    struct MonteCarloResult {
        int n_samples;
        int n_successful;
        Statistics apogee_altitude;
        Statistics range;
        Statistics flight_time;
        Statistics max_speed;
        std::vector<SimulationResult> individual_results;
    };
    
    MonteCarloAnalyzer(const RocketConfig& rocket, const MotorConfig& motor, const AtmosphereModel& atmosphere);
    
    MonteCarloResult run_analysis(const State& nominal_initial_state, int n_samples, const UncertaintyParams& params);
    
private:
    RocketConfig rocket_;
    MotorConfig motor_;
    AtmosphereModel atmosphere_;
    
#ifdef __APPLE__
    std::unique_ptr<MetalCompute> metal_compute_;
#endif
    
    std::vector<State> generate_perturbed_states(const State& nominal, int n_samples, const UncertaintyParams& params);
    std::vector<WindProfile> generate_wind_profiles(int n_samples, const UncertaintyParams& params);
    Statistics compute_statistics(const std::vector<float>& values);
};

} // namespace rocket_sim