#include "rocket_simulation_cpp.hpp"
#include <algorithm>

namespace rocket_sim {

MotorConfig::MotorConfig() {
    average_thrust = total_impulse / burn_time;
    
    // Initialize thrust curve
    time_data = {0.0f, 0.2f, 0.5f, 1.0f, 2.0f, 5.0f, 8.0f, 12.0f, 14.0f, 15.0f};
    
    // Normalized thrust curve
    std::vector<float> normalized_thrust = {0.0f, 2.2f, 2.0f, 1.8f, 1.5f, 1.2f, 1.0f, 0.8f, 0.3f, 0.0f};
    
    // Convert to actual thrust values
    thrust_data.resize(normalized_thrust.size());
    for (size_t i = 0; i < normalized_thrust.size(); ++i) {
        thrust_data[i] = normalized_thrust[i] * average_thrust;
    }
}

float MotorConfig::get_thrust(float time) const {
    if (time < 0.0f || time > burn_time) {
        return 0.0f;
    }
    
    // Linear interpolation
    if (time <= time_data.front()) {
        return thrust_data.front();
    }
    if (time >= time_data.back()) {
        return thrust_data.back();
    }
    
    // Find the two points to interpolate between
    size_t i = 0;
    while (i < time_data.size() - 1 && time_data[i + 1] < time) {
        ++i;
    }
    
    float t0 = time_data[i];
    float t1 = time_data[i + 1];
    float f0 = thrust_data[i];
    float f1 = thrust_data[i + 1];
    
    // Linear interpolation
    float alpha = (time - t0) / (t1 - t0);
    return f0 + alpha * (f1 - f0);
}

float MotorConfig::get_mass_flow_rate(float time) const {
    if (time < 0.0f || time > burn_time) {
        return 0.0f;
    }
    
    // Simplified: constant mass flow rate
    return mass_flow_rate;
}

} // namespace rocket_sim