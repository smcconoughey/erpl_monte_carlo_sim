#include "rocket_simulation_cpp.hpp"
#include <cmath>
#include <algorithm>

namespace rocket_sim {

RocketSimulator::RocketSimulator(const RocketConfig& rocket, const MotorConfig& motor, const AtmosphereModel& atmosphere)
    : rocket_(rocket), motor_(motor), atmosphere_(atmosphere) {
}

SimulationResult RocketSimulator::simulate(const State& initial_state, const WindProfile* wind_profile) {
    const float max_time = 300.0f;
    return integrate_rk45(initial_state, max_time, wind_profile);
}

State RocketSimulator::compute_derivatives(float time, const State& state, const WindProfile* wind_profile) {
    State derivatives = {};
    
    // Extract state variables
    Vec3 position(state[0], state[1], state[2]);
    Vec3 velocity(state[3], state[4], state[5]);
    Quaternion quaternion(state[6], state[7], state[8], state[9]);
    Vec3 angular_velocity(state[10], state[11], state[12]);
    float propellant_fraction = state[13];
    
    // Normalize quaternion
    quaternion = quaternion.normalized();
    
    // Get rocket mass properties
    float current_propellant = rocket_.propellant_mass * propellant_fraction;
    float total_mass = rocket_.dry_mass + current_propellant;
    
    // Update center of mass
    float propellant_cg = rocket_.center_of_mass_dry - 0.5f;
    float current_cg = (rocket_.dry_mass * rocket_.center_of_mass_dry + 
                       current_propellant * propellant_cg) / total_mass;
    
    // Update moments of inertia
    float propellant_length = 1.2f;
    float propellant_Ixx = current_propellant * (rocket_.diameter / 4.0f) * (rocket_.diameter / 4.0f);
    float propellant_Iyy = current_propellant * (propellant_length * propellant_length / 12.0f + 
                                                (propellant_cg - current_cg) * (propellant_cg - current_cg));
    
    float current_Ixx = rocket_.Ixx_dry + propellant_Ixx;
    float current_Iyy = rocket_.Iyy_dry + propellant_Iyy;
    float current_Izz = current_Iyy;
    
    // Rotation matrix (body to inertial)
    auto R = quaternion.to_rotation_matrix();
    
    // Atmospheric properties
    float altitude = position.z;
    auto atm_props = atmosphere_.get_properties(altitude);
    
    // Wind velocity
    Vec3 wind_velocity;
    if (wind_profile) {
        wind_velocity = wind_profile->get_wind_at_altitude(altitude);
    }
    
    // Relative velocity (inertial frame)
    Vec3 velocity_relative = velocity - wind_velocity;
    
    // Transform to body frame
    Vec3 velocity_body(
        R[0][0] * velocity_relative.x + R[1][0] * velocity_relative.y + R[2][0] * velocity_relative.z,
        R[0][1] * velocity_relative.x + R[1][1] * velocity_relative.y + R[2][1] * velocity_relative.z,
        R[0][2] * velocity_relative.x + R[1][2] * velocity_relative.y + R[2][2] * velocity_relative.z
    );
    
    // Aerodynamic angles
    float mach = mach_number(velocity_relative, atm_props.temperature);
    float alpha = angle_of_attack(velocity_body);
    
    // Dynamic pressure
    float q_dynamic = 0.5f * atm_props.density * velocity_relative.magnitude() * velocity_relative.magnitude();
    
    // Forces in body frame
    Vec3 forces_body;
    Vec3 moments_body;
    
    // Thrust force
    float thrust = motor_.get_thrust(time);
    forces_body.x += thrust;
    
    // Aerodynamic forces
    if (q_dynamic > 0.0f) {
        // Drag coefficient
        float cd0 = interpolate(mach, rocket_.mach_data, rocket_.cd0_data);
        float cda = interpolate(mach, rocket_.mach_data, rocket_.cda_data);
        float cd = cd0 + cda * alpha * alpha;
        
        // Lift coefficient
        float cl_alpha = 2.0f;
        float cl = cl_alpha * alpha;
        
        // Moment coefficient
        float static_margin = rocket_.cp_location - current_cg;
        float cm_alpha = -cl_alpha * static_margin / rocket_.diameter;
        float cm = cm_alpha * alpha;
        
        // Drag force (opposite to velocity)
        float drag_magnitude = q_dynamic * cd * rocket_.reference_area;
        if (velocity_body.magnitude() > 0.0f) {
            Vec3 drag_direction = velocity_body.normalized() * -1.0f;
            forces_body = forces_body + drag_direction * drag_magnitude;
        }
        
        // Lift and side forces
        float lift_force = q_dynamic * cl * rocket_.reference_area;
        forces_body.z += lift_force;
        
        // Aerodynamic moments
        moments_body.y += q_dynamic * cm * rocket_.reference_area * rocket_.diameter;
        moments_body.z += q_dynamic * cm * rocket_.reference_area * rocket_.diameter;
    }
    
    // Transform forces to inertial frame
    Vec3 forces_inertial(
        R[0][0] * forces_body.x + R[0][1] * forces_body.y + R[0][2] * forces_body.z,
        R[1][0] * forces_body.x + R[1][1] * forces_body.y + R[1][2] * forces_body.z,
        R[2][0] * forces_body.x + R[2][1] * forces_body.y + R[2][2] * forces_body.z
    );
    
    // Gravity force
    float gravity = atmosphere_.get_gravity(altitude);
    forces_inertial.z -= total_mass * gravity;
    
    // Translational equations
    Vec3 acceleration = forces_inertial * (1.0f / total_mass);
    
    // Rotational equations (Euler's equations)
    Vec3 angular_acceleration;
    if (current_Ixx > 0.0f) {
        angular_acceleration.x = (moments_body.x - (current_Izz - current_Iyy) * angular_velocity.y * angular_velocity.z) / current_Ixx;
    }
    if (current_Iyy > 0.0f) {
        angular_acceleration.y = (moments_body.y - (current_Ixx - current_Izz) * angular_velocity.z * angular_velocity.x) / current_Iyy;
    }
    if (current_Izz > 0.0f) {
        angular_acceleration.z = (moments_body.z - (current_Iyy - current_Ixx) * angular_velocity.x * angular_velocity.y) / current_Izz;
    }
    
    // Quaternion kinematics
    Vec3 quaternion_rate = quaternion_to_angular_velocity_derivative(angular_velocity, quaternion);
    
    // Propellant consumption
    float propellant_fraction_rate = -motor_.get_mass_flow_rate(time) / rocket_.propellant_mass;
    
    // Assemble state derivative
    derivatives[0] = velocity.x;
    derivatives[1] = velocity.y;
    derivatives[2] = velocity.z;
    derivatives[3] = acceleration.x;
    derivatives[4] = acceleration.y;
    derivatives[5] = acceleration.z;
    derivatives[6] = quaternion_rate.x;
    derivatives[7] = quaternion_rate.y;
    derivatives[8] = quaternion_rate.z;
    derivatives[9] = 0.0f; // quaternion.w derivative - need to fix this
    derivatives[10] = angular_acceleration.x;
    derivatives[11] = angular_acceleration.y;
    derivatives[12] = angular_acceleration.z;
    derivatives[13] = propellant_fraction_rate;
    
    return derivatives;
}

SimulationResult RocketSimulator::integrate_rk45(const State& initial_state, float max_time, const WindProfile* wind_profile) {
    SimulationResult result;
    
    const float dt = 0.01f;  // Fixed time step for simplicity
    const float ground_threshold = 0.5f;
    
    State current_state = initial_state;
    float current_time = 0.0f;
    
    float max_altitude = 0.0f;
    float apogee_time = 0.0f;
    
    while (current_time < max_time) {
        // Store current state
        result.time.push_back(current_time);
        result.position.push_back(Vec3(current_state[0], current_state[1], current_state[2]));
        result.velocity.push_back(Vec3(current_state[3], current_state[4], current_state[5]));
        result.attitude.push_back(Quaternion(current_state[6], current_state[7], current_state[8], current_state[9]));
        result.angular_velocity.push_back(Vec3(current_state[10], current_state[11], current_state[12]));
        result.propellant_fraction.push_back(current_state[13]);
        
        // Check for apogee
        float altitude = current_state[2];
        if (altitude > max_altitude) {
            max_altitude = altitude;
            apogee_time = current_time;
        }
        
        // Check for ground impact
        if (altitude < ground_threshold && current_time > 1.0f) {
            break;
        }
        
        // RK4 integration step
        State k1 = compute_derivatives(current_time, current_state, wind_profile);
        
        State temp_state = current_state;
        for (int i = 0; i < 14; ++i) {
            temp_state[i] += dt * 0.5f * k1[i];
        }
        State k2 = compute_derivatives(current_time + dt * 0.5f, temp_state, wind_profile);
        
        temp_state = current_state;
        for (int i = 0; i < 14; ++i) {
            temp_state[i] += dt * 0.5f * k2[i];
        }
        State k3 = compute_derivatives(current_time + dt * 0.5f, temp_state, wind_profile);
        
        temp_state = current_state;
        for (int i = 0; i < 14; ++i) {
            temp_state[i] += dt * k3[i];
        }
        State k4 = compute_derivatives(current_time + dt, temp_state, wind_profile);
        
        // Update state
        for (int i = 0; i < 14; ++i) {
            current_state[i] += dt * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]) / 6.0f;
        }
        
        // Normalize quaternion
        Quaternion q(current_state[6], current_state[7], current_state[8], current_state[9]);
        q = q.normalized();
        current_state[6] = q.w;
        current_state[7] = q.x;
        current_state[8] = q.y;
        current_state[9] = q.z;
        
        current_time += dt;
    }
    
    // Calculate final results
    result.apogee_altitude = max_altitude;
    result.apogee_time = apogee_time;
    result.flight_time = current_time;
    
    if (!result.position.empty()) {
        Vec3 final_pos = result.position.back();
        result.range = std::sqrt(final_pos.x * final_pos.x + final_pos.y * final_pos.y);
        
        float max_speed = 0.0f;
        for (const auto& vel : result.velocity) {
            max_speed = std::max(max_speed, vel.magnitude());
        }
        result.max_speed = max_speed;
    }
    
    result.success = true;
    return result;
}

float RocketSimulator::interpolate(float x, const std::vector<float>& x_data, const std::vector<float>& y_data) const {
    if (x_data.empty() || y_data.empty()) {
        return 0.0f;
    }
    
    if (x <= x_data.front()) {
        return y_data.front();
    }
    if (x >= x_data.back()) {
        return y_data.back();
    }
    
    // Find the two points to interpolate between
    size_t i = 0;
    while (i < x_data.size() - 1 && x_data[i + 1] < x) {
        ++i;
    }
    
    float x0 = x_data[i];
    float x1 = x_data[i + 1];
    float y0 = y_data[i];
    float y1 = y_data[i + 1];
    
    // Linear interpolation
    float alpha = (x - x0) / (x1 - x0);
    return y0 + alpha * (y1 - y0);
}

float RocketSimulator::mach_number(const Vec3& velocity, float temperature) const {
    const float gamma = 1.4f;
    const float R = 287.053f;
    float speed_of_sound = std::sqrt(gamma * R * temperature);
    return velocity.magnitude() / speed_of_sound;
}

float RocketSimulator::angle_of_attack(const Vec3& velocity_body) const {
    float V_total = velocity_body.magnitude();
    if (V_total < 1e-6f) {
        return 0.0f;
    }
    return std::atan2(std::sqrt(velocity_body.y*velocity_body.y + velocity_body.z*velocity_body.z), 
                      velocity_body.x);
}

} // namespace rocket_sim