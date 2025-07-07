#include <metal_stdlib>
using namespace metal;

// State vector: [x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, propellant_fraction]
typedef float14 State;

struct RocketParams {
    float dry_mass;
    float propellant_mass;
    float length;
    float diameter;
    float reference_area;
    float cp_location;
    float center_of_mass_dry;
    float Ixx_dry;
    float Iyy_dry;
    float Izz_dry;
};

struct MotorParams {
    float total_impulse;
    float burn_time;
    float average_thrust;
    float mass_flow_rate;
    // Thrust curve stored in separate buffer
};

struct AtmosphereParams {
    float sea_level_pressure;
    float sea_level_temperature;
    float sea_level_density;
    float temperature_lapse_rate;
    float gas_constant;
    float gravity;
};

struct ThrustCurvePoint {
    float time;
    float thrust;
};

struct SimulationParams {
    float max_time;
    float dt;
    float ground_threshold;
    int max_steps;
};

struct SimulationResult {
    float apogee_altitude;
    float apogee_time;
    float range;
    float flight_time;
    float max_speed;
    bool success;
};

// Utility functions
float3 normalize_vec3(float3 v) {
    float len = length(v);
    return len > 1e-12 ? v / len : float3(0);
}

float4 normalize_quaternion(float4 q) {
    float len = length(q);
    return len > 1e-12 ? q / len : float4(1, 0, 0, 0);
}

float3x3 quaternion_to_rotation_matrix(float4 q) {
    q = normalize_quaternion(q);
    float w = q.x, x = q.y, y = q.z, z = q.w;
    
    return float3x3(
        float3(1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)),
        float3(2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)),
        float3(2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))
    );
}

float interpolate_thrust(float time, constant ThrustCurvePoint* thrust_curve, int curve_length) {
    if (time <= 0.0 || curve_length == 0) {
        return 0.0;
    }
    
    if (time <= thrust_curve[0].time) {
        return thrust_curve[0].thrust;
    }
    
    if (time >= thrust_curve[curve_length-1].time) {
        return thrust_curve[curve_length-1].thrust;
    }
    
    // Binary search for efficiency
    int left = 0, right = curve_length - 1;
    while (right - left > 1) {
        int mid = (left + right) / 2;
        if (thrust_curve[mid].time <= time) {
            left = mid;
        } else {
            right = mid;
        }
    }
    
    float t0 = thrust_curve[left].time;
    float t1 = thrust_curve[right].time;
    float f0 = thrust_curve[left].thrust;
    float f1 = thrust_curve[right].thrust;
    
    float alpha = (time - t0) / (t1 - t0);
    return f0 + alpha * (f1 - f0);
}

float get_atmospheric_density(float altitude, constant AtmosphereParams& atm) {
    const float troposphere_height = 11000.0;
    const float stratosphere_height = 20000.0;
    const float stratosphere_temp = 216.65;
    
    float temperature, pressure;
    
    if (altitude <= troposphere_height) {
        temperature = atm.sea_level_temperature - atm.temperature_lapse_rate * altitude;
        pressure = atm.sea_level_pressure * 
            pow(temperature / atm.sea_level_temperature, 
                atm.gravity / (atm.gas_constant * atm.temperature_lapse_rate));
    } else if (altitude <= stratosphere_height) {
        temperature = stratosphere_temp;
        float pressure_11km = atm.sea_level_pressure * 
            pow(stratosphere_temp / atm.sea_level_temperature,
                atm.gravity / (atm.gas_constant * atm.temperature_lapse_rate));
        
        pressure = pressure_11km * 
            exp(-atm.gravity * (altitude - troposphere_height) / 
                (atm.gas_constant * temperature));
    } else {
        temperature = stratosphere_temp;
        pressure = 1000.0 * exp(-altitude / 8000.0);
    }
    
    return pressure / (atm.gas_constant * temperature);
}

float get_atmospheric_temperature(float altitude, constant AtmosphereParams& atm) {
    const float troposphere_height = 11000.0;
    const float stratosphere_temp = 216.65;
    
    if (altitude <= troposphere_height) {
        return atm.sea_level_temperature - atm.temperature_lapse_rate * altitude;
    } else {
        return stratosphere_temp;
    }
}

// Compute derivatives for the rocket dynamics
State compute_derivatives(float time, State state, 
                         constant RocketParams& rocket,
                         constant MotorParams& motor,
                         constant AtmosphereParams& atmosphere,
                         constant ThrustCurvePoint* thrust_curve,
                         int thrust_curve_length) {
    
    State derivatives = 0;
    
    // Extract state variables
    float3 position = state.xyz;
    float3 velocity = state.s345;
    float4 quaternion = state.s6789;
    float3 angular_velocity = state.s101112;
    float propellant_fraction = state.s13;
    
    // Normalize quaternion
    quaternion = normalize_quaternion(quaternion);
    
    // Get rocket mass properties
    float current_propellant = rocket.propellant_mass * propellant_fraction;
    float total_mass = rocket.dry_mass + current_propellant;
    
    // Rotation matrix (body to inertial)
    float3x3 R = quaternion_to_rotation_matrix(quaternion);
    
    // Atmospheric properties
    float altitude = position.z;
    float density = get_atmospheric_density(altitude, atmosphere);
    float temperature = get_atmospheric_temperature(altitude, atmosphere);
    
    // Relative velocity (assume no wind for GPU computation)
    float3 velocity_relative = velocity;
    
    // Transform to body frame
    float3 velocity_body = transpose(R) * velocity_relative;
    
    // Aerodynamic angles
    float alpha = atan2(sqrt(velocity_body.y*velocity_body.y + velocity_body.z*velocity_body.z), 
                       velocity_body.x);
    
    // Dynamic pressure
    float q_dynamic = 0.5 * density * dot(velocity_relative, velocity_relative);
    
    // Forces in body frame
    float3 forces_body = float3(0);
    float3 moments_body = float3(0);
    
    // Thrust force
    float thrust = interpolate_thrust(time, thrust_curve, thrust_curve_length);
    forces_body.x += thrust;
    
    // Aerodynamic forces (simplified)
    if (q_dynamic > 0.0) {
        float cd = 0.45 + 1.25 * alpha * alpha;  // Simplified drag model
        float cl_alpha = 2.0;
        float cl = cl_alpha * alpha;
        
        // Drag force
        float drag_magnitude = q_dynamic * cd * rocket.reference_area;
        if (length(velocity_body) > 0.0) {
            float3 drag_direction = -normalize_vec3(velocity_body);
            forces_body += drag_magnitude * drag_direction;
        }
        
        // Lift force
        float lift_force = q_dynamic * cl * rocket.reference_area;
        forces_body.z += lift_force;
        
        // Aerodynamic moments (simplified)
        float static_margin = rocket.cp_location - rocket.center_of_mass_dry;
        float cm = -cl_alpha * static_margin / rocket.diameter * alpha;
        moments_body.y += q_dynamic * cm * rocket.reference_area * rocket.diameter;
    }
    
    // Transform forces to inertial frame
    float3 forces_inertial = R * forces_body;
    
    // Gravity force
    const float earth_radius = 6.371e6;
    float gravity = atmosphere.gravity * pow(earth_radius / (earth_radius + altitude), 2.0);
    forces_inertial.z -= total_mass * gravity;
    
    // Translational equations
    float3 acceleration = forces_inertial / total_mass;
    
    // Rotational equations (simplified)
    float3 angular_acceleration = moments_body / float3(rocket.Ixx_dry, rocket.Iyy_dry, rocket.Izz_dry);
    
    // Quaternion kinematics (simplified)
    float4 quaternion_rate = 0.5 * float4(
        -dot(quaternion.yzw, angular_velocity),
        quaternion.x * angular_velocity.x + quaternion.z * angular_velocity.z - quaternion.w * angular_velocity.y,
        quaternion.x * angular_velocity.y - quaternion.y * angular_velocity.z + quaternion.w * angular_velocity.x,
        quaternion.x * angular_velocity.z + quaternion.y * angular_velocity.y - quaternion.z * angular_velocity.x
    );
    
    // Propellant consumption
    float propellant_fraction_rate = -motor.mass_flow_rate / rocket.propellant_mass;
    if (time > motor.burn_time) {
        propellant_fraction_rate = 0.0;
    }
    
    // Assemble derivatives
    derivatives.xyz = velocity;
    derivatives.s345 = acceleration;
    derivatives.s6789 = quaternion_rate;
    derivatives.s101112 = angular_acceleration;
    derivatives.s13 = propellant_fraction_rate;
    
    return derivatives;
}

// Main Monte Carlo kernel
kernel void monte_carlo_simulation(
    device const State* initial_states [[buffer(0)]],
    device SimulationResult* results [[buffer(1)]],
    constant RocketParams& rocket [[buffer(2)]],
    constant MotorParams& motor [[buffer(3)]],
    constant AtmosphereParams& atmosphere [[buffer(4)]],
    constant ThrustCurvePoint* thrust_curve [[buffer(5)]],
    constant SimulationParams& sim_params [[buffer(6)]],
    constant int& thrust_curve_length [[buffer(7)]],
    uint index [[thread_position_in_grid]])
{
    State current_state = initial_states[index];
    float current_time = 0.0;
    
    float max_altitude = 0.0;
    float apogee_time = 0.0;
    float max_speed = 0.0;
    
    bool success = true;
    int step = 0;
    
    // Simple RK4 integration
    while (current_time < sim_params.max_time && step < sim_params.max_steps) {
        // Store maximum values
        float altitude = current_state.z;
        float speed = length(current_state.s345);
        
        if (altitude > max_altitude) {
            max_altitude = altitude;
            apogee_time = current_time;
        }
        
        if (speed > max_speed) {
            max_speed = speed;
        }
        
        // Check for ground impact
        if (altitude < sim_params.ground_threshold && current_time > 1.0) {
            break;
        }
        
        // RK4 step
        State k1 = compute_derivatives(current_time, current_state, rocket, motor, atmosphere, thrust_curve, thrust_curve_length);
        State k2 = compute_derivatives(current_time + sim_params.dt * 0.5, current_state + sim_params.dt * 0.5 * k1, rocket, motor, atmosphere, thrust_curve, thrust_curve_length);
        State k3 = compute_derivatives(current_time + sim_params.dt * 0.5, current_state + sim_params.dt * 0.5 * k2, rocket, motor, atmosphere, thrust_curve, thrust_curve_length);
        State k4 = compute_derivatives(current_time + sim_params.dt, current_state + sim_params.dt * k3, rocket, motor, atmosphere, thrust_curve, thrust_curve_length);
        
        current_state += sim_params.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        
        // Normalize quaternion
        current_state.s6789 = normalize_quaternion(current_state.s6789);
        
        current_time += sim_params.dt;
        step++;
    }
    
    // Calculate final results
    float3 final_position = current_state.xyz;
    float range = sqrt(final_position.x * final_position.x + final_position.y * final_position.y);
    
    // Store results
    results[index].apogee_altitude = max_altitude;
    results[index].apogee_time = apogee_time;
    results[index].range = range;
    results[index].flight_time = current_time;
    results[index].max_speed = max_speed;
    results[index].success = success && (step < sim_params.max_steps);
}