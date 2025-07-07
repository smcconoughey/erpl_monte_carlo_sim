#include "rocket_simulation_cpp.hpp"
#include <cmath>

namespace rocket_sim {

Quaternion Quaternion::normalized() const {
    float norm = std::sqrt(w*w + x*x + y*y + z*z);
    if (norm > 1e-12f) {
        return Quaternion(w/norm, x/norm, y/norm, z/norm);
    } else {
        return Quaternion(1.0f, 0.0f, 0.0f, 0.0f);
    }
}

std::array<std::array<float, 3>, 3> Quaternion::to_rotation_matrix() const {
    Quaternion q = normalized();
    float w2 = q.w * q.w;
    float x2 = q.x * q.x;
    float y2 = q.y * q.y;
    float z2 = q.z * q.z;
    
    return {{
        {{1-2*(y2+z2), 2*(q.x*q.y-q.w*q.z), 2*(q.x*q.z+q.w*q.y)}},
        {{2*(q.x*q.y+q.w*q.z), 1-2*(x2+z2), 2*(q.y*q.z-q.w*q.x)}},
        {{2*(q.x*q.z-q.w*q.y), 2*(q.y*q.z+q.w*q.x), 1-2*(x2+y2)}}
    }};
}

// Utility functions
Vec3 euler_to_quaternion_vec3(float roll, float pitch, float yaw) {
    float cr = std::cos(roll * 0.5f);
    float sr = std::sin(roll * 0.5f);
    float cp = std::cos(pitch * 0.5f);
    float sp = std::sin(pitch * 0.5f);
    float cy = std::cos(yaw * 0.5f);
    float sy = std::sin(yaw * 0.5f);

    return Vec3(
        sr * cp * cy - cr * sp * sy, // x
        cr * sp * cy + sr * cp * sy, // y
        cr * cp * sy - sr * sp * cy  // z
    );
}

Quaternion euler_to_quaternion(float roll, float pitch, float yaw) {
    float cr = std::cos(roll * 0.5f);
    float sr = std::sin(roll * 0.5f);
    float cp = std::cos(pitch * 0.5f);
    float sp = std::sin(pitch * 0.5f);
    float cy = std::cos(yaw * 0.5f);
    float sy = std::sin(yaw * 0.5f);

    return Quaternion(
        cr * cp * cy + sr * sp * sy, // w
        sr * cp * cy - cr * sp * sy, // x
        cr * sp * cy + sr * cp * sy, // y
        cr * cp * sy - sr * sp * cy  // z
    );
}

Vec3 quaternion_to_angular_velocity_derivative(const Vec3& omega, const Quaternion& q) {
    // Convert angular velocity to quaternion rate
    // dq/dt = 0.5 * omega_quaternion * q
    return Vec3(
        0.5f * (-q.x * omega.x - q.y * omega.y - q.z * omega.z),
        0.5f * ( q.w * omega.x + q.y * omega.z - q.z * omega.y),
        0.5f * ( q.w * omega.y - q.x * omega.z + q.z * omega.x)
    );
}

float mach_number(const Vec3& velocity, float temperature) {
    const float gamma = 1.4f;
    const float R = 287.053f;
    float speed_of_sound = std::sqrt(gamma * R * temperature);
    return velocity.magnitude() / speed_of_sound;
}

float angle_of_attack(const Vec3& velocity_body) {
    float V_total = velocity_body.magnitude();
    if (V_total < 1e-6f) {
        return 0.0f;
    }
    return std::atan2(std::sqrt(velocity_body.y*velocity_body.y + velocity_body.z*velocity_body.z), 
                      velocity_body.x);
}

float sideslip_angle(const Vec3& velocity_body) {
    float V_total = velocity_body.magnitude();
    if (V_total < 1e-6f) {
        return 0.0f;
    }
    return std::atan2(velocity_body.z, velocity_body.x);
}

} // namespace rocket_sim