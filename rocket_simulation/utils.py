"""
Utility functions for rocket simulation
"""

import numpy as np
from scipy.spatial.transform import Rotation

def normalize_quaternion(q):
    """Normalize a quaternion to unit length."""
    norm = np.linalg.norm(q)
    if norm > 1e-12:
        return q / norm
    else:
        return np.array([1.0, 0.0, 0.0, 0.0])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    q = normalize_quaternion(q)
    w, x, y, z = q
    
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])

def angular_velocity_to_quaternion_rate(omega, q):
    """Convert angular velocity to quaternion rate."""
    omega_q = np.array([0, omega[0], omega[1], omega[2]])
    return 0.5 * quaternion_multiply(omega_q, q)

def skew_symmetric(v):
    """Create skew-symmetric matrix from vector."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion."""
    r = Rotation.from_euler('xyz', [roll, pitch, yaw])
    quat_scipy = r.as_quat()  # Returns [x, y, z, w] format
    # Convert to [w, x, y, z] format for our simulation
    return np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]])

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles."""
    # Convert from [w, x, y, z] to [x, y, z, w] format for scipy
    q_scipy = np.array([q[1], q[2], q[3], q[0]])
    r = Rotation.from_quat(q_scipy)
    return r.as_euler('xyz')

def interpolate_1d(x, x_data, y_data):
    """Linear interpolation for 1D data."""
    return np.interp(x, x_data, y_data)

def mach_number(velocity, temperature):
    """Calculate Mach number from velocity and temperature."""
    gamma = 1.4  # Specific heat ratio for air
    R = 287.053  # Specific gas constant for air (J/kg/K)
    speed_of_sound = np.sqrt(gamma * R * temperature)
    return np.linalg.norm(velocity) / speed_of_sound

def angle_of_attack(velocity_body):
    """Calculate angle of attack from body-frame velocity."""
    V_total = np.linalg.norm(velocity_body)
    if V_total < 1e-6:
        return 0.0
    return np.arctan2(np.sqrt(velocity_body[1]**2 + velocity_body[2]**2), velocity_body[0])

def sideslip_angle(velocity_body):
    """Calculate sideslip angle from body-frame velocity."""
    V_total = np.linalg.norm(velocity_body)
    if V_total < 1e-6:
        return 0.0
    return np.arctan2(velocity_body[2], velocity_body[0]) 