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

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix."""
    q = normalize_quaternion(q)
    w, x, y, z = q

    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ]
    )


def angular_velocity_to_quaternion_rate(omega, q):
    """Convert angular velocity to quaternion rate.

    Parameters
    ----------
    omega : array-like
        Angular velocity vector in the body frame [rad/s].
    q : array-like
        Current orientation quaternion representing the rotation from the body
        frame to the inertial frame.

    Returns
    -------
    numpy.ndarray
        Time derivative of the orientation quaternion.

    Notes
    -----
    The quaternion derivative for a body-to-inertial orientation is given by
    ``q_dot = 0.5 * q ⊗ [0, ω]`` where ``⊗`` denotes quaternion multiplication
    and ``ω`` is the angular velocity in the body frame.  The previous
    implementation incorrectly right-multiplied the current quaternion, leading
    to inverted attitude updates and large errors in the integrated attitude.
    """

    omega_q = np.array([0.0, omega[0], omega[1], omega[2]])
    return 0.5 * quaternion_multiply(q, omega_q)


def skew_symmetric(v):
    """Create skew-symmetric matrix from vector."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles to quaternion."""
    r = Rotation.from_euler("xyz", [roll, pitch, yaw])
    quat_scipy = r.as_quat()  # Returns [x, y, z, w] format
    # Convert to [w, x, y, z] format for our simulation
    return np.array(
        [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]
    )


def quaternion_to_euler(q):
    """Convert quaternion to Euler angles."""
    # Convert from [w, x, y, z] to [x, y, z, w] format for scipy
    q_scipy = np.array([q[1], q[2], q[3], q[0]])
    r = Rotation.from_quat(q_scipy)
    return r.as_euler("xyz")


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
    return np.arctan2(
        np.sqrt(velocity_body[1] ** 2 + velocity_body[2] ** 2),
        velocity_body[0],
    )


def sideslip_angle(velocity_body):
    """Calculate sideslip angle from body-frame velocity."""
    V_total = np.linalg.norm(velocity_body)
    if V_total < 1e-6:
        return 0.0
    return np.arctan2(velocity_body[1], velocity_body[0])


def wind_to_body_matrix(alpha, beta):
    """Rotation matrix from wind axes to body axes.

    Parameters
    ----------
    alpha : float
        Angle of attack (rad).
    beta : float
        Sideslip angle (rad).

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix that transforms vectors from the wind reference
        frame (x along the relative wind) to the body frame.  Positive ``alpha``
        corresponds to a nose-up rotation and positive ``beta`` corresponds to a
        nose-right rotation.
    """

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)

    return np.array(
        [
            [ca * cb, -sb, sa * cb],
            [ca * sb, cb, sa * sb],
            [-sa, 0.0, ca],
        ]
    )


def to_serializable(obj):
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    return obj


def object_to_serializable_dict(obj):
    """Convert an object's __dict__ to a JSON-serializable dictionary."""
    return {k: to_serializable(v) for k, v in obj.__dict__.items()}
