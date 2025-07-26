"""
Utility functions for rocket simulation
"""

import numpy as np

# Simple replacement for scipy.spatial.transform.Rotation to avoid dependency
class SimpleRotation:
    """Simple replacement for scipy Rotation class"""
    def __init__(self, quat):
        self.quat = quat
    
    @classmethod
    def from_euler(cls, seq, angles):
        """Convert Euler angles to quaternion (simplified implementation)"""
        if seq == "xyz":
            roll, pitch, yaw = angles
        else:
            raise NotImplementedError("Only 'xyz' sequence supported")
        
        # Convert to quaternion using standard formulas
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)
        
        # Quaternion in [x, y, z, w] format
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        w = cr * cp * cy + sr * sp * sy
        
        return cls([x, y, z, w])
    
    @classmethod
    def from_quat(cls, quat):
        """Create from quaternion"""
        return cls(quat)
    
    def as_quat(self):
        """Return quaternion in [x, y, z, w] format"""
        return self.quat
    
    def as_euler(self, seq):
        """Convert quaternion to Euler angles (simplified implementation)"""
        if seq != "xyz":
            raise NotImplementedError("Only 'xyz' sequence supported")
        
        x, y, z, w = self.quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if np.abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])

# Use our simple implementation instead of scipy
Rotation = SimpleRotation


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
    """Convert angular velocity to quaternion rate."""
    omega_q = np.array([0.0, omega[0], omega[1], omega[2]])
    q_dot = 0.5 * quaternion_multiply(q, omega_q)
    lambda_corr = 0.5  # Correction gain
    norm_error = np.dot(q, q) - 1.0
    q_dot -= lambda_corr * norm_error * q
    return q_dot


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
    """Calculate signed angle of attack from body-frame velocity."""
    if np.abs(velocity_body[0]) < 1e-6 and np.abs(velocity_body[2]) < 1e-6:
        return 0.0
    return np.arctan2(velocity_body[2], velocity_body[0])


def sideslip_angle(velocity_body):
    """Calculate signed sideslip angle from body-frame velocity."""
    V_xz = np.sqrt(velocity_body[0] ** 2 + velocity_body[2] ** 2)
    if V_xz < 1e-6:
        return 0.0
    return np.arctan2(velocity_body[1], V_xz)


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
