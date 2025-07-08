"""
6DOF flight dynamics and numerical integration
"""

import numpy as np
from scipy.integrate import solve_ivp
from utils import *

class FlightSimulator:
    """6DOF flight dynamics simulator."""
    
    def __init__(self, rocket, motor, atmosphere, wind_model):
        self.rocket = rocket
        self.motor = motor
        self.atmosphere = atmosphere
        self.wind_model = wind_model
        
        # Integration parameters
        self.max_time = 300.0  # Maximum flight time (s)
        self.dt_initial = 0.01  # Initial time step (s)
        self.rtol = 1e-4  # Relative tolerance (relaxed from 1e-6)
        self.atol = 1e-7  # Absolute tolerance (relaxed from 1e-9)
        
        # Event detection
        self.ground_altitude = 0.0  # Ground level (m)
        self.apogee_detected = False
        
        # Initialize wind profiles
        self.wind_profile = None
        self.altitude_profile = None
        
    def simulate_flight(self, initial_conditions, wind_profile=None, altitude_profile=None):
        """Simulate rocket flight with 6DOF dynamics."""
        # Initialize state vector
        # State: [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz, propellant_fraction]
        state0 = np.zeros(14)
        
        # Initial position
        state0[0:3] = initial_conditions.get('position', [0.0, 0.0, 0.0])
        
        # Initial velocity
        state0[3:6] = initial_conditions.get('velocity', [0.0, 0.0, 0.0])
        
        # Initial attitude (quaternion)
        initial_euler = initial_conditions.get('attitude', [0.0, 0.0, 0.0])
        state0[6:10] = euler_to_quaternion(initial_euler[0], initial_euler[1], initial_euler[2])
        
        # Initial angular velocity
        state0[10:13] = initial_conditions.get('angular_velocity', [0.0, 0.0, 0.0])
        
        # Initial propellant fraction
        state0[13] = 1.0
        
        # Store wind profile
        self.wind_profile = wind_profile
        self.altitude_profile = altitude_profile
        
        # Set up integration
        def dynamics(t, state):
            return self._rocket_dynamics(t, state)
        
        def ground_impact(t, state):
            # Only trigger when altitude goes below 0.5m (ground level)
            return state[2] - 0.5  # altitude minus ground threshold
        ground_impact.terminal = True
        ground_impact.direction = -1
        
        # Solve ODE
        solution = solve_ivp(
            dynamics, 
            [0, self.max_time], 
            state0,
            method='RK45',
            rtol=self.rtol,
            atol=self.atol,
            events=ground_impact,
            dense_output=True
        )
        
        # Extract results
        results = self._extract_results(solution)
        
        return results
    
    def _rocket_dynamics(self, t, state):
        """6DOF rocket dynamics equations."""
        # Extract state variables
        position = state[0:3]
        velocity = state[3:6]
        quaternion = state[6:10]
        angular_velocity = state[10:13]
        propellant_fraction = state[13]
        
        # Normalize quaternion
        quaternion = normalize_quaternion(quaternion)
        
        # Get rocket mass properties
        mass_props = self.rocket.get_mass_properties(propellant_fraction)
        mass = mass_props['mass']
        Ixx = mass_props['Ixx']
        Iyy = mass_props['Iyy']
        Izz = mass_props['Izz']
        
        # Rotation matrix (body to inertial)
        R_body_to_inertial = quaternion_to_rotation_matrix(quaternion)
        
        # Atmospheric properties
        altitude = position[2]
        atm_props = self.atmosphere.get_properties(altitude)
        density = atm_props['density']
        temperature = atm_props['temperature']
        
        # Wind velocity
        if self.wind_profile is not None and self.altitude_profile is not None:
            wind_velocity = self.wind_model.get_wind_at_altitude(
                altitude, self.wind_profile, self.altitude_profile
            )
        else:
            wind_velocity = np.array([0.0, 0.0, 0.0])
        
        # Relative velocity (inertial frame)
        velocity_relative = velocity - wind_velocity
        
        # Transform to body frame
        velocity_body = R_body_to_inertial.T @ velocity_relative
        
        # Aerodynamic angles
        mach = mach_number(velocity_relative, temperature)
        alpha = angle_of_attack(velocity_body)
        beta = sideslip_angle(velocity_body)
        
        # Dynamic pressure
        q_dynamic = 0.5 * density * np.linalg.norm(velocity_relative)**2
        
        # Forces in body frame
        forces_body = np.zeros(3)
        moments_body = np.zeros(3)
        
        # Thrust force
        thrust = self.motor.get_thrust(t)
        forces_body[0] += thrust  # Thrust along x-axis
        
        # Aerodynamic forces
        if q_dynamic > 0:
            aero_coeffs = self.rocket.get_aerodynamic_coefficients(mach, alpha, beta)
            
            # Drag force (opposite to velocity)
            drag_magnitude = q_dynamic * aero_coeffs['cd'] * self.rocket.reference_area
            if np.linalg.norm(velocity_body) > 0:
                drag_direction = -velocity_body / np.linalg.norm(velocity_body)
                forces_body += drag_magnitude * drag_direction
            
            # Lift and side forces (simplified)
            lift_force = q_dynamic * aero_coeffs['cl'] * self.rocket.reference_area
            side_force = q_dynamic * aero_coeffs['cy'] * self.rocket.reference_area
            
            forces_body[1] += side_force
            forces_body[2] += lift_force
            
            # Aerodynamic moments
            moments_body[0] += q_dynamic * aero_coeffs['croll'] * self.rocket.reference_area * self.rocket.reference_diameter
            moments_body[1] += q_dynamic * aero_coeffs['cpitch'] * self.rocket.reference_area * self.rocket.reference_diameter
            moments_body[2] += q_dynamic * aero_coeffs['cyaw'] * self.rocket.reference_area * self.rocket.reference_diameter
        
        # Transform forces to inertial frame
        forces_inertial = R_body_to_inertial @ forces_body
        
        # Gravity force
        gravity = self.atmosphere.get_gravity(altitude)
        forces_inertial[2] -= mass * gravity
        
        # Translational equations
        acceleration = forces_inertial / mass
        
        # Rotational equations (Euler's equations)
        angular_acceleration = np.zeros(3)
        
        # Moment equations
        if Ixx > 0:
            angular_acceleration[0] = (moments_body[0] - (Izz - Iyy) * angular_velocity[1] * angular_velocity[2]) / Ixx
        if Iyy > 0:
            angular_acceleration[1] = (moments_body[1] - (Ixx - Izz) * angular_velocity[2] * angular_velocity[0]) / Iyy
        if Izz > 0:
            angular_acceleration[2] = (moments_body[2] - (Iyy - Ixx) * angular_velocity[0] * angular_velocity[1]) / Izz
        
        # Quaternion kinematics
        quaternion_rate = angular_velocity_to_quaternion_rate(angular_velocity, quaternion)
        
        # Propellant consumption
        propellant_fraction_rate = -self.motor.get_mass_flow_rate(t) / self.rocket.propellant_mass
        
        # Assemble state derivative
        state_dot = np.zeros(14)
        state_dot[0:3] = velocity
        state_dot[3:6] = acceleration
        state_dot[6:10] = quaternion_rate
        state_dot[10:13] = angular_acceleration
        state_dot[13] = propellant_fraction_rate
        
        return state_dot
    
    def _extract_results(self, solution):
        """Extract and organize simulation results."""
        time = solution.t
        states = solution.y
        
        # Extract state components
        positions = states[0:3, :]
        velocities = states[3:6, :]
        quaternions = states[6:10, :]
        angular_velocities = states[10:13, :]
        propellant_fractions = states[13, :]
        
        # Calculate derived quantities
        altitudes = positions[2, :]
        speeds = np.linalg.norm(velocities, axis=0)
        
        # Find apogee
        apogee_index = np.argmax(altitudes)
        apogee_time = time[apogee_index]
        apogee_altitude = altitudes[apogee_index]
        
        # Calculate range
        final_position = positions[:, -1]
        range_distance = np.sqrt(final_position[0]**2 + final_position[1]**2)
        
        # Calculate Euler angles and additional flight metrics
        euler_angles = np.zeros((3, len(time)))
        center_of_mass = np.zeros(len(time))
        angle_of_attack_hist = np.zeros(len(time))
        sideslip_hist = np.zeros(len(time))
        stability_margin = np.zeros(len(time))

        for i in range(len(time)):
            euler_angles[:, i] = quaternion_to_euler(quaternions[:, i])

            # Center of mass and aerodynamic angles
            mass_props = self.rocket.get_mass_properties(propellant_fractions[i])
            center_of_mass[i] = mass_props['center_of_mass']
            stability_margin[i] = (self.rocket.cp_location - center_of_mass[i]) / self.rocket.reference_diameter

            alt = positions[2, i]
            atm_props = self.atmosphere.get_properties(alt)
            temp = atm_props['temperature']
            if self.wind_profile is not None and self.altitude_profile is not None:
                wind_vel = self.wind_model.get_wind_at_altitude(alt, self.wind_profile, self.altitude_profile)
            else:
                wind_vel = np.array([0.0, 0.0, 0.0])

            vel_rel = velocities[:, i] - wind_vel
            vel_body = quaternion_to_rotation_matrix(quaternions[:, i]).T @ vel_rel
            angle_of_attack_hist[i] = angle_of_attack(vel_body)
            sideslip_hist[i] = sideslip_angle(vel_body)
        
        results = {
            'time': time,
            'position': positions,
            'velocity': velocities,
            'quaternion': quaternions,
            'angular_velocity': angular_velocities,
            'propellant_fraction': propellant_fractions,
            'altitude': altitudes,
            'speed': speeds,
            'euler_angles': euler_angles,
            'center_of_mass': center_of_mass,
            'cp_location': self.rocket.cp_location,
            'stability_margin': stability_margin,
            'angle_of_attack': angle_of_attack_hist,
            'sideslip_angle': sideslip_hist,
            'apogee_time': apogee_time,
            'apogee_altitude': apogee_altitude,
            'range': range_distance,
            'flight_time': time[-1]
        }
        
        return results 