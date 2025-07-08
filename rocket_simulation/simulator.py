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

    def _simulate_launch_rail(self, state, rail_length=18.288):
        """Simulate simple guided motion along a launch rail.

        Returns
        -------
        tuple
            Updated state after leaving the rail, time at rail exit and a
            dictionary with additional metrics such as exit velocity and
            aerodynamic angles.
        """
        position = state[0:3].copy()
        velocity = state[3:6].copy()
        quaternion = state[6:10]
        prop_frac = state[13]

        direction = quaternion_to_rotation_matrix(quaternion)[:, 0]

        distance = 0.0
        t = 0.0
        dt = self.dt_initial

        while distance < rail_length and t < self.motor.burn_time:
            mass_props = self.rocket.get_mass_properties(prop_frac)
            mass = mass_props['mass']
            atm = self.atmosphere.get_properties(position[2])
            density = atm['density']
            temp = atm['temperature']

            speed = np.dot(velocity, direction)
            mach = mach_number(direction * speed, temp)
            aero_coeffs = self.rocket.get_aerodynamic_coefficients(mach, 0.0)
            drag = 0.5 * density * speed ** 2 * aero_coeffs['cd'] * self.rocket.reference_area

            thrust = self.motor.get_thrust(t, atm['pressure'])
            gravity = self.atmosphere.get_gravity(position[2])
            accel = (thrust - mass * gravity - drag) / mass

            speed += accel * dt
            position += direction * speed * dt
            distance += speed * dt
            velocity = direction * speed

            t += dt
            prop_frac = self.motor.get_propellant_remaining(t)

        state[0:3] = position
        state[3:6] = velocity
        state[13] = prop_frac

        # Collect additional information about rail exit conditions
        rail_info = {
            'rail_exit_time': t,
            'rail_exit_position': position.copy(),
            'rail_exit_velocity': velocity.copy(),
            'rail_exit_speed': float(np.linalg.norm(velocity)),
            'rail_exit_euler': quaternion_to_euler(quaternion)
        }

        # Determine aerodynamic angles and wind at rail exit
        if self.wind_profile is not None and self.altitude_profile is not None:
            wind_vel = self.wind_model.get_wind_at_altitude(
                position[2], self.wind_profile, self.altitude_profile
            )
        else:
            wind_vel = np.array([0.0, 0.0, 0.0])

        vel_rel = velocity - wind_vel
        vel_body = quaternion_to_rotation_matrix(quaternion).T @ vel_rel
        rail_info['rail_exit_angle_of_attack'] = angle_of_attack(vel_body)
        rail_info['rail_exit_sideslip'] = sideslip_angle(vel_body)
        rail_info['wind_at_exit'] = wind_vel

        return state, t, rail_info
        
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

        # Record the exact initial conditions used
        initial_conditions_used = {
            'position': state0[0:3].tolist(),
            'velocity': state0[3:6].tolist(),
            'attitude': initial_euler,
            'angular_velocity': state0[10:13].tolist(),
        }

        # Initial propellant fraction
        state0[13] = 1.0
        
        # Store wind profile
        self.wind_profile = wind_profile
        self.altitude_profile = altitude_profile
        
        # Simulate guided launch rail phase
        state0, rail_time, rail_info = self._simulate_launch_rail(state0)

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
            [rail_time, self.max_time],
            state0,
            method='RK45',
            rtol=self.rtol,
            atol=self.atol,
            events=ground_impact,
            dense_output=True
        )

        # Extract results and shift time to start at zero after rail
        results = self._extract_results(solution, time_offset=rail_time)

        # Append rail exit information for diagnostics
        results.update(rail_info)

        # Include metadata for debugging
        results['initial_conditions'] = initial_conditions_used
        results['rocket_parameters'] = object_to_serializable_dict(self.rocket)
        results['motor_parameters'] = object_to_serializable_dict(self.motor)
        results['simulation_assumptions'] = {
            'max_time': self.max_time,
            'dt_initial': self.dt_initial,
            'rtol': self.rtol,
            'atol': self.atol,
            'rail_length': 18.288,
        }
        if wind_profile is not None and altitude_profile is not None:
            results['wind_profile'] = wind_profile
            results['altitude_profile'] = altitude_profile

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
        thrust = self.motor.get_thrust(t, atm_props['pressure'])
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
    
    def _extract_results(self, solution, time_offset=0.0):
        """Extract and organize simulation results."""
        time = solution.t - time_offset
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
        mass_history = np.zeros(len(time))
        moi_history = np.zeros((3, len(time)))  # Ixx, Iyy, Izz
        thrust_history = np.zeros(len(time))
        drag_history = np.zeros(len(time))
        coeff_cd = np.zeros(len(time))
        coeff_cl = np.zeros(len(time))
        coeff_cm = np.zeros(len(time))
        angle_of_attack_hist = np.zeros(len(time))
        sideslip_hist = np.zeros(len(time))
        stability_margin = np.zeros(len(time))
        cp_history = np.zeros(len(time))

        for i in range(len(time)):
            euler_angles[:, i] = quaternion_to_euler(quaternions[:, i])

            # Mass properties and aerodynamic angles
            mass_props = self.rocket.get_mass_properties(propellant_fractions[i])
            center_of_mass[i] = mass_props['center_of_mass']
            mass_history[i] = mass_props['mass']
            moi_history[0, i] = mass_props['Ixx']
            moi_history[1, i] = mass_props['Iyy']
            moi_history[2, i] = mass_props['Izz']

            alt = positions[2, i]
            atm_props = self.atmosphere.get_properties(alt)
            temp = atm_props['temperature']
            if self.wind_profile is not None and self.altitude_profile is not None:
                wind_vel = self.wind_model.get_wind_at_altitude(alt, self.wind_profile, self.altitude_profile)
            else:
                wind_vel = np.array([0.0, 0.0, 0.0])

            vel_rel = velocities[:, i] - wind_vel
            vel_body = quaternion_to_rotation_matrix(quaternions[:, i]).T @ vel_rel
            mach = mach_number(vel_rel, temp)
            aoa = angle_of_attack(vel_body)
            cp_val = self.rocket.get_dynamic_cp(mach, aoa)
            aero_coeffs = self.rocket.get_aerodynamic_coefficients(mach, aoa)

            q_dyn = 0.5 * atm_props['density'] * np.linalg.norm(vel_rel) ** 2
            drag_history[i] = q_dyn * aero_coeffs['cd'] * self.rocket.reference_area
            thrust_history[i] = self.motor.get_thrust(time[i], atm_props['pressure'])
            coeff_cd[i] = aero_coeffs['cd']
            coeff_cl[i] = aero_coeffs['cl']
            coeff_cm[i] = aero_coeffs['cm']

            cp_history[i] = cp_val
            stability_margin[i] = (cp_val - center_of_mass[i]) / self.rocket.reference_diameter

            angle_of_attack_hist[i] = aoa
            sideslip_hist[i] = sideslip_angle(vel_body)
        
        results = {
            'time': time,
            'position': positions,
            'velocity': velocities,
            'quaternion': quaternions,
            'angular_velocity': angular_velocities,
            'propellant_fraction': propellant_fractions,
            'mass': mass_history,
            'moments_of_inertia': moi_history,
            'altitude': altitudes,
            'speed': speeds,
            'euler_angles': euler_angles,
            'center_of_mass': center_of_mass,
            'thrust': thrust_history,
            'drag': drag_history,
            'cd': coeff_cd,
            'cl': coeff_cl,
            'cm': coeff_cm,
            'cp_location_dynamic': cp_history,
            'cp_location': self.rocket.cp_location,
            'thrust_curve_time': getattr(self.motor, 'thrust_curve_time', None),
            'thrust_curve_thrust': getattr(self.motor, 'thrust_curve_thrust', None),
            'stability_margin': stability_margin,
            'angle_of_attack': angle_of_attack_hist,
            'sideslip_angle': sideslip_hist,
            'apogee_time': apogee_time,
            'apogee_altitude': apogee_altitude,
            'range': range_distance,
            'flight_time': time[-1]
        }
        
        return results 