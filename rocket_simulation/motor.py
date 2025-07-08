"""
Rocket motor and thrust modeling
"""

import numpy as np
from utils import interpolate_1d

class SolidMotor:
    """Solid propellant motor model."""

    def __init__(self, name="Solid Motor"):
        self.name = name

        # Motor properties
        self.total_impulse = 156297  # N-s (35,122 lbs)
        self.burn_time = 15.0  # seconds
        self.propellant_mass = 63.5  # kg (140 lb)
        self.average_thrust = self.total_impulse / self.burn_time

        # Thrust values at standard conditions (approximate)
        # Provided data: 2,290 lbf sea level, 2,590 lbf in vacuum
        self.thrust_sea_level = 2290 * 4.44822  # N
        self.thrust_vacuum = 2590 * 4.44822  # N

        # Nozzle exit area estimated from SL/vacuum difference
        self.nozzle_exit_area = (
            self.thrust_vacuum - self.thrust_sea_level
        ) / 101325.0  # Pa
        
        # Thrust curve definition (time vs thrust)
        self.thrust_curve_time = np.array([
            0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 8.0, 12.0, 14.0, 15.0
        ])
        
        # Normalized thrust curve (thrust/average_thrust) - High performance profile
        self.thrust_curve_normalized = np.array([
            0.0, 2.2, 2.0, 1.8, 1.5, 1.2, 1.0, 0.8, 0.3, 0.0
        ])
        
        # Convert to actual thrust values
        self.thrust_curve_thrust = self.thrust_curve_normalized * self.average_thrust
        
        # Mass flow rate (approximately constant)
        self.mass_flow_rate = 4.26  # kg/s (9.4 lb/s)
        
        # Exhaust velocity
        self.exhaust_velocity = self.average_thrust / self.mass_flow_rate
        
        # Uncertainty parameters for Monte Carlo
        self.thrust_uncertainty = 0.05  # 5% standard deviation
        self.burn_time_uncertainty = 0.02  # 2% standard deviation
        self.total_impulse_uncertainty = 0.03  # 3% standard deviation
        
    def get_thrust(self, time, ambient_pressure=None):
        """Get thrust at given time.

        Parameters
        ----------
        time : float
            Time since motor ignition (s).
        ambient_pressure : float, optional
            Ambient pressure in Pascals. If provided, the thrust is adjusted
            between sea level and vacuum values based on this pressure.
        """
        if time < 0 or time > self.burn_time:
            return 0.0

        # Base thrust (sea level reference)
        thrust_sl = interpolate_1d(time, self.thrust_curve_time, self.thrust_curve_thrust)

        if ambient_pressure is None:
            return thrust_sl

        # Pressure correction using estimated nozzle exit area
        pressure_correction = self.nozzle_exit_area * (101325.0 - ambient_pressure)
        return thrust_sl + pressure_correction
    
    def get_mass_flow_rate(self, time):
        """Get mass flow rate at given time."""
        if time < 0 or time > self.burn_time:
            return 0.0
        
        # Simplified: constant mass flow rate
        return self.mass_flow_rate
    
    def get_propellant_remaining(self, time):
        """Get fraction of propellant remaining."""
        if time <= 0:
            return 1.0
        elif time >= self.burn_time:
            return 0.0
        else:
            return max(0.0, 1.0 - time / self.burn_time)
    
    def perturb_for_monte_carlo(self, random_state=None):
        """Create perturbed motor for Monte Carlo analysis."""
        if random_state is None:
            random_state = np.random.RandomState()
        
        # Create perturbed motor
        perturbed_motor = SolidMotor(self.name + "_perturbed")
        
        # Perturb thrust
        thrust_multiplier = random_state.normal(1.0, self.thrust_uncertainty)
        perturbed_motor.thrust_curve_thrust = self.thrust_curve_thrust * thrust_multiplier
        perturbed_motor.average_thrust = self.average_thrust * thrust_multiplier
        perturbed_motor.thrust_sea_level = self.thrust_sea_level * thrust_multiplier
        perturbed_motor.thrust_vacuum = self.thrust_vacuum * thrust_multiplier
        
        # Perturb burn time
        burn_time_multiplier = random_state.normal(1.0, self.burn_time_uncertainty)
        perturbed_motor.burn_time = self.burn_time * burn_time_multiplier
        
        # Perturb total impulse
        impulse_multiplier = random_state.normal(1.0, self.total_impulse_uncertainty)
        perturbed_motor.total_impulse = self.total_impulse * impulse_multiplier
        
        # Recalculate derived properties
        perturbed_motor.mass_flow_rate = 4.26 * thrust_multiplier  # Scale mass flow rate with thrust
        perturbed_motor.exhaust_velocity = perturbed_motor.average_thrust / perturbed_motor.mass_flow_rate

        # Update nozzle exit area (assume linear with thrust multiplier)
        perturbed_motor.nozzle_exit_area = self.nozzle_exit_area * thrust_multiplier
        
        return perturbed_motor


class LiquidMotor:
    """Simple liquid rocket engine with altitude-dependent performance."""

    def __init__(self, name="Liquid Motor", thrust_vacuum=10000.0,
                 thrust_sea_level=9000.0, mass_flow_rate=5.0,
                 propellant_mass=100.0):
        self.name = name

        self.thrust_vacuum = thrust_vacuum
        self.thrust_sea_level = thrust_sea_level
        self.mass_flow_rate = mass_flow_rate
        self.propellant_mass = propellant_mass

        self.nozzle_exit_area = (
            self.thrust_vacuum - self.thrust_sea_level) / 101325.0

        self.burn_time = self.propellant_mass / self.mass_flow_rate
        self.total_impulse = self.thrust_vacuum * self.burn_time

        self.thrust_uncertainty = 0.05
        self.mass_flow_uncertainty = 0.03

    def get_thrust(self, time, ambient_pressure=101325.0):
        if time < 0 or time > self.burn_time:
            return 0.0

        return self.thrust_vacuum - self.nozzle_exit_area * ambient_pressure

    def get_mass_flow_rate(self, time):
        if time < 0 or time > self.burn_time:
            return 0.0
        return self.mass_flow_rate

    def get_propellant_remaining(self, time):
        if time <= 0:
            return 1.0
        elif time >= self.burn_time:
            return 0.0
        else:
            return max(0.0, 1.0 - time / self.burn_time)

    def perturb_for_monte_carlo(self, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState()

        thrust_mult = random_state.normal(1.0, self.thrust_uncertainty)
        mass_flow_mult = random_state.normal(1.0, self.mass_flow_uncertainty)

        perturbed = LiquidMotor(
            self.name + "_perturbed",
            thrust_vacuum=self.thrust_vacuum * thrust_mult,
            thrust_sea_level=self.thrust_sea_level * thrust_mult,
            mass_flow_rate=self.mass_flow_rate * mass_flow_mult,
            propellant_mass=self.propellant_mass,
        )

        return perturbed
