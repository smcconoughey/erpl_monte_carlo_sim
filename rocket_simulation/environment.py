"""
Atmospheric and environmental modeling
"""

import numpy as np
from utils import interpolate_1d

class StandardAtmosphere:
    """1976 U.S. Standard Atmosphere model."""
    
    def __init__(self):
        # Standard atmosphere constants
        self.sea_level_pressure = 101325.0  # Pa
        self.sea_level_temperature = 288.15  # K
        self.sea_level_density = 1.225  # kg/m^3
        self.temperature_lapse_rate = 0.0065  # K/m
        self.gas_constant = 287.053  # J/(kg*K)
        self.gravity = 9.80665  # m/s^2
        self.gamma = 1.4  # Specific heat ratio
        
        # Atmosphere layers
        self.troposphere_height = 11000.0  # m
        self.stratosphere_height = 20000.0  # m
        self.stratosphere_temp = 216.65  # K
        
    def get_properties(self, altitude):
        """Get atmospheric properties at given altitude."""
        if altitude <= self.troposphere_height:
            # Troposphere
            temperature = self.sea_level_temperature - self.temperature_lapse_rate * altitude
            pressure = self.sea_level_pressure * (
                temperature / self.sea_level_temperature
            )**(self.gravity / (self.gas_constant * self.temperature_lapse_rate))
            
        elif altitude <= self.stratosphere_height:
            # Lower stratosphere (isothermal)
            temperature = self.stratosphere_temp
            pressure_11km = self.sea_level_pressure * (
                self.stratosphere_temp / self.sea_level_temperature
            )**(self.gravity / (self.gas_constant * self.temperature_lapse_rate))
            
            pressure = pressure_11km * np.exp(
                -self.gravity * (altitude - self.troposphere_height) / 
                (self.gas_constant * temperature)
            )
            
        else:
            # Extended atmosphere (simplified exponential decay)
            temperature = self.stratosphere_temp
            pressure = 1000.0 * np.exp(-altitude / 8000.0)  # Simplified
            
        # Calculate density
        density = pressure / (self.gas_constant * temperature)
        
        # Calculate speed of sound
        speed_of_sound = np.sqrt(self.gamma * self.gas_constant * temperature)
        
        return {
            'temperature': temperature,
            'pressure': pressure,
            'density': density,
            'speed_of_sound': speed_of_sound
        }
    
    def get_gravity(self, altitude):
        """Get gravitational acceleration at altitude."""
        earth_radius = 6.371e6  # m
        return self.gravity * (earth_radius / (earth_radius + altitude))**2

class WindModel:
    """Wind modeling for rocket simulation."""
    
    def __init__(self):
        self.power_law_exponent = 0.14  # Typical value for neutral atmosphere
        self.turbulence_intensity = 2.0  # m/s
        self.correlation_length = 100.0  # m
        
    def power_law_profile(self, altitude, reference_wind_speed, reference_altitude=10.0):
        """Generate wind profile using power law."""
        if altitude <= reference_altitude:
            return reference_wind_speed * (altitude / reference_altitude)**self.power_law_exponent
        else:
            return reference_wind_speed * (altitude / reference_altitude)**self.power_law_exponent
    
    def generate_stochastic_profile(self, altitudes, base_wind_speed, base_wind_direction=0.0, 
                                   random_state=None):
        """Generate stochastic wind profile with turbulence."""
        if random_state is None:
            random_state = np.random.RandomState()
        
        n_points = len(altitudes)
        wind_profile = np.zeros((n_points, 3))  # [u, v, w] components
        
        # Base wind components
        base_u = base_wind_speed * np.cos(base_wind_direction)
        base_v = base_wind_speed * np.sin(base_wind_direction)
        
        for i, altitude in enumerate(altitudes):
            # Power law base wind
            wind_speed = self.power_law_profile(altitude, base_wind_speed)
            wind_u = wind_speed * np.cos(base_wind_direction)
            wind_v = wind_speed * np.sin(base_wind_direction)
            
            # Add turbulence (decreases with altitude)
            turbulence_scale = self.turbulence_intensity * np.exp(-altitude / 2000.0)
            
            # Correlated turbulence (simple model)
            if i > 0:
                correlation_factor = np.exp(-(altitudes[i] - altitudes[i-1]) / 
                                          self.correlation_length)
                
                wind_profile[i, 0] = wind_u + correlation_factor * wind_profile[i-1, 0] + \
                                   random_state.normal(0, turbulence_scale * np.sqrt(1 - correlation_factor**2))
                wind_profile[i, 1] = wind_v + correlation_factor * wind_profile[i-1, 1] + \
                                   random_state.normal(0, turbulence_scale * np.sqrt(1 - correlation_factor**2))
                wind_profile[i, 2] = correlation_factor * wind_profile[i-1, 2] + \
                                   random_state.normal(0, turbulence_scale * np.sqrt(1 - correlation_factor**2))
            else:
                wind_profile[i, 0] = wind_u + random_state.normal(0, turbulence_scale)
                wind_profile[i, 1] = wind_v + random_state.normal(0, turbulence_scale)
                wind_profile[i, 2] = random_state.normal(0, turbulence_scale)
        
        return wind_profile
    
    def get_wind_at_altitude(self, altitude, wind_profile, altitude_profile):
        """Interpolate wind from wind profile at given altitude."""
        if len(wind_profile) == 0:
            return np.array([0.0, 0.0, 0.0])
        
        # Interpolate each component
        wind_u = interpolate_1d(altitude, altitude_profile, wind_profile[:, 0])
        wind_v = interpolate_1d(altitude, altitude_profile, wind_profile[:, 1])
        wind_w = interpolate_1d(altitude, altitude_profile, wind_profile[:, 2])
        
        return np.array([wind_u, wind_v, wind_w]) 