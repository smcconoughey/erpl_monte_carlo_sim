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
    
    def generate_stochastic_profile(self, altitudes, base_wind_speed,
                                   base_wind_direction=None, random_state=None):
        """Generate stochastic wind profile with turbulence.

        Parameters
        ----------
        altitudes : array-like
            Altitudes at which to generate the wind profile.
        base_wind_speed : float
            Reference wind speed at the surface (m/s).
        base_wind_direction : float or None, optional
            Mean wind direction in radians. If ``None`` (default), the
            direction is chosen randomly from 0 to ``2π``.
        random_state : ``np.random.RandomState`` or ``None``
            Random state for reproducibility.
        """

        if random_state is None:
            random_state = np.random.RandomState()

        if base_wind_direction is None:
            base_wind_direction = random_state.uniform(0.0, 2 * np.pi)
        
        n_points = len(altitudes)
        wind_profile = np.zeros((n_points, 3))  # [u, v, w] components
        
        # Base wind components
        base_u = base_wind_speed * np.cos(base_wind_direction)
        base_v = base_wind_speed * np.sin(base_wind_direction)
        
        # Initialize first point separately to avoid correlation artifacts
        altitude = altitudes[0]
        wind_speed = self.power_law_profile(altitude, base_wind_speed)
        wind_u = wind_speed * np.cos(base_wind_direction)
        wind_v = wind_speed * np.sin(base_wind_direction)
        
        turbulence_scale = self.turbulence_intensity * np.exp(-altitude / 2000.0)
        wind_profile[0, 0] = wind_u + random_state.normal(0, turbulence_scale)
        wind_profile[0, 1] = wind_v + random_state.normal(0, turbulence_scale)
        wind_profile[0, 2] = random_state.normal(0, turbulence_scale * 0.3)  # Reduced vertical turbulence
        
        # Generate remaining points with proper correlation
        for i in range(1, n_points):
            altitude = altitudes[i]
            
            # Power law base wind
            wind_speed = self.power_law_profile(altitude, base_wind_speed)
            wind_u = wind_speed * np.cos(base_wind_direction)
            wind_v = wind_speed * np.sin(base_wind_direction)
            
            # Add turbulence (decreases with altitude)
            turbulence_scale = self.turbulence_intensity * np.exp(-altitude / 2000.0)
            
            # Correlated turbulence with improved stability
            altitude_diff = max(altitudes[i] - altitudes[i-1], 1e-6)  # Prevent division by zero
            correlation_factor = np.exp(-altitude_diff / self.correlation_length)
            correlation_factor = np.clip(correlation_factor, 0.1, 0.95)  # Limit correlation range
            
            # Separate turbulence components from mean wind
            prev_turbulence_u = wind_profile[i-1, 0] - (self.power_law_profile(altitudes[i-1], base_wind_speed) * np.cos(base_wind_direction))
            prev_turbulence_v = wind_profile[i-1, 1] - (self.power_law_profile(altitudes[i-1], base_wind_speed) * np.sin(base_wind_direction))
            prev_turbulence_w = wind_profile[i-1, 2]
            
            # Apply correlation to turbulence components only
            turbulence_variance = turbulence_scale * np.sqrt(max(1 - correlation_factor**2, 0.01))
            
            new_turbulence_u = correlation_factor * prev_turbulence_u + random_state.normal(0, turbulence_variance)
            new_turbulence_v = correlation_factor * prev_turbulence_v + random_state.normal(0, turbulence_variance)
            new_turbulence_w = correlation_factor * prev_turbulence_w + random_state.normal(0, turbulence_variance * 0.3)
            
            # Combine mean wind with turbulence
            wind_profile[i, 0] = wind_u + new_turbulence_u
            wind_profile[i, 1] = wind_v + new_turbulence_v
            wind_profile[i, 2] = new_turbulence_w
        
        return wind_profile

    def load_wind_profile_from_csv(self, file_path):
        """Load a wind profile from a CSV file.

        The file should contain columns ``altitude``, ``u``, ``v`` and
        optionally ``w`` in meters and meters per second.  This helper enables
        running the simulator with realistic forecasts exported from external
        tools such as NOAA's GFS or NAM models.
        """
        data = np.genfromtxt(file_path, delimiter=',', names=True)
        altitudes = data['altitude']
        if 'w' in data.dtype.names:
            wind = np.vstack([data['u'], data['v'], data['w']]).T
        else:
            wind = np.vstack([data['u'], data['v'], np.zeros_like(altitudes)]).T
        return altitudes, wind
    
    def get_wind_at_altitude(self, altitude, wind_profile, altitude_profile):
        """Interpolate wind from wind profile at given altitude."""
        if len(wind_profile) == 0:
            return np.array([0.0, 0.0, 0.0])
        
        # Interpolate each component
        wind_u = interpolate_1d(altitude, altitude_profile, wind_profile[:, 0])
        wind_v = interpolate_1d(altitude, altitude_profile, wind_profile[:, 1])
        wind_w = interpolate_1d(altitude, altitude_profile, wind_profile[:, 2])
                return np.array([wind_u, wind_v, wind_w]) 