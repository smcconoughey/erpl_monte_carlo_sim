#!/usr/bin/env python3
"""
Simple test to validate atmospheric model fixes without requiring SciPy
"""

import numpy as np

class StandardAtmosphere:
    """Simplified copy of the fixed atmospheric model for testing."""
    
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
            # Extended atmosphere - improved model for stratosphere/mesosphere
            # Use proper U.S. Standard Atmosphere 1976 equations
            if altitude <= 32000.0:  # Upper stratosphere
                # Linear temperature increase from 20-32 km
                temperature = self.stratosphere_temp + 0.001 * (altitude - self.stratosphere_height)
                temperature = min(temperature, 228.65)  # Cap at mesosphere base temp
                
                # Proper barometric formula for this layer
                pressure_20km = self.sea_level_pressure * (
                    self.stratosphere_temp / self.sea_level_temperature
                )**(self.gravity / (self.gas_constant * self.temperature_lapse_rate))
                pressure_20km *= np.exp(
                    -self.gravity * (self.stratosphere_height - self.troposphere_height) / 
                    (self.gas_constant * self.stratosphere_temp)
                )
                
                if altitude <= 25000.0:
                    # Isothermal layer continuation 20-25 km
                    pressure = pressure_20km * np.exp(
                        -self.gravity * (altitude - self.stratosphere_height) / 
                        (self.gas_constant * self.stratosphere_temp)
                    )
                else:
                    # Temperature gradient layer 25-32 km
                    pressure_25km = pressure_20km * np.exp(
                        -self.gravity * 5000.0 / 
                        (self.gas_constant * self.stratosphere_temp)
                    )
                    temp_gradient = 0.0028  # K/m temperature gradient
                    temp_25km = self.stratosphere_temp
                    
                    pressure = pressure_25km * (
                        temperature / temp_25km
                    )**(self.gravity / (self.gas_constant * temp_gradient))
            else:
                # Mesosphere (32-50 km) - realistic but still simplified
                temperature = 228.65 - 0.0028 * (altitude - 32000.0)
                temperature = max(temperature, 180.0)  # Don't go below realistic minimum
                
                # Exponential decay but with realistic scale height
                scale_height = self.gas_constant * temperature / self.gravity
                pressure_32km = 868.02  # Pa at 32 km from standard atmosphere
                pressure = pressure_32km * np.exp(-(altitude - 32000.0) / scale_height)
            
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

def test_atmospheric_model():
    """Test the atmospheric model improvements"""
    print("Testing Fixed Atmospheric Model")
    print("=" * 50)
    
    atmosphere = StandardAtmosphere()
    
    # Test altitudes from sea level to 50 km
    test_altitudes = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    
    print(f"{'Alt (km)':>8} {'Pressure (Pa)':>12} {'Density (kg/m³)':>15} {'Temp (K)':>10}")
    print("-" * 50)
    
    previous_pressure = float('inf')
    previous_density = float('inf')
    
    for alt in test_altitudes:
        props = atmosphere.get_properties(alt)
        
        print(f"{alt/1000:8.1f} {props['pressure']:12.1f} {props['density']:15.6f} {props['temperature']:10.1f}")
        
        # Check that pressure and density decrease with altitude
        assert props['pressure'] < previous_pressure, f"Pressure should decrease with altitude at {alt}m"
        assert props['density'] < previous_density, f"Density should decrease with altitude at {alt}m"
        
        # Check for reasonable bounds
        assert props['pressure'] > 0, f"Pressure should be positive at {alt}m"
        assert props['density'] > 0, f"Density should be positive at {alt}m"
        assert props['temperature'] > 100, f"Temperature should be reasonable at {alt}m"
        
        previous_pressure = props['pressure']
        previous_density = props['density']
    
    print("\n✓ Atmospheric model validation passed!")
    
    # Compare old vs new model at critical altitudes
    print("\nCritical altitude comparisons:")
    
    # At 25 km (where problems typically started)
    props_25km = atmosphere.get_properties(25000)
    print(f"25 km: P = {props_25km['pressure']:.1f} Pa, ρ = {props_25km['density']:.6f} kg/m³")
    
    # At 40 km (where old model became near-vacuum)
    props_40km = atmosphere.get_properties(40000)
    print(f"40 km: P = {props_40km['pressure']:.1f} Pa, ρ = {props_40km['density']:.6f} kg/m³")
    
    # Check that we have realistic density at high altitudes
    assert props_40km['density'] > 1e-6, "Density at 40km should not be near zero"
    assert props_40km['pressure'] > 1.0, "Pressure at 40km should be reasonable"
    
    print("\n✓ High altitude model improvements validated!")
    
    # Calculate dynamic pressure at various altitudes for a 300 m/s rocket
    print("\nDynamic pressure for 300 m/s velocity:")
    velocity = 300.0  # m/s
    
    for alt in [20000, 30000, 40000]:
        props = atmosphere.get_properties(alt)
        q_dynamic = 0.5 * props['density'] * velocity**2
        print(f"{alt/1000:4.1f} km: q = {q_dynamic:8.1f} Pa (density = {props['density']:.6f} kg/m³)")
        
        # Ensure there's still meaningful drag at high altitudes
        if alt == 40000:
            assert q_dynamic > 100, "Should still have meaningful dynamic pressure at 40km"
    
    print("\n" + "="*50)
    print("ATMOSPHERIC MODEL FIX VALIDATION: PASSED ✓")
    print("="*50)

if __name__ == "__main__":
    test_atmospheric_model()