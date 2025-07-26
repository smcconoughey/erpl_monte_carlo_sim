#!/usr/bin/env python3
"""
Test script to validate the rocket simulator fixes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'rocket_simulation'))

import numpy as np
from rocket_simulation.environment import StandardAtmosphere, WindModel
from rocket_simulation.rocket import Rocket
from rocket_simulation.motor import SolidMotor
from rocket_simulation.simulator import FlightSimulator
from rocket_simulation.monte_carlo import MonteCarloAnalyzer
from rocket_simulation.utils import *

def test_atmospheric_model():
    """Test the improved atmospheric model"""
    print("Testing atmospheric model fixes...")
    
    atmosphere = StandardAtmosphere()
    
    # Test altitudes
    test_altitudes = [15000, 20000, 25000, 30000, 35000, 40000, 50000]
    
    for alt in test_altitudes:
        props = atmosphere.get_properties(alt)
        print(f"Alt: {alt/1000:5.1f} km, P: {props['pressure']:8.1f} Pa, ρ: {props['density']:.6f} kg/m³, T: {props['temperature']:6.1f} K")
    
    # Check for reasonable values
    props_20km = atmosphere.get_properties(20000)
    props_30km = atmosphere.get_properties(30000)
    props_40km = atmosphere.get_properties(40000)
    
    assert props_20km['pressure'] > props_30km['pressure'] > props_40km['pressure'], "Pressure should decrease with altitude"
    assert props_40km['density'] > 1e-6, "Density at 40km should not be near zero"
    print("✓ Atmospheric model improvements validated\n")

def test_single_simulation():
    """Test a single simulation with nominal conditions"""
    print("Testing single nominal simulation...")
    
    # Create rocket and motor
    rocket = Rocket("Test Rocket")
    motor = SolidMotor()
    atmosphere = StandardAtmosphere()
    wind_model = WindModel()
    
    # Create simulator
    simulator = FlightSimulator(rocket, motor, atmosphere, wind_model)
    
    # Nominal initial conditions
    initial_conditions = {
        'position': [0.0, 0.0, 0.0],
        'velocity': [0.0, 0.0, 0.0],
        'attitude': [0.0, 0.0, 0.0],  # Vertical launch
        'angular_velocity': [0.0, 0.0, 0.0]
    }
    
    # Run simulation
    results = simulator.simulate_flight(initial_conditions)
    
    apogee = results['apogee_altitude']
    range_val = results['range']
    flight_time = results['flight_time']
    
    print(f"Nominal simulation results:")
    print(f"  Apogee: {apogee:.1f} m ({apogee*3.28084:.1f} ft)")
    print(f"  Range: {range_val:.1f} m")
    print(f"  Flight time: {flight_time:.1f} s")
    
    # Check for reasonable values
    assert 10000 < apogee < 30000, f"Apogee {apogee:.1f} m should be between 10-30 km"
    assert range_val < 10000, f"Range {range_val:.1f} m should be small for vertical launch"
    assert 100 < flight_time < 400, f"Flight time {flight_time:.1f} s should be reasonable"
    
    print("✓ Single simulation produces reasonable results\n")
    
    return results

def test_monte_carlo_sample():
    """Test Monte Carlo with small sample"""
    print("Testing Monte Carlo analysis with outlier filtering...")
    
    # WindModel already imported
    
    # Create components
    rocket = Rocket("Test Rocket")
    motor = SolidMotor()
    atmosphere = StandardAtmosphere()
    wind_model = WindModel()
    
    # Create Monte Carlo analyzer
    mc_analyzer = MonteCarloAnalyzer(rocket, motor, atmosphere, wind_model)
    
    # Nominal initial conditions
    initial_conditions = {
        'position': [0.0, 0.0, 0.0],
        'velocity': [0.0, 0.0, 0.0],
        'attitude': [0.0, 0.05, 0.0],  # Slight tilt for dispersion
        'angular_velocity': [0.0, 0.0, 0.0]
    }
    
    # Run small Monte Carlo
    analysis = mc_analyzer.run_monte_carlo(initial_conditions, n_samples=10, optimized=False)
    
    print(f"Monte Carlo results (n={analysis['n_samples']}):")
    print(f"  Valid simulations: {analysis['n_samples']}")
    print(f"  Failed simulations: {analysis['n_failed']}")
    print(f"  Outlier simulations: {analysis['n_outliers']}")
    print(f"  Apogee range: {analysis['apogee_altitude']['min']:.1f} - {analysis['apogee_altitude']['max']:.1f} m")
    print(f"  Median apogee: {analysis['apogee_altitude']['percentiles'][2]:.1f} m")
    
    # Check that outlier filtering worked
    max_apogee = analysis['apogee_altitude']['max']
    assert max_apogee < 80000, f"Max apogee {max_apogee:.1f} m should be below 80 km limit"
    
    print("✓ Monte Carlo analysis with outlier filtering working\n")
    
    return analysis

if __name__ == "__main__":
    print("=" * 60)
    print("ROCKET SIMULATOR FIXES VALIDATION TEST")
    print("=" * 60)
    
    try:
        # Test 1: Atmospheric model
        test_atmospheric_model()
        
        # Test 2: Single simulation
        nominal_results = test_single_simulation()
        
        # Test 3: Monte Carlo sample
        mc_results = test_monte_carlo_sample()
        
        print("=" * 60)
        print("✓ ALL TESTS PASSED - Fixes successfully implemented!")
        print("=" * 60)
        
        print("\nKey improvements:")
        print("1. ✓ Fixed atmospheric model discontinuity above 20km")
        print("2. ✓ Added apogee detection and altitude bounds checking")
        print("3. ✓ Implemented physics-based outlier filtering")
        print("4. ✓ Reduced Monte Carlo parameter extremes to realistic ranges")
        print("5. ✓ Added early termination for excessive altitudes")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)