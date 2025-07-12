#!/usr/bin/env python3
"""Test script to verify the Monte Carlo simulation fixes"""

import numpy as np
from rocket import Rocket
from motor import LiquidMotor
from environment import StandardAtmosphere, WindModel
from simulator import FlightSimulator
from monte_carlo import MonteCarloAnalyzer

def test_simulation():
    """Test the fixed simulation with edge cases"""
    print("Testing Monte Carlo simulation with edge case handling...")
    
    # Create rocket configuration
    rocket = Rocket("Test Rocket")
    motor = LiquidMotor("Test Motor")
    atmosphere = StandardAtmosphere()
    wind_model = WindModel()
    
    # Create simulator
    simulator = FlightSimulator(rocket, motor, atmosphere, wind_model)
    
    # Test single simulation with extreme launch angle to trigger edge case
    initial_conditions = {
        'position': [0.0, 0.0, 10.0],
        'velocity': [0.0, 0.0, 0.0],
        'attitude': [0.0, -np.pi/4, 0.0],  # 45 degree launch angle (might go far)
        'angular_velocity': [0.0, 0.0, 0.0]
    }
    
    print("\nRunning single simulation with 45° launch angle...")
    results = simulator.simulate_flight(initial_conditions)
    
    print(f"Apogee: {results['apogee_altitude']:.1f} m")
    print(f"Range: {results['range']:.1f} m")
    print(f"Flight time: {results['flight_time']:.1f} s")
    print(f"Final altitude: {results['altitude'][-1]:.1f} m")
    
    # Run small Monte Carlo test
    monte_carlo = MonteCarloAnalyzer(rocket, motor, atmosphere, wind_model)
    
    # Use larger wind variations to trigger some edge cases
    monte_carlo.uncertainty_params['wind_speed_range'] = [0.0, 25.0]
    monte_carlo.uncertainty_params['initial_attitude'] = [0.05, 0.05, 0.05]  # Larger attitude variations
    
    print("\nRunning Monte Carlo with 10 samples...")
    mc_results = monte_carlo.run_monte_carlo(initial_conditions, n_samples=10)
    
    print(f"\nMonte Carlo completed with {mc_results['n_samples']} valid results")
    print(f"Failed simulations: {mc_results['n_failed']}")
    
    # Try to plot (this was causing the error)
    print("\nAttempting to plot results...")
    try:
        output_dir = monte_carlo.plot_results(mc_results, save_plots=True)
        if output_dir:
            print(f"Plots saved successfully to: {output_dir}")
        else:
            print("No valid data to plot")
    except Exception as e:
        print(f"Error during plotting: {e}")
        import traceback
        traceback.print_exc()
    
    return mc_results

if __name__ == "__main__":
    test_simulation()