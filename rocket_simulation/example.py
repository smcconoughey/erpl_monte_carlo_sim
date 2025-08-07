import numpy as np
import matplotlib.pyplot as plt
import os
from rocket import Rocket
from motor import SolidMotor, LiquidMotor
from environment import StandardAtmosphere, WindModel
from simulator import FlightSimulator
from monte_carlo import MonteCarloAnalyzer

def main():
    print("6DOF Monte Carlo Rocket Simulation")
    print("=" * 50)

    rocket = Rocket("Sounding Rocket")
    motor = LiquidMotor("Liquid Motor")
    atmosphere = StandardAtmosphere()
    wind_model = WindModel()

    wind_file = os.path.join(os.path.dirname(__file__), 'sample_wind.csv')
    altitude_profile, wind_profile = wind_model.load_wind_profile_from_csv(wind_file)

    simulator = FlightSimulator(rocket, motor, atmosphere, wind_model)

    initial_conditions = {
        'position': [0.0, 0.0, 10.0],
        'velocity': [0, 0, 0.0],
        'attitude': [0.0, -np.pi/2 + 0.02, 0.0],
        'angular_velocity': [0.0, 0.0, 0.0]
    }

    print("\nRunning single simulation...")
    results = simulator.simulate_flight(initial_conditions, wind_profile, altitude_profile)

    print(f"Rail exit speed: {results['rail_exit_speed']:.2f} m/s")
    print(
        f"Rail AoA: {np.degrees(results['rail_exit_angle_of_attack']):.2f} deg, "
        f"sideslip: {np.degrees(results['rail_exit_sideslip']):.2f} deg"
    )
    print(f"Apogee altitude: {results['apogee_altitude']:.1f} m ({results['apogee_altitude']*3.28084:.1f} ft)")
    print(f"Range: {results['range']:.1f} m")
    print(f"Flight time: {results['flight_time']:.1f} s")

    print("\nRunning Monte Carlo analysis...")
    monte_carlo = MonteCarloAnalyzer(rocket, motor, atmosphere, wind_model)
    monte_carlo.base_altitude_profile = altitude_profile
    monte_carlo.base_wind_profile = wind_profile
    mc_results = monte_carlo.run_monte_carlo(initial_conditions, n_samples=50)

    output_dir = monte_carlo.plot_results(mc_results)
    monte_carlo.plot_trajectory_cloud_3d(mc_results, save_plots=True)

    return results, mc_results

if __name__ == "__main__":
    main()

