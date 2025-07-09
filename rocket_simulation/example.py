import numpy as np
import matplotlib.pyplot as plt
import os
from rocket import Rocket
from motor import SolidMotor, LiquidMotor
from environment import StandardAtmosphere, WindModel
from simulator import FlightSimulator
from monte_carlo import MonteCarloAnalyzer

def main():
    """Run example rocket simulation."""
    print("6DOF Monte Carlo Rocket Simulation")
    print("=" * 50)
    
    # Create rocket configuration
    rocket = Rocket("Sounding Rocket")
    
    # Create motor (liquid in this example)
    motor = LiquidMotor("Liquid Motor")
    
    # Create environment models
    atmosphere = StandardAtmosphere()
    wind_model = WindModel()

    # Load a realistic wind profile from CSV (e.g. exported from a weather
    # model).  The provided sample is simplistic but demonstrates the API.
    wind_file = os.path.join(os.path.dirname(__file__), 'sample_wind.csv')
    altitude_profile, wind_profile = wind_model.load_wind_profile_from_csv(wind_file)
    
    # Create simulator
    simulator = FlightSimulator(rocket, motor, atmosphere, wind_model)
    
    # Define initial conditions for vertical launch
    initial_conditions = {
        'position': [0.0, 0.0, 10.0],  # Launch from 10m above ground
        'velocity': [0, 0, 0.0],  # Start from rest
        'attitude': [0.0, -np.pi/2 + 0.02, 0.0],  # Nearly vertical (-90° + small angle)
        'angular_velocity': [0.0, 0.0, 0.0]  # No initial rotation
    }
    
    # Run single simulation
    print("\nRunning single simulation...")
    results = simulator.simulate_flight(initial_conditions, wind_profile, altitude_profile)
    
    # Print results
    print(f"Rail exit speed: {results['rail_exit_speed']:.2f} m/s")
    print(
        f"Rail AoA: {np.degrees(results['rail_exit_angle_of_attack']):.2f} deg, "
        f"sideslip: {np.degrees(results['rail_exit_sideslip']):.2f} deg"
    )
    print(f"Apogee altitude: {results['apogee_altitude']:.1f} m ({results['apogee_altitude']*3.28084:.1f} ft)")
    print(f"Range: {results['range']:.1f} m")
    print(f"Flight time: {results['flight_time']:.1f} s")
    
    # Run Monte Carlo analysis
    print("\nRunning Monte Carlo analysis...")
    monte_carlo = MonteCarloAnalyzer(rocket, motor, atmosphere, wind_model)
    monte_carlo.base_altitude_profile = altitude_profile
    monte_carlo.base_wind_profile = wind_profile
    # Each Monte Carlo run will perturb this forecast using the sampled wind
    # parameters and turbulence model
    
    # Run with fewer samples for example (increase for production)
    mc_results = monte_carlo.run_monte_carlo(initial_conditions, n_samples=50)
    
    # Plot Monte Carlo results and get output directory
    output_dir = monte_carlo.plot_results(mc_results)
    monte_carlo.plot_trajectory_cloud_3d(mc_results, save_plots=True)

    # Plot single simulation results to the same directory
    plot_single_simulation(results, output_dir)
    
    return results, mc_results

def plot_single_simulation(results, output_dir=None):
    """Plot results from single simulation."""
    # Create output directory if it doesn't exist
    if output_dir is None:
        output_dir = "simulation_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    
    # Altitude vs time
    axes[0, 0].plot(results['time'], results['altitude'])
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Altitude (m)')
    axes[0, 0].set_title('Altitude vs Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Velocity vs time
    axes[0, 1].plot(results['time'], results['speed'])
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Speed (m/s)')
    axes[0, 1].set_title('Speed vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Trajectory (x-z plane)
    axes[1, 0].plot(results['position'][0, :], results['position'][2, :])
    axes[1, 0].set_xlabel('Downrange (m)')
    axes[1, 0].set_ylabel('Altitude (m)')
    axes[1, 0].set_title('Trajectory')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Euler angles
    axes[1, 1].plot(results['time'], np.degrees(results['euler_angles'][0, :]), label='Roll')
    axes[1, 1].plot(results['time'], np.degrees(results['euler_angles'][1, :]), label='Pitch')
    axes[1, 1].plot(results['time'], np.degrees(results['euler_angles'][2, :]), label='Yaw')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Angle (degrees)')
    axes[1, 1].set_title('Euler Angles')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Center of pressure and CG over time
    axes[2, 0].plot(results['time'], results['center_of_mass'], label='CG')
    axes[2, 0].plot(results['time'], results['cp_location_dynamic'], '--', color='r', label='CP')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Position along body (m)')
    axes[2, 0].set_title('CP and CG vs Time')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Angle of attack over time
    axes[2, 1].plot(results['time'], np.degrees(results['angle_of_attack']))
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('Angle (deg)')
    axes[2, 1].set_title('Angle of Attack vs Time')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot instead of showing it
    plot_filename = os.path.join(output_dir, "single_simulation_results.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Single simulation plot saved to: {plot_filename}")
    plt.close()

    # Plot thrust and thrust-to-weight ratio
    atmosphere = StandardAtmosphere()
    gravity = np.array([atmosphere.get_gravity(alt) for alt in results['altitude']])
    twr = results['thrust'] / (results['mass'] * gravity)

    fig2, axes2 = plt.subplots(2, 1, figsize=(8, 8))
    axes2[0].plot(results['time'], results['thrust'])
    axes2[0].set_xlabel('Time (s)')
    axes2[0].set_ylabel('Thrust (N)')
    axes2[0].set_title('Thrust vs Time')
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(results['time'], twr)
    axes2[1].set_xlabel('Time (s)')
    axes2[1].set_ylabel('TWR')
    axes2[1].set_title('Thrust-to-Weight Ratio vs Time')
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    twr_plot = os.path.join(output_dir, 'thrust_twr.png')
    plt.savefig(twr_plot, dpi=300, bbox_inches='tight')
    print(f"Thrust/TWR plot saved to: {twr_plot}")
    plt.close()

if __name__ == "__main__":    results, mc_results = main() 

