import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from .rocket import Rocket
from .motor import SolidMotor, LiquidMotor
from .environment import StandardAtmosphere, WindModel
from .simulator import FlightSimulator
from .monte_carlo import MonteCarloAnalyzer


def run_single_flight(simulator: FlightSimulator,
                      initial_conditions,
                      wind_profile=None,
                      altitude_profile=None,
                      output_dir=None):
    results = simulator.simulate_flight(initial_conditions, wind_profile, altitude_profile)

    # Plot basic single-flight outputs
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(results['time'], results['altitude'])
        axes[0, 0].set_title('Altitude vs Time')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Altitude (m)')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(results['time'], results['speed'])
        axes[0, 1].set_title('Speed vs Time')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Speed (m/s)')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(results['position'][0, :], results['position'][2, :])
        axes[1, 0].set_title('Downrange vs Altitude')
        axes[1, 0].set_xlabel('Downrange (m)')
        axes[1, 0].set_ylabel('Altitude (m)')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(results['time'], np.degrees(results['angle_of_attack']))
        axes[1, 1].set_title('Angle of Attack')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('deg')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'single_simulation_results.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return results


def main():
    parser = argparse.ArgumentParser(description='Run 6DOF rocket flight and Monte Carlo analysis')
    parser.add_argument('--motor', choices=['solid', 'liquid'], default='liquid', help='Motor type')
    parser.add_argument('--wind-csv', type=str, default=os.path.join(os.path.dirname(__file__), 'sample_wind.csv'), help='CSV with altitude,u,v,w')
    parser.add_argument('--no-wind', action='store_true', help='Disable wind')
    parser.add_argument('--mc-samples', type=int, default=50, help='Monte Carlo samples')
    parser.add_argument('--output-root', type=str, default=None, help='Optional output directory root')
    parser.add_argument('--tilt-deg', type=float, default=1.0, help='Initial tilt from vertical (deg)')
    parser.add_argument('--alt0', type=float, default=10.0, help='Initial altitude (m)')
    args = parser.parse_args()

    # Components
    rocket = Rocket('Sounding Rocket')
    motor = LiquidMotor('Liquid Motor') if args.motor == 'liquid' else SolidMotor('Solid Motor')
    atmosphere = StandardAtmosphere()
    wind_model = WindModel()

    # Wind profile
    altitude_profile = None
    wind_profile = None
    if not args.no_wind and os.path.isfile(args.wind_csv):
        altitude_profile, wind_profile = wind_model.load_wind_profile_from_csv(args.wind_csv)

    # Simulator
    simulator = FlightSimulator(rocket, motor, atmosphere, wind_model)

    # Initial conditions: body x-axis up with small tilt
    tilt_rad = np.radians(args.tilt_deg)
    initial_conditions = {
        'position': [0.0, 0.0, float(args.alt0)],
        'velocity': [0.0, 0.0, 0.0],
        'attitude': [0.0, -np.pi/2 + tilt_rad, 0.0],
        'angular_velocity': [0.0, 0.0, 0.0],
    }

    # Single simulation
    results, mc_results = None, None
    print('\nRunning single simulation...')
    results = run_single_flight(simulator, initial_conditions, wind_profile, altitude_profile, args.output_root)
    print(f"Apogee: {results['apogee_altitude']:.1f} m, Range: {results['range']:.1f} m, Flight time: {results['flight_time']:.1f} s")

    # Monte Carlo
    if args.mc_samples and args.mc_samples > 0:
        print('\nRunning Monte Carlo analysis...')
        mc = MonteCarloAnalyzer(rocket, motor, atmosphere, wind_model)
        if altitude_profile is not None and wind_profile is not None:
            mc.base_altitude_profile = altitude_profile
            mc.base_wind_profile = wind_profile
        analysis = mc.run_monte_carlo(initial_conditions, n_samples=int(args.mc_samples))
        out_dir = mc.plot_results(analysis)
        mc.plot_trajectory_cloud_3d(analysis, save_plots=True, max_trajectories=min(50, len(analysis['results'])))
        print(f"Outputs written to: {out_dir}")
        mc_results = analysis

    return results, mc_results


if __name__ == '__main__':
    main()

