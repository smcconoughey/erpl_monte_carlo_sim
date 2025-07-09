"""
Monte Carlo analysis framework
"""

import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from functools import partial
import time
from datetime import datetime
import json
from simulator import FlightSimulator
from utils import *

class MonteCarloAnalyzer:
    """Monte Carlo analysis for rocket simulation."""
    
    def __init__(self, rocket, motor, atmosphere, wind_model):
        self.rocket = rocket
        self.motor = motor
        self.atmosphere = atmosphere
        self.wind_model = wind_model
        self.n_cores = os.cpu_count()

        # Optional externally supplied wind profile (altitude array and
        # corresponding wind vectors).  When provided, all Monte Carlo samples
        # will use this deterministic profile instead of generating a new one
        # each iteration.
        self.base_altitude_profile = None
        self.base_wind_profile = None
        
        # Uncertainty parameters
        self.uncertainty_params = {
            'initial_position': [0.0, 0.0, 0.0],  # Standard deviation (m)
            'initial_velocity': [0.1, 0.1, 0.1],  # Standard deviation (m/s)
            'initial_attitude': [0.01, 0.01, 0.01],  # Standard deviation (rad)
            'initial_angular_velocity': [0.01, 0.01, 0.01],  # Standard deviation (rad/s)
            'mass_uncertainty': 0.02,  # 2% standard deviation
            'thrust_uncertainty': 0.05,  # 5% standard deviation
            'wind_speed_range': [0.0, 15.0],  # Wind speed range (m/s)
            'wind_direction_range': [0.0, 2*np.pi],  # Wind direction range (rad)
            'atmospheric_density_uncertainty': 0.1  # 10% standard deviation
        }
        
        print(f"Initialized Monte Carlo analyzer with {self.n_cores} cores")
        
    def run_monte_carlo(self, initial_conditions, n_samples=1000, n_processes=None, optimized=False):
        """Run Monte Carlo analysis with optional optimization."""
        if optimized:
            return self.run_optimized_monte_carlo(initial_conditions, n_samples)
        
        print(f"Running Monte Carlo analysis with {n_samples} samples...")
        
        # Generate parameter samples
        parameter_samples = self._generate_parameter_samples(n_samples)
        
        # Run simulations
        if n_processes is None:
            n_processes = min(self.n_cores, n_samples)
        
        results = []
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            futures = []
            
            for i in range(n_samples):
                params = parameter_samples[i]
                future = executor.submit(self._run_single_simulation, initial_conditions, params, i)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    if len(results) % 100 == 0:
                        print(f"Completed {len(results)}/{n_samples} simulations")
                except Exception as e:
                    print(f"Simulation failed: {e}")
        
        print(f"Completed {len(results)} out of {n_samples} simulations")
        
        # Analyze results
        analysis = self._analyze_results(results)
        
        return analysis
    
    def run_optimized_monte_carlo(self, initial_conditions, n_samples=1000, chunk_size=None):
        """Run highly optimized Monte Carlo analysis."""
        print(f"Running optimized Monte Carlo with {n_samples} samples on {self.n_cores} cores")
        
        start_time = time.time()
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, n_samples // (self.n_cores * 8))  # 8 chunks per core
        
        # Generate all parameters vectorized
        parameter_samples = self._generate_parameter_samples_vectorized(n_samples)
        
        # Split into chunks for processing
        chunks = [parameter_samples[i:i + chunk_size] for i in range(0, n_samples, chunk_size)]
        
        print(f"Processing {len(chunks)} chunks of size ~{chunk_size}")
        
        # Use multiprocessing with shared memory optimization
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # Create partial function with fixed arguments
            worker_func = partial(
                self._process_chunk,
                initial_conditions=initial_conditions,
                base_analyzer=self
            )
            
            # Submit all chunks
            futures = [executor.submit(worker_func, chunk, chunk_id) 
                      for chunk_id, chunk in enumerate(chunks)]
            
            # Collect results
            all_results = []
            completed_simulations = 0
            
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    all_results.extend(chunk_results)
                    completed_simulations += len(chunk_results)
                    
                    if completed_simulations % 500 == 0:
                        elapsed = time.time() - start_time
                        rate = completed_simulations / elapsed
                        print(f"Completed {completed_simulations}/{n_samples} simulations "
                              f"({rate:.1f} sims/sec)")
                        
                except Exception as e:
                    print(f"Chunk processing failed: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"Completed {len(all_results)} simulations in {elapsed_time:.2f} seconds "
              f"({len(all_results)/elapsed_time:.1f} sims/sec)")
        
        # Analyze results
        analysis = self._analyze_results(all_results)
        analysis['performance'] = {
            'total_time': elapsed_time,
            'simulations_per_second': len(all_results) / elapsed_time,
            'cores_used': self.n_cores
        }
        
        return analysis
    
    def _generate_parameter_samples(self, n_samples):
        """Generate parameter samples for Monte Carlo analysis."""
        samples = []
        
        for i in range(n_samples):
            # Set random seed for reproducibility
            np.random.seed(i)
            
            sample = {
                'initial_position_offset': np.random.normal(0, self.uncertainty_params['initial_position']),
                'initial_velocity_offset': np.random.normal(0, self.uncertainty_params['initial_velocity']),
                'initial_attitude_offset': np.random.normal(0, self.uncertainty_params['initial_attitude']),
                'initial_angular_velocity_offset': np.random.normal(0, self.uncertainty_params['initial_angular_velocity']),
                'mass_multiplier': np.random.normal(1.0, self.uncertainty_params['mass_uncertainty']),
                'thrust_multiplier': np.random.normal(1.0, self.uncertainty_params['thrust_uncertainty']),
                'wind_speed': np.random.uniform(*self.uncertainty_params['wind_speed_range']),
                'wind_direction': np.random.uniform(*self.uncertainty_params['wind_direction_range']),
                'density_multiplier': np.random.normal(1.0, self.uncertainty_params['atmospheric_density_uncertainty']),
                'random_seed': i
            }
            
            samples.append(sample)
        
        return samples
    
    def _generate_parameter_samples_vectorized(self, n_samples):
        """Generate parameter samples using vectorized operations."""
        np.random.seed(42)  # For reproducible results
        
        samples = []
        for i in range(n_samples):
            sample = {
                'initial_position_offset': np.random.normal(0, self.uncertainty_params['initial_position']),
                'initial_velocity_offset': np.random.normal(0, self.uncertainty_params['initial_velocity']),
                'initial_attitude_offset': np.random.normal(0, self.uncertainty_params['initial_attitude']),
                'initial_angular_velocity_offset': np.random.normal(0, self.uncertainty_params['initial_angular_velocity']),
                'mass_multiplier': np.random.normal(1.0, self.uncertainty_params['mass_uncertainty']),
                'thrust_multiplier': np.random.normal(1.0, self.uncertainty_params['thrust_uncertainty']),
                'wind_speed': np.random.uniform(*self.uncertainty_params['wind_speed_range']),
                'wind_direction': np.random.uniform(*self.uncertainty_params['wind_direction_range']),
                'density_multiplier': np.random.normal(1.0, self.uncertainty_params['atmospheric_density_uncertainty']),
                'random_seed': i
            }
            samples.append(sample)
        
        return samples
    
    @staticmethod
    def _process_chunk(chunk, chunk_id, initial_conditions, base_analyzer):
        """Process a chunk of simulations."""
        results = []
        
        for i, params in enumerate(chunk):
            simulation_id = chunk_id * len(chunk) + i
            
            try:
                # Run single simulation
                result = base_analyzer._run_single_simulation(
                    initial_conditions, params, simulation_id
                )
                if result is not None:
                    results.append(result)
                    
            except Exception as e:
                print(f"Simulation {simulation_id} failed: {e}")
                continue
        
        return results
    
    def _run_single_simulation(self, base_initial_conditions, params, simulation_id):
        """Run a single Monte Carlo simulation."""
        # Create perturbed initial conditions
        initial_conditions = base_initial_conditions.copy()
        
        # Perturb initial conditions
        if 'position' in initial_conditions:
            initial_conditions['position'] = np.array(initial_conditions['position']) + params['initial_position_offset']
        else:
            initial_conditions['position'] = params['initial_position_offset']
        
        if 'velocity' in initial_conditions:
            initial_conditions['velocity'] = np.array(initial_conditions['velocity']) + params['initial_velocity_offset']
        else:
            initial_conditions['velocity'] = params['initial_velocity_offset']
        
        if 'attitude' in initial_conditions:
            initial_conditions['attitude'] = np.array(initial_conditions['attitude']) + params['initial_attitude_offset']
        else:
            initial_conditions['attitude'] = params['initial_attitude_offset']
        
        if 'angular_velocity' in initial_conditions:
            initial_conditions['angular_velocity'] = np.array(initial_conditions['angular_velocity']) + params['initial_angular_velocity_offset']
        else:
            initial_conditions['angular_velocity'] = params['initial_angular_velocity_offset']
        
        # Create perturbed rocket
        perturbed_rocket = self._perturb_rocket(params)
        
        # Create perturbed motor
        perturbed_motor = self._perturb_motor(params)
        
        # Create perturbed atmosphere
        perturbed_atmosphere = self._perturb_atmosphere(params)
        
        # Generate wind profile.  If a base profile has been provided, reuse it
        # for every simulation to mimic a deterministic forecast.  Otherwise a
        # stochastic profile is generated as before.
        if self.base_wind_profile is not None and self.base_altitude_profile is not None:
            altitude_profile = self.base_altitude_profile
            wind_profile = self.base_wind_profile
        else:
            altitude_profile = np.linspace(0, 25000, 100)  # Up to 25 km
            wind_profile = self.wind_model.generate_stochastic_profile(
                altitude_profile,
                params['wind_speed'],
                params['wind_direction'],
                random_state=np.random.RandomState(params['random_seed'])
            )
        
        # Create simulator
        simulator = FlightSimulator(perturbed_rocket, perturbed_motor, perturbed_atmosphere, self.wind_model)
        
        # Run simulation
        try:
            results = simulator.simulate_flight(initial_conditions, wind_profile, altitude_profile)
            results['simulation_id'] = simulation_id
            results['parameters'] = params
            results['trajectory'] = {
                'time': results['time'],
                'altitude': results['altitude'],
                'position': results['position'].T
            }
            return results
        except Exception as e:
            print(f"Simulation {simulation_id} failed: {e}")
            return None
    
    def _perturb_rocket(self, params):
        """Create perturbed rocket for Monte Carlo."""
        # Create a copy of the rocket (simplified - in practice, implement proper deep copy)
        from copy import deepcopy
        perturbed_rocket = deepcopy(self.rocket)
        
        # Perturb mass
        perturbed_rocket.dry_mass *= params['mass_multiplier']
        perturbed_rocket.propellant_mass *= params['mass_multiplier']
        
        return perturbed_rocket
    
    def _perturb_motor(self, params):
        """Create perturbed motor for Monte Carlo."""
        # Use the motor's built-in perturbation method
        random_state = np.random.RandomState(params['random_seed'])
        return self.motor.perturb_for_monte_carlo(random_state)
    
    def _perturb_atmosphere(self, params):
        """Create perturbed atmosphere for Monte Carlo."""
        # Create a copy of the atmosphere
        from copy import deepcopy
        perturbed_atmosphere = deepcopy(self.atmosphere)
        
        # Perturb density (simplified model)
        perturbed_atmosphere.sea_level_density *= params['density_multiplier']
        
        return perturbed_atmosphere
    
    def _analyze_results(self, results):
        """Analyze Monte Carlo simulation results."""
        # Filter out failed simulations
        valid_results = [r for r in results if r is not None]
        
        if len(valid_results) == 0:
            raise ValueError("No valid simulation results")
        
        # Extract key metrics
        apogee_altitudes = [r['apogee_altitude'] for r in valid_results]
        ranges = [r['range'] for r in valid_results]
        flight_times = [r['flight_time'] for r in valid_results]

        # Determine ranges of Monte Carlo parameters actually used
        param_ranges = {}
        for r in valid_results:
            params = r.get('parameters', {})
            for key, val in params.items():
                arr = np.array(val)
                if key not in param_ranges:
                    param_ranges[key] = {
                        'min': arr.astype(float),
                        'max': arr.astype(float),
                    }
                else:
                    param_ranges[key]['min'] = np.minimum(param_ranges[key]['min'], arr)
                    param_ranges[key]['max'] = np.maximum(param_ranges[key]['max'], arr)

        for key in param_ranges:
            param_ranges[key]['min'] = param_ranges[key]['min'].tolist()
            param_ranges[key]['max'] = param_ranges[key]['max'].tolist()
        
        # Calculate statistics
        analysis = {
            'n_samples': len(valid_results),
            'n_failed': len(results) - len(valid_results),
            'apogee_altitude': {
                'mean': float(np.mean(apogee_altitudes)),
                'std': float(np.std(apogee_altitudes)),
                'min': float(np.min(apogee_altitudes)),
                'max': float(np.max(apogee_altitudes)),
                'percentiles': np.percentile(apogee_altitudes, [5, 25, 50, 75, 95]).tolist()
            },
            'range': {
                'mean': float(np.mean(ranges)),
                'std': float(np.std(ranges)),
                'min': float(np.min(ranges)),
                'max': float(np.max(ranges)),
                'percentiles': np.percentile(ranges, [5, 25, 50, 75, 95]).tolist()
            },
            'flight_time': {
                'mean': float(np.mean(flight_times)),
                'std': float(np.std(flight_times)),
                'min': float(np.min(flight_times)),
                'max': float(np.max(flight_times)),
                'percentiles': np.percentile(flight_times, [5, 25, 50, 75, 95]).tolist()
            },
            'results': valid_results,
            'parameter_ranges_observed': param_ranges
        }
        
        return analysis
    
    def _create_output_directory(self):
        """Create versioned output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/monte_carlo_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def _save_report(self, analysis, output_dir):
        """Save Monte Carlo analysis report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'simulation_summary': {
                'total_simulations': analysis['n_samples'],
                'failed_simulations': analysis['n_failed'],
                'success_rate': analysis['n_samples'] / (analysis['n_samples'] + analysis['n_failed']) * 100
            },
            'apogee_altitude_stats': analysis['apogee_altitude'],
            'range_stats': analysis['range'],
            'flight_time_stats': analysis['flight_time'],
            'uncertainty_parameters': self.uncertainty_params,
            'parameter_ranges_observed': analysis.get('parameter_ranges_observed'),
            'rocket_parameters': object_to_serializable_dict(self.rocket),
            'motor_parameters': object_to_serializable_dict(self.motor),
            'atmosphere_parameters': object_to_serializable_dict(self.atmosphere),
            'wind_model_parameters': object_to_serializable_dict(self.wind_model)
        }
        
        if 'performance' in analysis:
            report['performance'] = analysis['performance']

        # Save JSON report
        with open(os.path.join(output_dir, 'monte_carlo_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        # Save each simulation result to a separate JSON file for detailed analysis
        sims_dir = os.path.join(output_dir, 'simulation_results')
        os.makedirs(sims_dir, exist_ok=True)
        for result in analysis.get('results', []):
            sim_id = result.get('simulation_id', len(os.listdir(sims_dir)))
            filename = os.path.join(sims_dir, f'sim_{sim_id}.json')
            with open(filename, 'w') as sf:
                json.dump(to_serializable(result), sf)

        # Save human-readable report
        with open(os.path.join(output_dir, 'monte_carlo_report.txt'), 'w') as f:
            f.write("Monte Carlo Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            f.write("Simulation Summary:\n")
            f.write(f"  Total simulations: {report['simulation_summary']['total_simulations']}\n")
            f.write(f"  Failed simulations: {report['simulation_summary']['failed_simulations']}\n")
            f.write(f"  Success rate: {report['simulation_summary']['success_rate']:.1f}%\n\n")
            
            f.write("Apogee Altitude Statistics:\n")
            stats = report['apogee_altitude_stats']
            f.write(f"  Mean: {stats['mean']:.1f} m\n")
            f.write(f"  Standard Deviation: {stats['std']:.1f} m\n")
            f.write(f"  Min: {stats['min']:.1f} m\n")
            f.write(f"  Max: {stats['max']:.1f} m\n")
            f.write(f"  95% Confidence Interval: [{stats['percentiles'][0]:.1f}, {stats['percentiles'][4]:.1f}] m\n\n")
            
            f.write("Range Statistics:\n")
            stats = report['range_stats']
            f.write(f"  Mean: {stats['mean']:.1f} m\n")
            f.write(f"  Standard Deviation: {stats['std']:.1f} m\n")
            f.write(f"  Min: {stats['min']:.1f} m\n")
            f.write(f"  Max: {stats['max']:.1f} m\n")
            f.write(f"  95% Confidence Interval: [{stats['percentiles'][0]:.1f}, {stats['percentiles'][4]:.1f}] m\n\n")
            
            f.write("Flight Time Statistics:\n")
            stats = report['flight_time_stats']
            f.write(f"  Mean: {stats['mean']:.1f} s\n")
            f.write(f"  Standard Deviation: {stats['std']:.1f} s\n")
            f.write(f"  Min: {stats['min']:.1f} s\n")
            f.write(f"  Max: {stats['max']:.1f} s\n")
            f.write(f"  95% Confidence Interval: [{stats['percentiles'][0]:.1f}, {stats['percentiles'][4]:.1f}] s\n\n")
            
            if 'performance' in report:
                f.write("Performance Statistics:\n")
                perf = report['performance']
                f.write(f"  Total time: {perf['total_time']:.2f} s\n")
                f.write(f"  Simulations per second: {perf['simulations_per_second']:.1f}\n")
                f.write(f"  Cores used: {perf['cores_used']}\n")
    
    def plot_results(self, analysis, save_plots=True):
        """Plot Monte Carlo analysis results."""
        output_dir = None
        _, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Apogee altitude histogram
        apogee_altitudes = [r['apogee_altitude'] for r in analysis['results']]
        axes[0, 0].hist(apogee_altitudes, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Apogee Altitude (m)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Apogee Altitude Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Range histogram
        ranges = [r['range'] for r in analysis['results']]
        axes[0, 1].hist(ranges, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Range (m)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Range Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Flight time histogram
        flight_times = [r['flight_time'] for r in analysis['results']]
        axes[1, 0].hist(flight_times, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Flight Time (s)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Flight Time Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot: Range vs Apogee
        axes[1, 1].scatter(apogee_altitudes, ranges, alpha=0.6, s=10)
        axes[1, 1].set_xlabel('Apogee Altitude (m)')
        axes[1, 1].set_ylabel('Range (m)')
        axes[1, 1].set_title('Range vs Apogee Altitude')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plots if requested
        if save_plots:
            output_dir = self._create_output_directory()
            plot_path = os.path.join(output_dir, 'monte_carlo_distributions.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {plot_path}")
            
            # Save the report
            self._save_report(analysis, output_dir)
            print(f"Report saved to: {output_dir}")
        
        # plt.show()
        
        # Print statistics
        print("\nMonte Carlo Analysis Results:")
        print(f"Number of successful simulations: {analysis['n_samples']}")
        print(f"Number of failed simulations: {analysis['n_failed']}")
        print(f"\nApogee Altitude Statistics:")
        print(f"  Mean: {analysis['apogee_altitude']['mean']:.1f} m")
        print(f"  Standard Deviation: {analysis['apogee_altitude']['std']:.1f} m")
        print(f"  95% Confidence Interval: [{analysis['apogee_altitude']['percentiles'][0]:.1f}, {analysis['apogee_altitude']['percentiles'][4]:.1f}] m")
        print(f"\nRange Statistics:")
        print(f"  Mean: {analysis['range']['mean']:.1f} m")
        print(f"  Standard Deviation: {analysis['range']['std']:.1f} m")
        print(f"  95% Confidence Interval: [{analysis['range']['percentiles'][0]:.1f}, {analysis['range']['percentiles'][4]:.1f}] m")
        
        return output_dir
    
    def plot_trajectory_cloud(self, analysis, save_plots=True, max_trajectories=50):
        """Plot a cloud of trajectories from Monte Carlo results."""
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sample trajectories to plot
        trajectories = analysis['results'][:max_trajectories]
        
        # Plot altitude vs time
        for i, result in enumerate(trajectories):
            if 'trajectory' in result:
                t = result['trajectory']['time']
                alt = result['trajectory']['altitude']
                ax1.plot(t, alt, alpha=0.3, linewidth=0.5, color='blue')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Altitude (m)')
        ax1.set_title(f'Trajectory Cloud - Altitude vs Time\\n({len(trajectories)} trajectories)')
        ax1.grid(True, alpha=0.3)
        
        # Plot ground track (if position data available)
        for i, result in enumerate(trajectories):
            if 'trajectory' in result and 'position' in result['trajectory']:
                pos = result['trajectory']['position']
                x = pos[:, 0]  # East position
                y = pos[:, 1]  # North position
                ax2.plot(x, y, alpha=0.3, linewidth=0.5, color='red')
        
        ax2.set_xlabel('East Position (m)')
        ax2.set_ylabel('North Position (m)')
        ax2.set_title(f'Ground Track Cloud\\n({len(trajectories)} trajectories)')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        plt.tight_layout()
        
        # Save plots if requested
        if save_plots:
            output_dir = self._create_output_directory()
            plot_path = os.path.join(output_dir, 'monte_carlo_trajectories.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory plots saved to: {plot_path}")

        # plt.show()

    def plot_trajectory_cloud_3d(self, analysis, save_plots=True, max_trajectories=50):
        """Plot 3D trajectories from Monte Carlo results."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        trajectories = analysis['results'][:max_trajectories]
        for result in trajectories:
            if 'trajectory' in result and 'position' in result['trajectory']:
                pos = result['trajectory']['position']
                x = pos[:, 0]
                y = pos[:, 1]
                z = pos[:, 2]
                ax.plot(x, y, z, alpha=0.3, linewidth=0.5)

        ax.set_xlabel('East Position (m)')
        ax.set_ylabel('North Position (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title(f'3D Trajectory Cloud ({len(trajectories)} trajectories)')
        ax.grid(True, alpha=0.3)

        if save_plots:
            output_dir = self._create_output_directory()
            plot_path = os.path.join(output_dir, 'monte_carlo_trajectories_3d.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"3D trajectory plot saved to: {plot_path}")

        # plt.show()
