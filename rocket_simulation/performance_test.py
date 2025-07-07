"""
Performance test script for Monte Carlo optimizations
"""

import time
import numpy as np
from monte_carlo import MonteCarloAnalyzer
from monte_carlo_optimized import create_memory_efficient_analyzer

def create_test_objects():
    """Create test objects for performance testing."""
    # Mock objects for testing
    class MockRocket:
        def __init__(self):
            self.dry_mass = 1.5
            self.propellant_mass = 0.8
    
    class MockMotor:
        def perturb_for_monte_carlo(self, random_state):
            return self
    
    class MockAtmosphere:
        def __init__(self):
            self.sea_level_density = 1.225
    
    class MockWindModel:
        def generate_stochastic_profile(self, altitude_profile, wind_speed, wind_direction, random_state):
            return np.zeros((len(altitude_profile), 2))
    
    return MockRocket(), MockMotor(), MockAtmosphere(), MockWindModel()

def run_performance_test():
    """Run performance comparison test."""
    print("=== Monte Carlo Performance Test ===")
    
    # Create test objects
    rocket, motor, atmosphere, wind_model = create_test_objects()
    
    # Create analyzers
    base_analyzer = MonteCarloAnalyzer(rocket, motor, atmosphere, wind_model)
    optimized_analyzer = create_memory_efficient_analyzer(base_analyzer)
    
    # Test parameters
    initial_conditions = {
        'position': [0, 0, 0],
        'velocity': [0, 0, 0],
        'attitude': [0, 0, 0],
        'angular_velocity': [0, 0, 0]
    }
    
    test_samples = [100, 500, 1000]
    
    print(f"Testing with {test_samples} samples each")
    print(f"Using {optimized_analyzer.n_cores} CPU cores")
    
    for n_samples in test_samples:
        print(f"\n--- Testing with {n_samples} samples ---")
        
        # Test parameter generation speed
        print("Testing parameter generation...")
        start_time = time.time()
        params = base_analyzer._generate_parameter_samples_vectorized(n_samples)
        gen_time = time.time() - start_time
        print(f"Generated {n_samples} parameter sets in {gen_time:.3f}s ({n_samples/gen_time:.1f} params/sec)")
        
        # Test memory usage estimation
        param_size = len(str(params)) / 1024 / 1024  # Rough MB estimate
        print(f"Estimated parameter memory usage: {param_size:.2f} MB")
        
        print("Performance test complete!")

if __name__ == "__main__":
    run_performance_test()