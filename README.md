# 6DOF Monte Carlo Rocket Simulation

A complete Python implementation of a 6-degree-of-freedom Monte Carlo simulation for suborbital sounding rockets targeting 60,000 feet altitude.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from rocket_simulation import *

# Create rocket and environment
rocket = Rocket("My Rocket")
motor = SolidMotor("Motor")
atmosphere = StandardAtmosphere()
wind_model = WindModel()

# Run simulation
simulator = FlightSimulator(rocket, motor, atmosphere, wind_model)
initial_conditions = {
    'position': [0.0, 0.0, 0.0],
    'velocity': [0.0, 0.0, 0.0],
    'attitude': [0.0, 0.02, 0.0],  # Small pitch angle
    'angular_velocity': [0.0, 0.0, 0.0]
}

# Single simulation
results = simulator.simulate_flight(initial_conditions)

# Monte Carlo analysis
monte_carlo = MonteCarloAnalyzer(rocket, motor, atmosphere, wind_model)
mc_results = monte_carlo.run_monte_carlo(initial_conditions, n_samples=1000)
```

## Example Usage

Run the complete example:

```python
python rocket_simulation/example.py
```

## Package Structure

- `rocket.py` - Rocket configuration and properties
- `motor.py` - Motor and thrust modeling
- `environment.py` - Atmospheric and wind models
- `simulator.py` - Flight dynamics and integration
- `monte_carlo.py` - Monte Carlo framework
- `utils.py` - Utility functions
- `example.py` - Example usage and visualization

## Key Features

- Full 6DOF dynamics with quaternion-based attitude representation
- Monte Carlo uncertainty propagation
- Standard atmosphere and wind modeling
- Parallel processing for Monte Carlo simulations
- Comprehensive visualization and analysis tools 