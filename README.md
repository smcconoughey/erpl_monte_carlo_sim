# 6DOF Monte Carlo Rocket Simulation

A complete Python implementation of a 6-degree-of-freedom Monte Carlo simulation for suborbital sounding rockets targeting 60,000 feet altitude.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Activate venv (first time)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run configurable entrypoint
python -m rocket_simulation.main --help
python -m rocket_simulation.main --motor liquid --mc-samples 50 --tilt-deg 1.0 --alt0 10
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
- `main.py` - Configurable entrypoint (replaces `example.py`)
  - `python -m rocket_simulation.main --help`
  - `--no-wind` to disable winds; `--wind-csv` to provide your own profile

## Key Features

- Full 6DOF dynamics with quaternion-based attitude representation
- Monte Carlo uncertainty propagation
- Standard atmosphere and wind modeling, with the ability to load
  altitude-resolved wind profiles from CSV files
  and add stochastic perturbations for Monte Carlo analysis
- Parallel processing for Monte Carlo simulations
- Comprehensive visualization and analysis tools
- 3D trajectory plotting and CP/CG tracking
- Detailed launch metrics including rail exit speed and aerodynamic angles
- Separate power-on and power-off drag modeling
- Simple parachute deployment logic to terminate free-fall

