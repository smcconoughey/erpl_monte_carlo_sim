"""
6DOF Monte Carlo Rocket Simulation Package
==========================================

A comprehensive simulation framework for suborbital sounding rockets
with full 6-degree-of-freedom dynamics and Monte Carlo uncertainty analysis.
"""

__version__ = "1.0.0"
__author__ = "Rocket Simulation Team"

from rocket import Rocket
from motor import SolidMotor
from environment import StandardAtmosphere, WindModel
from simulator import FlightSimulator
from monte_carlo import MonteCarloAnalyzer
from utils import *

__all__ = [
    'Rocket',
    'SolidMotor', 
    'StandardAtmosphere',
    'WindModel',
    'FlightSimulator',
    'MonteCarloAnalyzer'
] 