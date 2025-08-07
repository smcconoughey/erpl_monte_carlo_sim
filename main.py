#!/usr/bin/env python3
"""Convenience runner.

Use this file to run the simulator without worrying about package imports:

  python main.py --help
  python main.py --no-wind --mc-samples 15 --tilt-deg 1 --alt0 10
"""

from rocket_simulation.main import main


if __name__ == "__main__":
    main()

