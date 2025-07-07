#!/bin/bash

# Build script for C++/Metal Rocket Simulation

set -e

echo "Building C++/Metal Rocket Simulation..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

echo "Build complete!"
echo "Run the simulation with: ./rocket_sim"