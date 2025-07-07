#!/bin/bash

# CPU-only build script using clang directly

set -e

echo "Building C++ Rocket Simulation (CPU-only)..."

# Create build directory
mkdir -p build

# Compile source files (excluding Metal)
echo "Compiling source files..."

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -DCPU_ONLY \
    -c src/rocket_config.cpp -o build/rocket_config.o

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -DCPU_ONLY \
    -c src/motor.cpp -o build/motor.o

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -DCPU_ONLY \
    -c src/atmosphere.cpp -o build/atmosphere.o

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -DCPU_ONLY \
    -c src/utils.cpp -o build/utils.o

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -DCPU_ONLY \
    -c src/simulator.cpp -o build/simulator.o

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -DCPU_ONLY \
    -c src/monte_carlo_cpu.cpp -o build/monte_carlo.o

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -DCPU_ONLY \
    -c src/main_cpu.cpp -o build/main.o

# Link executable
echo "Linking executable..."

clang++ -std=c++17 -O3 \
    build/*.o -o build/rocket_sim_cpu

echo "Build complete!"
echo "Run the simulation with: cd build && ./rocket_sim_cpu"