#!/bin/bash

# Manual build script using clang directly

set -e

echo "Building C++/Metal Rocket Simulation manually..."

# Create build directory
mkdir -p build

# Compile source files
echo "Compiling source files..."

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -framework Metal -framework MetalKit -framework Foundation -framework Cocoa \
    -c src/rocket_config.cpp -o build/rocket_config.o

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -framework Metal -framework MetalKit -framework Foundation -framework Cocoa \
    -c src/motor.cpp -o build/motor.o

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -framework Metal -framework MetalKit -framework Foundation -framework Cocoa \
    -c src/atmosphere.cpp -o build/atmosphere.o

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -framework Metal -framework MetalKit -framework Foundation -framework Cocoa \
    -c src/utils.cpp -o build/utils.o

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -framework Metal -framework MetalKit -framework Foundation -framework Cocoa \
    -c src/simulator.cpp -o build/simulator.o

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -framework Metal -framework MetalKit -framework Foundation -framework Cocoa \
    -c src/metal_compute.cpp -o build/metal_compute.o

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -framework Metal -framework MetalKit -framework Foundation -framework Cocoa \
    -c src/monte_carlo.cpp -o build/monte_carlo.o

clang++ -std=c++17 -O3 -march=native -ffast-math \
    -I./include \
    -framework Metal -framework MetalKit -framework Foundation -framework Cocoa \
    -c src/main.cpp -o build/main.o

# Link executable
echo "Linking executable..."

clang++ -std=c++17 -O3 \
    -framework Metal -framework MetalKit -framework Foundation -framework Cocoa \
    build/*.o -o build/rocket_sim

echo "Build complete!"
echo "Run the simulation with: cd build && ./rocket_sim"