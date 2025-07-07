cmake_minimum_required(VERSION 3.20)
project(RocketSimulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PkgConfig REQUIRED)

# Metal framework (macOS only)
if(APPLE)
    find_library(METAL_LIBRARY Metal)
    find_library(METALKIT_LIBRARY MetalKit)
    find_library(FOUNDATION_LIBRARY Foundation)
    find_library(COCOA_LIBRARY Cocoa)
    
    if(NOT METAL_LIBRARY)
        message(FATAL_ERROR "Metal framework not found")
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/rocket_config.cpp
    src/motor.cpp
    src/atmosphere.cpp
    src/simulator.cpp
    src/monte_carlo.cpp
    src/metal_compute.cpp
    src/utils.cpp
)

# Create library
add_library(rocket_simulation SHARED ${SOURCES})

# Link libraries
if(APPLE)
    target_link_libraries(rocket_simulation 
        ${METAL_LIBRARY}
        ${METALKIT_LIBRARY}
        ${FOUNDATION_LIBRARY}
        ${COCOA_LIBRARY}
    )
endif()

# Create executable
add_executable(rocket_sim src/main.cpp)
target_link_libraries(rocket_sim rocket_simulation)

# Python bindings (optional)
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(rocket_simulation_py src/python_bindings.cpp)
    target_link_libraries(rocket_simulation_py PRIVATE rocket_simulation)
endif()

# Compiler flags
target_compile_options(rocket_simulation PRIVATE
    -O3
    -march=native
    -ffast-math
    -DNDEBUG
)

# Install
install(TARGETS rocket_simulation rocket_sim
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)