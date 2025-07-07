#ifdef __APPLE__

#include "rocket_simulation.hpp"
#include <Metal/Metal.h>
#include <MetalKit/MetalKit.h>
#include <iostream>
#include <fstream>
#include <string>

namespace rocket_sim {

// Metal structures matching the shader
struct MetalRocketParams {
    float dry_mass;
    float propellant_mass;
    float length;
    float diameter;
    float reference_area;
    float cp_location;
    float center_of_mass_dry;
    float Ixx_dry;
    float Iyy_dry;
    float Izz_dry;
};

struct MetalMotorParams {
    float total_impulse;
    float burn_time;
    float average_thrust;
    float mass_flow_rate;
};

struct MetalAtmosphereParams {
    float sea_level_pressure;
    float sea_level_temperature;
    float sea_level_density;
    float temperature_lapse_rate;
    float gas_constant;
    float gravity;
};

struct MetalThrustCurvePoint {
    float time;
    float thrust;
};

struct MetalSimulationParams {
    float max_time;
    float dt;
    float ground_threshold;
    int max_steps;
};

struct MetalSimulationResult {
    float apogee_altitude;
    float apogee_time;
    float range;
    float flight_time;
    float max_speed;
    bool success;
};

MetalCompute::MetalCompute() : device_(nullptr), command_queue_(nullptr), monte_carlo_pipeline_(nullptr), library_(nullptr) {
}

MetalCompute::~MetalCompute() {
    shutdown();
}

bool MetalCompute::initialize() {
    @autoreleasepool {
        // Get default Metal device
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            std::cerr << "Metal is not supported on this device" << std::endl;
            return false;
        }
        
        std::cout << "Metal device: " << [[device_ name] UTF8String] << std::endl;
        
        // Create command queue
        command_queue_ = [device_ newCommandQueue];
        if (!command_queue_) {
            std::cerr << "Failed to create Metal command queue" << std::endl;
            return false;
        }
        
        // Create compute pipeline
        if (!create_compute_pipeline()) {
            std::cerr << "Failed to create compute pipeline" << std::endl;
            return false;
        }
        
        return true;
    }
}

void MetalCompute::shutdown() {
    @autoreleasepool {
        if (monte_carlo_pipeline_) {
            [monte_carlo_pipeline_ release];
            monte_carlo_pipeline_ = nullptr;
        }
        
        if (library_) {
            [library_ release];
            library_ = nullptr;
        }
        
        if (command_queue_) {
            [command_queue_ release];
            command_queue_ = nullptr;
        }
        
        if (device_) {
            [device_ release];
            device_ = nullptr;
        }
    }
}

bool MetalCompute::create_compute_pipeline() {
    @autoreleasepool {
        // Load Metal shader source
        NSString* shaderPath = @"shaders/monte_carlo.metal";
        NSError* error = nil;
        NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                            encoding:NSUTF8StringEncoding
                                                               error:&error];
        
        if (!shaderSource) {
            // Try reading from current directory
            std::ifstream file("cpp_metal/shaders/monte_carlo.metal");
            if (!file.is_open()) {
                std::cerr << "Failed to load Metal shader source" << std::endl;
                return false;
            }
            
            std::string source((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
            shaderSource = [NSString stringWithUTF8String:source.c_str()];
        }
        
        // Create library from source
        library_ = [device_ newLibraryWithSource:shaderSource options:nil error:&error];
        if (!library_) {
            std::cerr << "Failed to create Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        
        // Get the compute function
        id<MTLFunction> computeFunction = [library_ newFunctionWithName:@"monte_carlo_simulation"];
        if (!computeFunction) {
            std::cerr << "Failed to find compute function 'monte_carlo_simulation'" << std::endl;
            return false;
        }
        
        // Create compute pipeline state
        monte_carlo_pipeline_ = [device_ newComputePipelineStateWithFunction:computeFunction error:&error];
        if (!monte_carlo_pipeline_) {
            std::cerr << "Failed to create compute pipeline: " << [[error localizedDescription] UTF8String] << std::endl;
            return false;
        }
        
        [computeFunction release];
        return true;
    }
}

std::vector<SimulationResult> MetalCompute::run_monte_carlo_batch(
    const std::vector<State>& initial_states,
    const RocketConfig& rocket,
    const MotorConfig& motor,
    const AtmosphereModel& atmosphere,
    const std::vector<WindProfile>& wind_profiles) {
    
    @autoreleasepool {
        const size_t n_simulations = initial_states.size();
        std::vector<SimulationResult> results(n_simulations);
        
        if (n_simulations == 0) {
            return results;
        }
        
        // Prepare Metal structures
        MetalRocketParams metal_rocket = {
            rocket.dry_mass, rocket.propellant_mass, rocket.length, rocket.diameter,
            rocket.reference_area, rocket.cp_location, rocket.center_of_mass_dry,
            rocket.Ixx_dry, rocket.Iyy_dry, rocket.Izz_dry
        };
        
        MetalMotorParams metal_motor = {
            motor.total_impulse, motor.burn_time, motor.average_thrust, motor.mass_flow_rate
        };
        
        MetalAtmosphereParams metal_atmosphere = {
            atmosphere.sea_level_pressure, atmosphere.sea_level_temperature,
            atmosphere.sea_level_density, atmosphere.temperature_lapse_rate,
            atmosphere.gas_constant, atmosphere.gravity
        };
        
        // Prepare thrust curve
        std::vector<MetalThrustCurvePoint> thrust_curve;
        for (size_t i = 0; i < motor.time_data.size(); ++i) {
            thrust_curve.push_back({motor.time_data[i], motor.thrust_data[i]});
        }
        
        MetalSimulationParams sim_params = {
            300.0f,  // max_time
            0.01f,   // dt
            0.5f,    // ground_threshold
            30000    // max_steps
        };
        
        // Create Metal buffers
        id<MTLBuffer> statesBuffer = [device_ newBufferWithBytes:initial_states.data()
                                                          length:sizeof(State) * n_simulations
                                                         options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> resultsBuffer = [device_ newBufferWithLength:sizeof(MetalSimulationResult) * n_simulations
                                                           options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> rocketBuffer = [device_ newBufferWithBytes:&metal_rocket
                                                          length:sizeof(MetalRocketParams)
                                                         options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> motorBuffer = [device_ newBufferWithBytes:&metal_motor
                                                         length:sizeof(MetalMotorParams)
                                                        options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> atmosphereBuffer = [device_ newBufferWithBytes:&metal_atmosphere
                                                              length:sizeof(MetalAtmosphereParams)
                                                             options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> thrustCurveBuffer = [device_ newBufferWithBytes:thrust_curve.data()
                                                              length:sizeof(MetalThrustCurvePoint) * thrust_curve.size()
                                                             options:MTLResourceStorageModeShared];
        
        id<MTLBuffer> simParamsBuffer = [device_ newBufferWithBytes:&sim_params
                                                             length:sizeof(MetalSimulationParams)
                                                            options:MTLResourceStorageModeShared];
        
        int thrust_curve_length = static_cast<int>(thrust_curve.size());
        id<MTLBuffer> thrustLengthBuffer = [device_ newBufferWithBytes:&thrust_curve_length
                                                               length:sizeof(int)
                                                              options:MTLResourceStorageModeShared];
        
        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Set pipeline and buffers
        [encoder setComputePipelineState:monte_carlo_pipeline_];
        [encoder setBuffer:statesBuffer offset:0 atIndex:0];
        [encoder setBuffer:resultsBuffer offset:0 atIndex:1];
        [encoder setBuffer:rocketBuffer offset:0 atIndex:2];
        [encoder setBuffer:motorBuffer offset:0 atIndex:3];
        [encoder setBuffer:atmosphereBuffer offset:0 atIndex:4];
        [encoder setBuffer:thrustCurveBuffer offset:0 atIndex:5];
        [encoder setBuffer:simParamsBuffer offset:0 atIndex:6];
        [encoder setBuffer:thrustLengthBuffer offset:0 atIndex:7];
        
        // Calculate thread group sizes
        NSUInteger threadExecutionWidth = monte_carlo_pipeline_.threadExecutionWidth;
        NSUInteger maxThreadsPerGroup = monte_carlo_pipeline_.maxTotalThreadsPerThreadgroup;
        
        NSUInteger threadsPerGroup = std::min(threadExecutionWidth, maxThreadsPerGroup);
        NSUInteger numGroups = (n_simulations + threadsPerGroup - 1) / threadsPerGroup;
        
        MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerGroup, 1, 1);
        MTLSize threadgroupsPerGrid = MTLSizeMake(numGroups, 1, 1);
        
        // Dispatch compute
        [encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [encoder endEncoding];
        
        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // Copy results back
        MetalSimulationResult* metal_results = static_cast<MetalSimulationResult*>([resultsBuffer contents]);
        
        for (size_t i = 0; i < n_simulations; ++i) {
            results[i].apogee_altitude = metal_results[i].apogee_altitude;
            results[i].apogee_time = metal_results[i].apogee_time;
            results[i].range = metal_results[i].range;
            results[i].flight_time = metal_results[i].flight_time;
            results[i].max_speed = metal_results[i].max_speed;
            results[i].success = metal_results[i].success;
        }
        
        // Clean up
        [statesBuffer release];
        [resultsBuffer release];
        [rocketBuffer release];
        [motorBuffer release];
        [atmosphereBuffer release];
        [thrustCurveBuffer release];
        [simParamsBuffer release];
        [thrustLengthBuffer release];
        
        return results;
    }
}

} // namespace rocket_sim

#endif // __APPLE__