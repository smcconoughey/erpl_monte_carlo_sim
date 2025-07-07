#include "rocket_simulation_cpp.hpp"
#include <cmath>

namespace rocket_sim {

AtmosphereModel::AtmosphereProperties AtmosphereModel::get_properties(float altitude) const {
    AtmosphereProperties props;
    
    const float troposphere_height = 11000.0f;
    const float stratosphere_height = 20000.0f;
    const float stratosphere_temp = 216.65f;
    const float gamma = 1.4f;
    
    if (altitude <= troposphere_height) {
        // Troposphere
        props.temperature = sea_level_temperature - temperature_lapse_rate * altitude;
        props.pressure = sea_level_pressure * 
            std::pow(props.temperature / sea_level_temperature, 
                    gravity / (gas_constant * temperature_lapse_rate));
    }
    else if (altitude <= stratosphere_height) {
        // Lower stratosphere (isothermal)
        props.temperature = stratosphere_temp;
        float pressure_11km = sea_level_pressure * 
            std::pow(stratosphere_temp / sea_level_temperature,
                    gravity / (gas_constant * temperature_lapse_rate));
        
        props.pressure = pressure_11km * 
            std::exp(-gravity * (altitude - troposphere_height) / 
                    (gas_constant * props.temperature));
    }
    else {
        // Extended atmosphere (simplified exponential decay)
        props.temperature = stratosphere_temp;
        props.pressure = 1000.0f * std::exp(-altitude / 8000.0f);
    }
    
    // Calculate density
    props.density = props.pressure / (gas_constant * props.temperature);
    
    // Calculate speed of sound
    props.speed_of_sound = std::sqrt(gamma * gas_constant * props.temperature);
    
    return props;
}

float AtmosphereModel::get_gravity(float altitude) const {
    const float earth_radius = 6.371e6f; // m
    return gravity * std::pow(earth_radius / (earth_radius + altitude), 2.0f);
}

Vec3 WindProfile::get_wind_at_altitude(float altitude) const {
    if (altitudes.empty()) {
        return Vec3();
    }
    
    // Linear interpolation
    if (altitude <= altitudes.front()) {
        return wind_velocities.front();
    }
    if (altitude >= altitudes.back()) {
        return wind_velocities.back();
    }
    
    // Find the two points to interpolate between
    size_t i = 0;
    while (i < altitudes.size() - 1 && altitudes[i + 1] < altitude) {
        ++i;
    }
    
    float h0 = altitudes[i];
    float h1 = altitudes[i + 1];
    Vec3 w0 = wind_velocities[i];
    Vec3 w1 = wind_velocities[i + 1];
    
    // Linear interpolation
    float alpha = (altitude - h0) / (h1 - h0);
    return Vec3(
        w0.x + alpha * (w1.x - w0.x),
        w0.y + alpha * (w1.y - w0.y),
        w0.z + alpha * (w1.z - w0.z)
    );
}

} // namespace rocket_sim