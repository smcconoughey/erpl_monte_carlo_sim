#include "rocket_simulation_cpp.hpp"
#include <cmath>

namespace rocket_sim {

RocketConfig::RocketConfig() {
    // Initialize aerodynamic data
    mach_data = {0.0f, 0.5f, 0.8f, 1.0f, 1.2f, 1.5f, 2.0f, 3.0f};
    cd0_data = {0.4f, 0.42f, 0.48f, 0.65f, 0.52f, 0.45f, 0.40f, 0.38f};
    cda_data = {1.2f, 1.25f, 1.3f, 1.4f, 1.35f, 1.25f, 1.2f, 1.15f};
    
    calculate_derived_properties();
}

void RocketConfig::calculate_derived_properties() {
    // Reference area
    reference_area = M_PI * (diameter / 2.0f) * (diameter / 2.0f);
    
    // Center of pressure calculation (Barrowman method)
    float nose_length = 0.3f;
    float fin_span = 0.15f;
    float fin_root_chord = 0.2f;
    float fin_tip_chord = 0.1f;
    int fin_count = 4;
    
    // Nose cone contribution
    float CN_nose = 2.0f;
    float X_nose = 0.666f * nose_length;
    
    // Fin contribution
    float fin_area = 0.5f * (fin_root_chord + fin_tip_chord) * fin_span;
    float fin_mid_chord = (fin_root_chord + fin_tip_chord) / 2.0f;
    
    float CN_fins = 2.0f * fin_count * (1.0f + diameter / (2.0f * fin_span)) *
                    (fin_area / reference_area);
    
    float X_fins = length - fin_root_chord +
                   (fin_tip_chord + 2.0f * fin_root_chord) * fin_mid_chord /
                   (3.0f * (fin_tip_chord + fin_root_chord));
    
    // Total center of pressure
    float CN_total = CN_nose + CN_fins;
    if (CN_total > 0) {
        cp_location = (CN_nose * X_nose + CN_fins * X_fins) / CN_total;
    } else {
        cp_location = length / 2.0f;
    }
}

} // namespace rocket_sim