"""
Rocket configuration and mass properties
"""

import numpy as np
from utils import *

class Rocket:
    """Represents a rocket with all its physical properties."""
    
    def __init__(self, name="Sounding Rocket"):
        self.name = name
        
        # Geometric properties
        self.length = 7.62  # Total length (m)
        self.diameter = 0.219  # Body diameter (m) - 8.625 in
        self.nose_length = 0.2  # Nose cone length (m)
        self.fin_span = 0.2  # Fin span (m)
        self.fin_root_chord = 0.20  # Fin root chord (m)
        self.fin_tip_chord = 0.1 # Fin tip chord (m)
        self.fin_count = 4  # Number of fins
        
        # Mass properties (dry mass)
        self.dry_mass = 113.4  # kg (250 lb)
        self.propellant_mass = 63.5  # kg (140 lb)
        self.center_of_mass_dry = 5.9 # m from nose
        
        # Moments of inertia (dry, kg*m^2)
        self.Ixx_dry = 1.683   # Roll moment of inertia 
        self.Iyy_dry = 971.9  # Pitch moment of inertia
        self.Izz_dry = 971.693  # Yaw moment of inertia
        
        # Aerodynamic properties
        self.reference_area = np.pi * (self.diameter / 2)**2
        self.reference_diameter = self.diameter
        
        # Aerodynamic coefficients (functions of Mach number and angle of attack)
        self.Cd_data = {
            'mach': [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0],
            'cd0': [0.4, 0.42, 0.48, 0.65, 0.52, 0.45, 0.40, 0.38],
            'cda': [1.2, 1.25, 1.3, 1.4, 1.35, 1.25, 1.2, 1.15]  # per radian
        }

        # Simple dynamic CP shift with Mach number (forward negative)
        self.CP_shift_data = {
            'mach': [0.0, 0.8, 1.0, 1.2, 2.0, 3.0],
            'cp_shift': [0.0, -0.05, -0.1, -0.05, 0.0, 0.0]  # meters
        }
        
        # Center of pressure calculation (Barrowman method)
        self.cp_location = self._calculate_center_of_pressure()
        
        # Recovery system
        self.parachute_area = 20.0  # m^2 (assuming 8 in is recovery bay diameter, not chute)
        self.parachute_cd = 1.3
        self.parachute_deployment_altitude = 2000  # m
        
    def _calculate_center_of_pressure(self):
        """Calculate center of pressure using Barrowman equations."""
        # Nose cone contribution
        CN_nose = 2.0
        X_nose = 0.666 * self.nose_length
        
        # Body contribution (negligible for cylindrical body)
        CN_body = 0.0
        X_body = 0.0
        
        # Fin contribution
        fin_area = 0.5 * (self.fin_root_chord + self.fin_tip_chord) * self.fin_span
        fin_mid_chord = (self.fin_root_chord + self.fin_tip_chord) / 2
        
        CN_fins = 2 * self.fin_count * (1 + self.diameter / (2 * self.fin_span)) * \
                  (fin_area / self.reference_area)
        
        X_fins = self.length - self.fin_root_chord + \
                 (self.fin_tip_chord + 2 * self.fin_root_chord) * fin_mid_chord / \
                 (3 * (self.fin_tip_chord + self.fin_root_chord))
        
        # Total center of pressure
        CN_total = CN_nose + CN_body + CN_fins
        if CN_total > 0:
            X_cp = (CN_nose * X_nose + CN_body * X_body + CN_fins * X_fins) / CN_total
        else:
            X_cp = self.length / 2
            
        return X_cp

    def get_dynamic_cp(self, mach, alpha=0.0):
        """Return center of pressure shifted with Mach number."""
        shift = interpolate_1d(mach, self.CP_shift_data['mach'], self.CP_shift_data['cp_shift'])
        return self.cp_location + shift
    
    def get_mass_properties(self, propellant_fraction_remaining):
        """Get current mass properties based on propellant remaining."""
        current_propellant = self.propellant_mass * propellant_fraction_remaining
        total_mass = self.dry_mass + current_propellant
        
        # Update center of mass (assuming propellant is uniformly distributed)
        propellant_cg = self.center_of_mass_dry - 0.5  # Propellant CG forward of dry CG
        current_cg = (self.dry_mass * self.center_of_mass_dry + 
                     current_propellant * propellant_cg) / total_mass
        
        # Update moments of inertia (simplified model)
        propellant_length = 2  # m 
        propellant_Ixx = current_propellant * (self.diameter / 4)**2
        propellant_Iyy = current_propellant * (propellant_length**2 / 12 + 
                                              (propellant_cg - current_cg)**2)
        
        current_Ixx = self.Ixx_dry + propellant_Ixx
        current_Iyy = self.Iyy_dry + propellant_Iyy
        current_Izz = current_Iyy  # Assume symmetry
        
        return {
            'mass': total_mass,
            'center_of_mass': current_cg,
            'Ixx': current_Ixx,
            'Iyy': current_Iyy,
            'Izz': current_Izz
        }
    
    def get_aerodynamic_coefficients(self, mach, alpha, beta=0.0):
        """Get aerodynamic coefficients for current flight conditions."""
        # Drag coefficient
        cd0 = interpolate_1d(mach, self.Cd_data['mach'], self.Cd_data['cd0'])
        cda = interpolate_1d(mach, self.Cd_data['mach'], self.Cd_data['cda'])
        cd = cd0 + cda * alpha**2
        
        # Lift coefficient (simplified model)
        cl_alpha = 2.0  # per radian
        cl = cl_alpha * alpha
        
        # Moment coefficient using dynamic CP
        cp_current = self.get_dynamic_cp(mach, alpha)
        static_margin = cp_current - self.center_of_mass_dry
        cm_alpha = -cl_alpha * static_margin / self.reference_diameter
        cm = cm_alpha * alpha
        
        return {
            'cd': cd,
            'cl': cl,
            'cm': cm,
            'cp': cp_current,
            'cn': cl,  # Normal force coefficient
            'cy': 0.0,  # Side force coefficient (simplified)
            'croll': 0.0,  # Roll moment coefficient
            'cpitch': cm,  # Pitch moment coefficient
            'cyaw': cm   # Yaw moment coefficient
        }
    
    def get_stability_margin(self, propellant_fraction_remaining):
        """Calculate static stability margin."""
        mass_props = self.get_mass_properties(propellant_fraction_remaining)
        return (self.cp_location - mass_props['center_of_mass']) / self.reference_diameter 