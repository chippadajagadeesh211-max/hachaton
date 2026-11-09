"""
SlopeGuard AI — Geotechnical Factor of Safety (FS) Engine
Calculates limit-equilibrium slope stability using Mohr-Coulomb failure criteria.
"""

import numpy as np
from config import GAMMA_WATER


def calculate_factor_of_safety(
    slope_deg: float,
    height_m: float = 12.0,
    cohesion_kpa: float = 14.0,
    unit_weight_kn_m3: float = 19.0,
    friction_deg: float = 30.0,
    water_table_ratio: float = 0.35
):
    """
    Computes Factor of Safety (FS) for translational infinite slope failure with seepage:
      FS = Resisting Shear Capacity (tau_r) / Mobilizing Driving Shear Stress (tau_d)
      tau_d = gamma * H * sin(beta) * cos(beta)
      sigma_n = gamma * H * cos^2(beta)
      u = gamma_w * hw * cos^2(beta)   where hw = water_table_ratio * H
      sigma_eff = max(0.0, sigma_n - u)
      tau_r = c' + sigma_eff * tan(phi')
      FS = tau_r / tau_d
    """
    beta_rad = np.radians(max(0.1, min(89.0, float(slope_deg))))
    phi_rad = np.radians(max(0.0, min(60.0, float(friction_deg))))
    H = max(0.5, float(height_m))
    gamma = max(10.0, float(unit_weight_kn_m3))
    c = max(0.0, float(cohesion_kpa))
    m = max(0.0, min(1.0, float(water_table_ratio)))

    sin_b = np.sin(beta_rad)
    cos_b = np.cos(beta_rad)
    cos2_b = cos_b ** 2
    tan_phi = np.tan(phi_rad)

    # Driving shear stress along the potential slip surface (kPa)
    tau_d = gamma * H * sin_b * cos_b

    # Total normal stress (kPa)
    sigma_n = gamma * H * cos2_b

    # Pore water pressure u (kPa)
    u = GAMMA_WATER * (m * H) * cos2_b

    # Effective normal stress (kPa)
    sigma_eff = max(0.0, sigma_n - u)

    # Resisting shear stress (Mohr-Coulomb criterion) (kPa)
    tau_r = c + sigma_eff * tan_phi

    if tau_d <= 1e-4:
        fs = 99.0
    else:
        fs = tau_r / tau_d

    return {
        'fs': round(float(fs), 3),
        'tau_d': round(float(tau_d), 2),
        'tau_r': round(float(tau_r), 2),
        'sigma_n': round(float(sigma_n), 2),
        'pore_pressure_u': round(float(u), 2),
        'sigma_eff': round(float(sigma_eff), 2),
        'slope_deg': slope_deg,
        'height_m': height_m,
        'cohesion_kpa': cohesion_kpa,
        'unit_weight_kn_m3': unit_weight_kn_m3,
        'friction_deg': friction_deg,
        'water_table_ratio': water_table_ratio
    }


def classify_fs_condition(fs: float):
    """
    Categorizes Factor of Safety into engineering stability states:
      FS > 1.50: Stable (Standard design threshold)
      1.20 <= FS <= 1.50: Caution (Marginal safety margin)
      1.00 <= FS < 1.20: High Risk (Critical equilibrium)
      FS < 1.00: Potential Instability (Imminent failure)
    """
    if fs >= 1.50:
        return {
            'status': 'Stable',
            'code': 'STABLE',
            'color': '#10b981',
            'badge': 'badge-low',
            'description': 'Slope possesses an adequate geotechnical safety margin under specified parameters.'
        }
    elif fs >= 1.20:
        return {
            'status': 'Caution',
            'code': 'CAUTION',
            'color': '#f59e0b',
            'badge': 'badge-moderate',
            'description': 'Marginally stable. Highly vulnerable to intense rainfall infiltration and transient pore-pressure spikes.'
        }
    elif fs >= 1.00:
        return {
            'status': 'High Risk',
            'code': 'HIGH_RISK',
            'color': '#f97316',
            'badge': 'badge-high',
            'description': 'Near-critical limit equilibrium. Slight reduction in shear strength or groundwater rise may trigger shear failure.'
        }
    else:
        return {
            'status': 'Potential Instability',
            'code': 'UNSTABLE',
            'color': '#ef4444',
            'badge': 'badge-critical',
            'description': 'Mobilized shear stress exceeds available resisting shear strength. Active deformation or progressive failure likely.'
        }


def sweep_fs_sensitivity(base_params: dict, param_key: str, min_v: float, max_v: float, steps: int = 25):
    """
    Parametric sweep of a geotechnical parameter across a specified range.
    """
    vals = np.linspace(min_v, max_v, steps)
    results = []
    for v in vals:
        p = base_params.copy()
        p[param_key] = v
        out = calculate_factor_of_safety(**p)
        results.append({'param_value': float(v), 'fs': out['fs']})
    return results
