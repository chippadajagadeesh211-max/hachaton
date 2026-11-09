"""
SlopeGuard AI — 3 to 5 Day Scenario Forecast Engine
Simulates dynamic physical stress testing under projected precipitation and groundwater surges.
"""

from modules.factor_of_safety import calculate_factor_of_safety, classify_fs_condition
from modules.prediction import predict_susceptibility


def execute_5day_scenario_forecast(
    base_env_sample: dict,
    base_geo_params: dict,
    scenario_preset: str = 'Monsoon Cloudburst Spike',
    model=None
) -> dict:
    """
    Evaluates risk response under simulated boundary condition progressions.
    """
    days = [
        'Day 1 (Pre-Storm Baseline)',
        'Day 2 (+24h Onset)',
        'Day 3 (+48h Cloudburst Peak)',
        'Day 4 (+72h Drainage Lag)',
        'Day 5 (+96h Residual Pore Pressure)'
    ]

    if scenario_preset == 'Monsoon Cloudburst Spike':
        rain_deltas = [0.0, 1.2, 2.8, 1.4, 0.4]  # Z-score shifts
        wt_ratios = [0.20, 0.45, 0.90, 0.70, 0.45]
        desc = "Simulates sudden intense Himalayan cloudburst event followed by slow drainage dissipation."
    elif scenario_preset == 'Continuous Monsoon Soaking':
        rain_deltas = [0.2, 0.8, 1.5, 2.1, 2.6]
        wt_ratios = [0.30, 0.50, 0.70, 0.85, 0.95]
        desc = "Prolonged multi-day soaking causing progressive soil saturation and deep phreatic table rise."
    else:  # Dry Weather Drainage Recovery
        rain_deltas = [-0.2, -0.5, -0.8, -1.0, -1.2]
        wt_ratios = [0.55, 0.40, 0.28, 0.18, 0.10]
        desc = "Prolonged dry weather enabling pore-pressure dissipation and stability recovery."

    base_rain_z = float(base_env_sample.get('Rainfall_z', 0.0))

    timeline = []
    for i in range(5):
        # Update environmental input
        env_day = base_env_sample.copy()
        current_rain_z = base_rain_z + rain_deltas[i]
        env_day['Rainfall_z'] = current_rain_z

        # AI Prediction
        ai_res = predict_susceptibility(env_day, model=model)

        # Geotechnical calculation
        geo_day = base_geo_params.copy()
        current_wt = wt_ratios[i]
        geo_day['water_table_ratio'] = current_wt
        if current_wt > 0.85:
            # Saturated softening reduction in cohesion
            geo_day['cohesion_kpa'] = max(2.0, geo_day['cohesion_kpa'] * 0.85)

        fs_res = calculate_factor_of_safety(**geo_day)
        fs_cond = classify_fs_condition(fs_res['fs'])

        timeline.append({
            'day': days[i],
            'rainfall_z': round(current_rain_z, 2),
            'water_table_ratio': round(current_wt, 2),
            'pore_pressure_kpa': fs_res['pore_pressure_u'],
            'ai_probability_pct': ai_res['percentage'],
            'ai_susceptibility': ai_res['susceptibility'],
            'ai_color': ai_res['color'],
            'fs_value': fs_res['fs'],
            'fs_status': fs_cond['status'],
            'fs_color': fs_cond['color']
        })

    return {
        'scenario_preset': scenario_preset,
        'description': desc,
        'timeline': timeline
    }
