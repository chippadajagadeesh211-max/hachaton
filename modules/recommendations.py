"""
SlopeGuard AI — Early Warning & Geotechnical Recommendations
Synthesizes the Dual Risk Engine (AI Susceptibility + Engineering FS)
into prioritized civil engineering action directives.
"""


def evaluate_dual_risk_engine(ai_probability: float, fs_value: float) -> dict:
    """
    Synthesizes the Dual Risk Matrix.
    Combines:
      - AI Geospatial Susceptibility (0.0 to 1.0)
      - Geotechnical Factor of Safety (FS)
    """
    # AI Classification level
    if ai_probability >= 0.75:
        ai_level = 'VERY HIGH'
        ai_weight = 3
    elif ai_probability >= 0.55:
        ai_level = 'HIGH'
        ai_weight = 2
    elif ai_probability >= 0.35:
        ai_level = 'MODERATE'
        ai_weight = 1
    else:
        ai_level = 'LOW'
        ai_weight = 0

    # Geotechnical FS Classification
    if fs_value < 1.00:
        fs_level = 'POTENTIAL INSTABILITY'
        fs_weight = 3
    elif fs_value < 1.20:
        fs_level = 'HIGH RISK'
        fs_weight = 2
    elif fs_value <= 1.50:
        fs_level = 'CAUTION'
        fs_weight = 1
    else:
        fs_level = 'STABLE'
        fs_weight = 0

    composite_score = ai_weight + fs_weight

    if composite_score >= 5 or (fs_value < 1.00 and ai_probability >= 0.55):
        warning_title = 'CRITICAL ALERT — IMMEDIATE GEOTECHNICAL INTERVENTION'
        warning_color = '#ef4444'
        priority = 'CRITICAL'
        summary = (
            "Severe convergence of regional terrain susceptibility and limit-equilibrium failure threshold. "
            "Imminent shear displacement or active tension crack propagation suspected."
        )
        actions = [
            "Evacuate or restrict traffic along downhill roadway or infrastructure corridors immediately.",
            "Deploy emergency plastic sheeting over crest tension cracks to prevent surface runoff infiltration.",
            "Install automated vibrating wire piezometers and wireless tiltmeters with 1-minute alert triggers.",
            "Commence emergency vacuum horizontal dewatering drains to lower the localized phreatic line.",
            "Notify the Himachal Pradesh State Disaster Management Authority (HPSDMA) and district engineering cell."
        ]
    elif composite_score >= 3 or fs_value < 1.15:
        warning_title = 'HIGH PRIORITY — ACTIVE FIELD ASSESSMENT & MONITORING'
        warning_color = '#f97316'
        priority = 'HIGH'
        summary = (
            "High susceptibility terrain combined with marginal geotechnical safety margin. "
            "Vulnerable to rapid stability deterioration during peak monsoon storms."
        )
        actions = [
            "Conduct walkover visual inspection for crown scarps, bulging toe terraces, and tilted tree trunks.",
            "Clear debris and rockfall blockages from concrete catchwater drains and roadside culverts.",
            "Install manual displacement monitoring pegs and continuous borehole inclinometer tubes.",
            "Commission detailed geotechnical site investigation including core recovery and triaxial shear testing.",
            "Design contingency slope stabilization measures (e.g. soil nailed walls or rockfall protection netting)."
        ]
    elif composite_score >= 2 or ai_probability >= 0.45:
        warning_title = 'MODERATE CAUTION — ROUTINE SURVEILLANCE & DRAINAGE CARE'
        warning_color = '#eab308'
        priority = 'MEDIUM'
        summary = (
            "Moderate hazard profile. Slope is currently holding equilibrium, but drainage blockages could degrade safety."
        )
        actions = [
            "Inspect interceptor berms and unlined catchwater drains prior to seasonal monsoon rains.",
            "Maintain weep hole functionality on existing masonry or gabion retaining walls.",
            "Establish baseline photographic records of slope face to monitor long-term creep."
        ]
    else:
        warning_title = 'LOW RISK — STABLE PROFILE UNDER TESTED PARAMETERS'
        warning_color = '#10b981'
        priority = 'LOW'
        summary = (
            "Favorable topography and adequate geotechnical safety margin. Standard maintenance required."
        )
        actions = [
            "Continue standard annual highway maintenance and culvert inspections.",
            "Preserve vegetative slope cover to mitigate superficial raindrop erosion."
        ]

    return {
        'ai_level': ai_level,
        'ai_probability': round(float(ai_probability), 4),
        'fs_level': fs_level,
        'fs_value': round(float(fs_value), 3),
        'warning_title': warning_title,
        'warning_color': warning_color,
        'priority': priority,
        'summary': summary,
        'actions': actions
    }
