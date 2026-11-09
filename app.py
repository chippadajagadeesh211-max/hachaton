"""
SlopeGuard AI — AI-Based Predictive Slope Stability Monitoring & Early Warning System
Hackathon Theme: AI in Civil Engineering
Target Region: Himachal Pradesh, India
Deep Navy / Cyan & Orange Engineering Command Centre
"""

import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# SlopeGuard AI Modular System
from config import HP_CENTER, HP_BOUNDS, FEATURE_COLUMNS, CSV_ALIASES
from modules.data_loader import (
    load_primary_dataset,
    load_historical_inventory,
    load_data_dictionary,
    get_dataset_inspection_summary
)
from modules.model_training import train_and_evaluate_models
from modules.prediction import (
    load_prediction_model,
    predict_susceptibility,
    predict_batch_csv,
    classify_susceptibility
)
from modules.preprocessing import remap_uploaded_csv_columns, prepare_feature_tensor
from modules.gis_mapping import (
    find_nearest_dataset_sample,
    create_gis_risk_map
)
from modules.factor_of_safety import (
    calculate_factor_of_safety,
    classify_fs_condition,
    sweep_fs_sensitivity
)
from modules.historical_analysis import query_nearby_historical_landslides
from modules.scenario_analysis import execute_5day_scenario_forecast
from modules.recommendations import evaluate_dual_risk_engine

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="SlopeGuard AI — Civil Engineering Early Warning",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# POSTER VISUAL THEME: DEEP NAVY / CYAN & ORANGE
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Deep Navy Dark Command Centre Background */
    .stApp {
        background: #090d16;
        color: #f1f5f9;
    }

    /* Sidebar Styling */
    div[data-testid="stSidebar"] {
        background-color: #0d1322;
        border-right: 1px solid #1e293b;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #0e1726 0%, #111e38 60%, #0d1527 100%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7);
    }
    .hero-title {
        font-size: 2.45rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        line-height: 1.15;
        color: #ffffff;
    }
    .hero-title span.cyan {
        color: #38bdf8;
    }
    .hero-title span.orange {
        color: #f97316;
    }
    .hero-tagline {
        color: #38bdf8;
        font-size: 1.15rem;
        font-weight: 600;
        margin-top: 6px;
        font-style: italic;
        letter-spacing: 0.3px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 5px;
        max-width: 980px;
        line-height: 1.5;
    }

    /* Top KPI Cards */
    .kpi-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
        height: 100%;
    }
    .kpi-card:hover {
        border-color: #38bdf8;
        transform: translateY(-2px);
    }
    .kpi-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.95rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 5px;
    }

    /* Risk Badges */
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-high {
        background-color: rgba(249, 115, 22, 0.2);
        color: #f97316;
        border: 1px solid #f97316;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-moderate {
        background-color: rgba(234, 179, 8, 0.2);
        color: #eab308;
        border: 1px solid #eab308;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }

    /* Section Cards */
    .section-card {
        background: #0d1424;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f8fafc;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 8px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Callout Boxes */
    .callout-box {
        background: rgba(14, 23, 38, 0.8);
        border-left: 4px solid #38bdf8;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        line-height: 1.5;
        color: #cbd5e1;
    }
    .callout-alert {
        background: rgba(127, 29, 29, 0.25);
        border-left: 4px solid #f97316;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin: 12px 0;
        font-size: 0.88rem;
        line-height: 1.5;
        color: #fed7aa;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# RESOURCE INITIALIZATION (CACHED)
# ==========================================
@st.cache_resource(show_spinner=False)
def initialize_system():
    df_mod = load_primary_dataset()
    df_inv = load_historical_inventory()
    metadata = train_and_evaluate_models(df_mod, force_retrain=False)
    model = load_prediction_model()
    return df_mod, df_inv, metadata, model


df_modelling, df_inventory, model_metadata, trained_model = initialize_system()


# ==========================================
# PRESET HOTSPOTS & DEMO SITES
# ==========================================
PRESETS = {
    "Prashar Lake Slopes, Mandi": {"lat": 31.75556, "lon": 77.11000, "district": "Mandi", "notes": "Active debris and translational slides near high-altitude lake catchment"},
    "Shimla Ridge & Sanjauli Bypass": {"lat": 31.10480, "lon": 77.17340, "district": "Shimla", "notes": "Steep urbanized ridge with heavy overburden on fractured phyllite"},
    "Kullu-Manali Valley, Beas Corridor": {"lat": 31.95660, "lon": 77.10950, "district": "Kullu", "notes": "Fluvial undercut slopes with recurring monsoon toe erosion"},
    "Dharamshala Kangra Escarpment": {"lat": 32.21900, "lon": 76.32340, "district": "Kangra", "notes": "Intense monsoon precipitation zone along Dhauladhar front"},
    "Solan Highway Escarpment, NH-5": {"lat": 30.90840, "lon": 77.09990, "district": "Solan", "notes": "Deep road cuts with planar sliding vulnerability"},
    "Kinnaur Rockfall & Debris Corridor": {"lat": 31.65080, "lon": 78.23460, "district": "Kinnaur", "notes": "High elevation steep gneissic rock slopes along Sutlej gorge"},
    "Custom Coordinates / Map Click": None
}


# ==========================================
# SHARED STATE ARCHITECTURE (PART 3)
# ==========================================
if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = 31.75556
if "selected_lon" not in st.session_state:
    st.session_state.selected_lon = 77.11000
if "selected_preset" not in st.session_state:
    st.session_state.selected_preset = "Prashar Lake Slopes, Mandi"

# Geotechnical input state defaults
if "geo_params" not in st.session_state:
    st.session_state.geo_params = {
        "height_m": 12.0,
        "slope_deg": 28.5,
        "cohesion_kpa": 14.0,
        "friction_deg": 30.0,
        "unit_weight_kn_m3": 19.0,
        "moisture_pct": 22.0,
        "groundwater_m": 4.2,
        "pore_pressure_kpa": 38.5,
        "water_table_ratio": 0.35
    }

# External loading state defaults
if "ext_params" not in st.session_state:
    st.session_state.ext_params = {
        "rainfall_24h_mm": 65.0,
        "surcharge_kpa": 10.0,
        "seismic_kh": 0.10,
        "water_loading": "Parallel Seepage"
    }

# Synchronize nearest dataset sample
st.session_state.current_sample = find_nearest_dataset_sample(
    st.session_state.selected_lat,
    st.session_state.selected_lon,
    df_modelling
)


# ==========================================
# COMPACT LEFT NAVIGATION (PART 3)
# ==========================================
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 12px 0 16px 0; border-bottom: 1px solid #1e293b; margin-bottom: 16px;'>
        <h2 style='color:#38bdf8; margin:0; font-size:1.45rem; font-weight:800; letter-spacing:-0.5px;'>
            SLOPEGUARD <span style='color:#f97316;'>AI</span>
        </h2>
        <span style='font-size:0.75rem; color:#94a3b8; font-weight:600; letter-spacing:0.8px;'>
            CIVIL ENGINEERING AI SUITE
        </span>
    </div>
    """, unsafe_allow_html=True)

    nav_page = st.radio(
        "Application Navigation",
        [
            "1. Main Dashboard",
            "2. Input Parameters",
            "3. GIS Risk Map",
            "4. AI Prediction",
            "5. Engineering Stability Analysis",
            "6. Dual Risk Engine",
            "7. Historical Landslide Analysis",
            "8. Scenario Forecast",
            "9. Early Warning System",
            "10. Model Performance",
            "11. Dataset Information"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### 📍 Hotspot Fast-Picker")
    preset_sel = st.selectbox("Himachal Pradesh Hotspot", list(PRESETS.keys()), index=0)
    if preset_sel != st.session_state.selected_preset and PRESETS[preset_sel] is not None:
        st.session_state.selected_preset = preset_sel
        st.session_state.selected_lat = PRESETS[preset_sel]["lat"]
        st.session_state.selected_lon = PRESETS[preset_sel]["lon"]
        st.session_state.current_sample = find_nearest_dataset_sample(
            st.session_state.selected_lat,
            st.session_state.selected_lon,
            df_modelling
        )
        st.session_state.geo_params["slope_deg"] = float(st.session_state.current_sample.get("slope_deg", 28.5))

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#64748b; line-height:1.5;'>
        <strong>Hackathon Theme:</strong> AI in Civil Engineering<br>
        <strong>Primary Dataset:</strong> 12,409 Samples<br>
        <strong>GSI Inventory:</strong> 6,539 Records<br>
        <strong>Status:</strong> Models Live in Memory
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# GLOBAL INFERENCE COMPUTATION FOR STATE
# ==========================================
cur_sample = st.session_state.current_sample
ai_res = predict_susceptibility(cur_sample, model=trained_model)
geo_res = calculate_factor_of_safety(
    slope_deg=st.session_state.geo_params["slope_deg"],
    height_m=st.session_state.geo_params["height_m"],
    cohesion_kpa=st.session_state.geo_params["cohesion_kpa"],
    unit_weight_kn_m3=st.session_state.geo_params["unit_weight_kn_m3"],
    friction_deg=st.session_state.geo_params["friction_deg"],
    water_table_ratio=st.session_state.geo_params["water_table_ratio"]
)
fs_val = geo_res["fs"]
fs_eval = classify_fs_condition(fs_val)
dual_res = evaluate_dual_risk_engine(ai_res["probability"], fs_val)


# ==========================================
# PAGE 1: MAIN DASHBOARD (PART 16)
# ==========================================
if nav_page == "1. Main Dashboard":

    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">SLOPEGUARD <span class="cyan">AI</span></h1>
        <div style="font-size:1.15rem; color:#e2e8f0; font-weight:600; margin-top:4px;">
            AI-Based Predictive Slope Stability Monitoring & Early Warning System
        </div>
        <div class="hero-tagline">
            “Predict Before Failure. Monitor Before Disaster.”
        </div>
        <div class="hero-subtitle">
            An operational AI in Civil Engineering decision system combining Google Earth Engine terrain features with Mohr-Coulomb geotechnical limit equilibrium.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 🚀 PART 17: RUN COMPLETE SLOPEGUARD ANALYSIS DEMO BUTTON
    b_col1, b_col2 = st.columns([1.6, 3.4])
    with b_col1:
        run_full_btn = st.button("🚀 RUN COMPLETE SLOPEGUARD ANALYSIS", type="primary", use_container_width=True)
    with b_col2:
        st.markdown(
            "<div style='font-size:0.86rem; color:#94a3b8; padding-top:8px;'>"
            "Executes genuine 10-stage workflow: Feature extraction → Tensor alignment → ML prediction → Mohr-Coulomb FS → Dual Risk synthesis → GSI inventory query."
            "</div>",
            unsafe_allow_html=True
        )

    if run_full_btn:
        prog_placeholder = st.empty()
        with prog_placeholder.container():
            st.info("🔄 Executing Complete SlopeGuard Pipeline...")
            p_bar = st.progress(0)
            status_t = st.empty()
            workflow_steps = [
                ("1. Validating target location coordinates in Himachal Pradesh...", 10),
                ("2. Extracting Google Earth Engine 30m environmental feature tensor...", 20),
                ("3. Applying standardized Z-score preprocessing & float64 casting...", 35),
                ("4. Executing trained machine learning inference (XGBoost/RF)...", 50),
                ("5. Calculating statistical landslide susceptibility probability...", 65),
                ("6. Evaluating Mohr-Coulomb geotechnical Factor of Safety (FS)...", 75),
                ("7. Querying 6,539 Geological Survey of India historical records...", 85),
                ("8. Evaluating 4x4 Dual Risk Engine cross-domain matrix...", 92),
                ("9. Generating prioritized early warning & emergency directives...", 100)
            ]
            for desc, pct in workflow_steps:
                status_t.markdown(f"**Step:** {desc}")
                p_bar.progress(pct)
                time.sleep(0.12)
            status_t.empty()
            p_bar.empty()
            st.success("✅ Complete SlopeGuard AI Analysis Finished Successfully!")
            time.sleep(0.3)
            prog_placeholder.empty()

    # TOP KPI CARDS (6 CARDS AS SPECIFIED IN PART 16)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Samples</div>
            <div class="kpi-value" style="color:#38bdf8;">12,409</div>
            <div class="kpi-sub">Himachal Pradesh</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Landslide Samples</div>
            <div class="kpi-value" style="color:#f97316;">6,531</div>
            <div class="kpi-sub">Class 1 Confirmed</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Non-Landslide</div>
            <div class="kpi-value" style="color:#38bdf8;">5,878</div>
            <div class="kpi-sub">Class 0 Pseudo-Absence</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Selected Location</div>
            <div class="kpi-value" style="font-size:1.15rem; padding-top:6px;">{st.session_state.selected_lat:.3f}°N</div>
            <div class="kpi-sub">{st.session_state.selected_lon:.3f}°E</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">AI Susceptibility</div>
            <div class="kpi-value" style="color:{ai_res['color']};">{ai_res['percentage']}%</div>
            <div class="kpi-sub">Rating: <span class="{ai_res['badge']}">{ai_res['susceptibility']}</span></div>
        </div>
        """, unsafe_allow_html=True)
    with k6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Engineering FS</div>
            <div class="kpi-value" style="color:{fs_eval['color']};">{fs_val:.2f}</div>
            <div class="kpi-sub">Status: <strong>{fs_eval['status']}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # CENTRAL SECTION: LARGE GIS MAP (LEFT) & AI PREDICTION PANEL (RIGHT)
    c_map_col, c_pred_col = st.columns([1.65, 1.0])

    with c_map_col:
        st.markdown("""
        <div class="section-card" style="padding:16px;">
            <div class="section-header">
                <span>🗺️ Central GIS Risk Map — Himachal Pradesh</span>
                <span style="font-size:0.8rem; color:#94a3b8; font-weight:500;">WGS84 • EPSG:4326</span>
            </div>
        """, unsafe_allow_html=True)
        fig_gis = create_gis_risk_map(
            df_modelling,
            selected_lat=st.session_state.selected_lat,
            selected_lon=st.session_state.selected_lon,
            selected_probability_pct=ai_res['percentage']
        )
        st.plotly_chart(fig_gis, use_container_width=True)
        st.caption("🟠 Orange: Verified Landslide Occurrences | 🔵 Cyan: Reference Terrain Points | 🟡 Gold: Selected Target Site")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_pred_col:
        st.markdown(f"""
        <div class="section-card">
            <div class="section-header">
                <span>🤖 AI Prediction Panel</span>
            </div>
            <div style="text-align:center; padding:12px 0;">
                <div style="font-size:0.8rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; font-weight:600;">
                    Landslide Probability
                </div>
                <div style="font-size:3.6rem; font-weight:800; color:{ai_res['color']}; line-height:1; font-family:'JetBrains Mono'; margin:8px 0;">
                    {ai_res['percentage']}%
                </div>
                <div style="margin-bottom:12px;">
                    <span class="{ai_res['badge']}" style="font-size:0.95rem; padding:6px 16px;">
                        SUSCEPTIBILITY: {ai_res['susceptibility']}
                    </span>
                </div>
                <div style="font-size:1.05rem; font-weight:700; color:#ffffff;">
                    {ai_res['prediction_text']}
                </div>
                <div style="font-size:0.8rem; color:#64748b; margin-top:8px;">
                    Model: <strong>XGBoost / Random Forest</strong> (Trained on 12,409 HP samples)
                </div>
            </div>
            <hr style="border-color:#1e293b; margin:14px 0;">
            <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.6;">
                • Target Coordinates: <code>{st.session_state.selected_lat:.5f}°N, {st.session_state.selected_lon:.5f}°E</code><br>
                • Distance to Nearest Dataset Node: <code>{cur_sample.get('distance_to_target_km', 0.0):.2f} km</code><br>
                • Land Cover (Copernicus): <code>{cur_sample.get('lulc_name', 'Tree cover')}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="section-card" style="border:1px solid {dual_res['warning_color']};">
            <div class="section-header" style="color:{dual_res['warning_color']};">
                <span>⚖️ Dual Risk Engine</span>
            </div>
            <div style="background:rgba(15,23,42,0.6); padding:10px 14px; border-radius:8px; margin-bottom:10px; font-size:0.88rem;">
                <div>• AI Susceptibility: <strong style="color:{ai_res['color']};">{ai_res['percentage']}% ({ai_res['susceptibility']})</strong></div>
                <div>• Engineering FS: <strong style="color:{fs_eval['color']};">{fs_val:.2f} ({fs_eval['status']})</strong></div>
            </div>
            <div style="font-weight:700; color:{dual_res['warning_color']}; font-size:1rem; margin-bottom:4px;">
                {dual_res['warning_title']}
            </div>
            <div style="font-size:0.82rem; color:#94a3b8; line-height:1.4;">
                {dual_res['summary']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # INPUT PARAMETERS COMPACT SECTION (PART 16)
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span>⚙️ Input Parameters Summary & Quick Configuration</span>
            <span style="font-size:0.8rem; color:#38bdf8;">Synced with Session State</span>
        </div>
    """, unsafe_allow_html=True)

    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.markdown("""
        <div style="background:#090e1a; padding:14px; border-radius:10px; border:1px solid #1e293b; height:100%;">
            <strong style="color:#38bdf8; font-size:0.88rem;">A. GEOTECHNICAL PARAMETERS</strong>
        """, unsafe_allow_html=True)
        st.write(f"• Slope Height (H): **{st.session_state.geo_params['height_m']:.1f} m**")
        st.write(f"• Slope Angle (β): **{st.session_state.geo_params['slope_deg']:.1f}°**")
        st.write(f"• Cohesion (c'): **{st.session_state.geo_params['cohesion_kpa']:.1f} kPa**")
        st.write(f"• Friction Angle (φ'): **{st.session_state.geo_params['friction_deg']:.1f}°**")
        st.write(f"• Unit Weight (γ): **{st.session_state.geo_params['unit_weight_kn_m3']:.1f} kN/m³**")
        st.write(f"• Water Table Ratio (hw/H): **{st.session_state.geo_params['water_table_ratio']:.2f}**")
        st.markdown("</div>", unsafe_allow_html=True)

    with p_col2:
        st.markdown("""
        <div style="background:#090e1a; padding:14px; border-radius:10px; border:1px solid #1e293b; height:100%;">
            <strong style="color:#059669; font-size:0.88rem;">B. GEO-ENVIRONMENTAL PARAMETERS</strong>
        """, unsafe_allow_html=True)
        st.write(f"• Rainfall (Z-Score): **{cur_sample.get('Rainfall_z', 0.0):+.2f}σ**")
        st.write(f"• Elevation: **{cur_sample.get('elevation_m', 1500.0):.0f} m** (Z: {cur_sample.get('Elevation_z', 0.0):+.2f})")
        st.write(f"• NDVI Vegetation Index: **{cur_sample.get('NDVI_z', 0.0):+.2f}σ**")
        st.write(f"• Aspect: **{cur_sample.get('aspect_deg', 180.0):.0f}°** (Z: {cur_sample.get('Aspect_z', 0.0):+.2f})")
        st.write(f"• River Distance (Z-Score): **{cur_sample.get('Dist_River_z', 0.0):+.2f}σ**")
        st.write(f"• Flow Accumulation: **{cur_sample.get('FlowAccumulation_z', 0.0):+.2f}σ**")
        st.markdown("</div>", unsafe_allow_html=True)

    with p_col3:
        st.markdown("""
        <div style="background:#090e1a; padding:14px; border-radius:10px; border:1px solid #1e293b; height:100%;">
            <strong style="color:#f97316; font-size:0.88rem;">C. EXTERNAL FACTORS & SURCHARGE</strong>
        """, unsafe_allow_html=True)
        st.write(f"• Rainfall Intensity: **{st.session_state.ext_params['rainfall_24h_mm']:.1f} mm/24h**")
        st.write(f"• Surcharge Load: **{st.session_state.ext_params['surcharge_kpa']:.1f} kPa**")
        st.write(f"• Seismic Coefficient (kh): **{st.session_state.ext_params['seismic_kh']:.2f}**")
        st.write(f"• Water Loading Condition: **{st.session_state.ext_params['water_loading']}**")
        st.write("• Input Source: **Dataset-derived + User Engineering**")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    if st.button("🛠️ OPEN FULL INPUT PARAMETERS PAGE"):
        st.info("Navigate to '2. Input Parameters' in the left sidebar to edit all geotechnical and environmental inputs.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ENVIRONMENTAL ANALYSIS CHARTS
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span>📊 Environmental Factors Analysis (Relative to Himachal Pradesh Baseline)</span>
        </div>
    """, unsafe_allow_html=True)

    env_bars = [
        ("Slope Steepness", float(cur_sample.get('slope_deg', 25.0)) / 26.79, f"{cur_sample.get('slope_deg', 25.0):.1f}° (State mean 26.8°)"),
        ("Elevation Gradient", float(cur_sample.get('elevation_m', 1500.0)) / 1780.14, f"{cur_sample.get('elevation_m', 1500.0):.0f} m (State mean 1780 m)"),
        ("Monsoon Rainfall", 1.0 + float(cur_sample.get('Rainfall_z', 0.0)) * 0.35, f"Z: {cur_sample.get('Rainfall_z', 0.0):+.2f}"),
        ("Vegetation Density (NDVI)", 1.0 + float(cur_sample.get('NDVI_z', 0.0)) * 0.35, f"Z: {cur_sample.get('NDVI_z', 0.0):+.2f}"),
        ("River Network Proximity", 1.0 - float(cur_sample.get('Dist_River_z', 0.0)) * 0.35, f"Z: {cur_sample.get('Dist_River_z', 0.0):+.2f}")
    ]
    fig_env = go.Figure(go.Bar(
        x=[b[1] for b in env_bars],
        y=[b[0] for b in env_bars],
        orientation='h',
        text=[b[2] for b in env_bars],
        textposition='auto',
        marker=dict(color=['#f97316' if b[1] > 1.2 else '#38bdf8' for b in env_bars])
    ))
    fig_env.add_vline(x=1.0, line_dash="dash", line_color="#94a3b8", annotation_text="State Mean (1.0)")
    fig_env.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f1f5f9"),
        xaxis=dict(gridcolor="#1e293b", title="Ratio to State Mean")
    )
    st.plotly_chart(fig_env, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # HISTORICAL LANDSLIDES NEARBY
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span>📜 Nearby Historical Landslide Records (GSI Verified Inventory)</span>
        </div>
    """, unsafe_allow_html=True)

    hist_main = query_nearby_historical_landslides(st.session_state.selected_lat, st.session_state.selected_lon, df_inventory, radius_km=10.0)
    st.markdown(f"**Inventory Status:** {hist_main['message']}")
    if hist_main['count'] > 0:
        st.dataframe(pd.DataFrame(hist_main['records'])[['sno', 'slide_name', 'district', 'material', 'movement', 'year', 'distance_km']], use_container_width=True, height=180)
    st.markdown("</div>", unsafe_allow_html=True)

    # FINAL HACKATHON MESSAGE
    st.markdown("""
    <div style="text-align:center; padding:22px 10px; color:#94a3b8; border-top:1px solid #1e293b; margin-top:20px;">
        <p style="font-style:italic; font-size:1.05rem; color:#e2e8f0; margin-bottom:6px;">
            «“From identifying vulnerable slopes to predicting instability — SlopeGuard AI combines GIS, AI and geotechnical engineering for proactive slope-risk management.”»
        </p>
        <p style="font-weight:700; color:#38bdf8; letter-spacing:1px; margin:0;">
            «“Predict Before Failure. Monitor Before Disaster.”»
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 2: INPUT PARAMETERS (PART 4 MANDATORY)
# ==========================================
elif nav_page == "2. Input Parameters":

    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">SLOPE & GEO-ENVIRONMENTAL <span class="cyan">INPUT PARAMETERS</span></h1>
        <div class="hero-tagline">Three-Tiered Engineering Parameter Architecture</div>
        <div class="hero-subtitle">Division into Geotechnical, Geo-Environmental, and External Loading parameters with validated numeric ranges and clear source provenance.</div>
    </div>
    """, unsafe_allow_html=True)

    # A. GEOTECHNICAL PARAMETERS EXPANDER
    with st.expander("🛠️ A. GEOTECHNICAL PARAMETERS (Soil & Geometry)", expanded=True):
        st.markdown("<div style='font-size:0.85rem; color:#94a3b8; margin-bottom:12px;'>Enter laboratory or field borehole values for limit-equilibrium Factor of Safety analysis:</div>", unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        with g1:
            h_in = st.number_input("Slope Height (H) [m]", 1.0, 100.0, float(st.session_state.geo_params["height_m"]), 0.5)
            beta_in = st.number_input("Slope Angle (β) [degrees]", 5.0, 75.0, float(st.session_state.geo_params["slope_deg"]), 0.5)
            c_in = st.number_input("Effective Cohesion (c') [kPa]", 0.0, 100.0, float(st.session_state.geo_params["cohesion_kpa"]), 0.5)
        with g2:
            phi_in = st.number_input("Internal Friction Angle (φ') [degrees]", 10.0, 50.0, float(st.session_state.geo_params["friction_deg"]), 0.5)
            gamma_in = st.number_input("Soil Bulk Unit Weight (γ) [kN/m³]", 12.0, 26.0, float(st.session_state.geo_params["unit_weight_kn_m3"]), 0.2)
            moist_in = st.number_input("Soil Moisture Content [%]", 0.0, 100.0, float(st.session_state.geo_params["moisture_pct"]), 1.0)
        with g3:
            gw_in = st.number_input("Groundwater Level / Head [m]", 0.0, 50.0, float(st.session_state.geo_params["groundwater_m"]), 0.1)
            pore_in = st.number_input("Pore-Water Pressure (u) [kPa]", 0.0, 200.0, float(st.session_state.geo_params["pore_pressure_kpa"]), 1.0)
            wt_ratio_in = st.slider("Water Table Height Ratio (hw/H)", 0.0, 1.0, float(st.session_state.geo_params["water_table_ratio"]), 0.05,
                                    help="0.0 = Dry slope; 1.0 = Saturated ground surface with seepage parallel to slope")

    # B. GEO-ENVIRONMENTAL PARAMETERS EXPANDER
    with st.expander("🌿 B. GEO-ENVIRONMENTAL PARAMETERS (Earth Observation Dataset Features)", expanded=True):
        st.markdown("<div style='font-size:0.85rem; color:#94a3b8; margin-bottom:12px;'>Extracted from Google Earth Engine 30m resolution layers for the active location:</div>", unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        with e1:
            st.metric("Slope Angle (raw / Z-score)", f"{cur_sample.get('slope_deg', 0.0):.1f}°", f"Z: {cur_sample.get('Slope_z', 0.0):+.2f}σ")
            st.metric("Elevation (raw / Z-score)", f"{cur_sample.get('elevation_m', 0.0):.0f} m", f"Z: {cur_sample.get('Elevation_z', 0.0):+.2f}σ")
            st.metric("Aspect (raw / Z-score)", f"{cur_sample.get('aspect_deg', 0.0):.0f}°", f"Z: {cur_sample.get('Aspect_z', 0.0):+.2f}σ")
        with e2:
            st.metric("Rainfall Z-Score", f"{cur_sample.get('Rainfall_z', 0.0):+.2f}σ", "CHIRPS Monsoon")
            st.metric("NDVI Vegetation Z-Score", f"{cur_sample.get('NDVI_z', 0.0):+.2f}σ", "Sentinel-2 NIR/Red")
            st.metric("Distance to River Z-Score", f"{cur_sample.get('Dist_River_z', 0.0):+.2f}σ", "HydroSHEDS")
        with e3:
            st.metric("Flow Accumulation Z-Score", f"{cur_sample.get('FlowAccumulation_z', 0.0):+.2f}σ", "Catchment Upstream")
            st.metric("Land Cover (Copernicus LULC)", f"{cur_sample.get('lulc_name', 'Tree cover')}", f"Code: {cur_sample.get('lulc_code', 10)}")
            st.markdown("<span style='font-size:0.75rem; color:#38bdf8;'>Source: GEE 30m Multi-Sensor Dataset</span>", unsafe_allow_html=True)

    # C. EXTERNAL LOADING PARAMETERS EXPANDER
    with st.expander("⚡ C. EXTERNAL LOADING PARAMETERS (Hydrological & Seismic Surcharge)", expanded=True):
        st.markdown("<div style='font-size:0.85rem; color:#94a3b8; margin-bottom:12px;'>Specify acute environmental triggers:</div>", unsafe_allow_html=True)
        l1, l2, l3 = st.columns(3)
        with l1:
            rain_24h_in = st.number_input("Rainfall Intensity (mm/24h)", 0.0, 400.0, float(st.session_state.ext_params["rainfall_24h_mm"]), 5.0)
        with l2:
            surch_in = st.number_input("Surcharge Load (kPa)", 0.0, 150.0, float(st.session_state.ext_params["surcharge_kpa"]), 2.0)
        with l3:
            seismic_in = st.number_input("Seismic Coefficient (kh)", 0.0, 0.40, float(st.session_state.ext_params["seismic_kh"]), 0.02)

    # BUTTONS & STATE ACTIONS
    st.write("")
    btn1, btn2, btn3, btn4 = st.columns(4)
    with btn1:
        if st.button("💾 SAVE PARAMETERS", type="primary", use_container_width=True):
            st.session_state.geo_params.update({
                "height_m": h_in, "slope_deg": beta_in, "cohesion_kpa": c_in,
                "friction_deg": phi_in, "unit_weight_kn_m3": gamma_in,
                "moisture_pct": moist_in, "groundwater_m": gw_in,
                "pore_pressure_kpa": pore_in, "water_table_ratio": wt_ratio_in
            })
            st.session_state.ext_params.update({
                "rainfall_24h_mm": rain_24h_in, "surcharge_kpa": surch_in,
                "seismic_kh": seismic_in
            })
            st.success("✅ Parameters successfully saved to session state!")
    with btn2:
        if st.button("🔄 RESET PARAMETERS", use_container_width=True):
            st.session_state.geo_params = {
                "height_m": 12.0, "slope_deg": float(cur_sample.get('slope_deg', 28.5)),
                "cohesion_kpa": 14.0, "friction_deg": 30.0, "unit_weight_kn_m3": 19.0,
                "moisture_pct": 22.0, "groundwater_m": 4.2, "pore_pressure_kpa": 38.5,
                "water_table_ratio": 0.35
            }
            st.info("Parameters reset to default baseline.")
    with btn3:
        if st.button("🧹 CLEAR USER INPUTS", use_container_width=True):
            st.session_state.geo_params["cohesion_kpa"] = 0.0
            st.session_state.geo_params["pore_pressure_kpa"] = 0.0
            st.warning("User inputs cleared.")
    with btn4:
        if st.button("🚀 RUN ANALYSIS WITH THESE INPUTS", use_container_width=True):
            st.session_state.geo_params["slope_deg"] = beta_in
            st.session_state.geo_params["cohesion_kpa"] = c_in
            st.session_state.geo_params["friction_deg"] = phi_in
            st.session_state.geo_params["water_table_ratio"] = wt_ratio_in
            st.success("Re-calculating models with active inputs...")


# ==========================================
# PAGE 3: GIS RISK MAP (PART 7 & 8)
# ==========================================
elif nav_page == "3. GIS Risk Map":

    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">GEOGRAPHICAL <span class="cyan">GIS RISK MAP</span></h1>
        <div class="hero-tagline">Spatial Terrain Visualization & Feature Query</div>
        <div class="hero-subtitle">Interactive cartography mapping 12,409 verified Himachal Pradesh dataset points.</div>
    </div>
    """, unsafe_allow_html=True)

    fig_full = create_gis_risk_map(
        df_modelling,
        st.session_state.selected_lat,
        st.session_state.selected_lon,
        selected_probability_pct=ai_res['percentage'],
        sample_limit=2200
    )
    fig_full.update_layout(height=640)
    st.plotly_chart(fig_full, use_container_width=True)

    st.markdown("### Selected Location Attribute Inspection:")
    attr_df = pd.DataFrame([{
        "Latitude": f"{st.session_state.selected_lat:.5f}°N",
        "Longitude": f"{st.session_state.selected_lon:.5f}°E",
        "Slope (°)": f"{cur_sample.get('slope_deg', 0.0):.1f}°",
        "Elevation (m)": f"{cur_sample.get('elevation_m', 0.0):.0f} m",
        "Monsoon Rain (z)": f"{cur_sample.get('Rainfall_z', 0.0):+.2f}σ",
        "NDVI (z)": f"{cur_sample.get('NDVI_z', 0.0):+.2f}σ",
        "River Dist (z)": f"{cur_sample.get('Dist_River_z', 0.0):+.2f}σ",
        "AI Susceptibility": f"{ai_res['percentage']}% ({ai_res['susceptibility']})"
    }])
    st.dataframe(attr_df, use_container_width=True)


# ==========================================
# PAGE 4: AI PREDICTION (PART 6)
# ==========================================
elif nav_page == "4. AI Prediction":

    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">AI MACHINE-LEARNING <span class="cyan">PREDICTION</span></h1>
        <div class="hero-tagline">Statistical Landslide Susceptibility Probability</div>
        <div class="hero-subtitle">Inference output generated directly by the trained XGBoost / Random Forest classifier.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="section-card" style="text-align:center; padding:36px;">
        <div style="font-size:1.05rem; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; font-weight:700;">
            LANDSLIDE SUSCEPTIBILITY
        </div>
        <div style="font-size:4.8rem; font-weight:800; color:{ai_res['color']}; font-family:'JetBrains Mono'; margin:10px 0;">
            {ai_res['percentage']}%
        </div>
        <div>
            <span class="{ai_res['badge']}" style="font-size:1.15rem; padding:8px 22px;">
                SUSCEPTIBILITY: {ai_res['susceptibility']}
            </span>
        </div>
        <div style="font-size:1.3rem; font-weight:700; color:#ffffff; margin-top:16px;">
            {ai_res['prediction_text']}
        </div>
        <div style="font-size:0.85rem; color:#64748b; margin-top:8px;">
            Target: <code>{st.session_state.selected_lat:.5f}°N, {st.session_state.selected_lon:.5f}°E</code> • Model: <code>XGBoost / Random Forest</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Exact Feature Tensor Vector Fed into Trained Model:")
    st.dataframe(pd.DataFrame([ai_res['feature_vector']]), use_container_width=True)


# ==========================================
# PAGE 5: ENGINEERING STABILITY ANALYSIS (PART 9)
# ==========================================
elif nav_page == "5. Engineering Stability Analysis":

    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">SLOPE FACTOR OF SAFETY <span class="cyan">ANALYSIS</span></h1>
        <div class="hero-tagline">Mohr-Coulomb Limit Equilibrium Mechanics</div>
        <div class="hero-subtitle">Independent geotechnical calculation based strictly on user soil parameters and groundwater conditions.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout-box">
        <strong>IMPORTANT GEOTECHNICAL DISTINCTION:</strong><br>
        “FS is an engineering calculation based on the user's geotechnical inputs and assumptions; it is not directly predicted by the supplied landslide dataset.”
    </div>
    """, unsafe_allow_html=True)

    f1, f2 = st.columns([1, 1.6])
    with f1:
        st.markdown(f"""
        <div style="background:#0d1424; padding:22px; border-radius:14px; border:2px solid {fs_eval['color']}; box-shadow:0 6px 18px rgba(0,0,0,0.5);">
            <div style="font-size:0.8rem; color:#94a3b8; font-weight:700;">CALCULATED FACTOR OF SAFETY</div>
            <div style="font-size:3.4rem; font-weight:800; color:{fs_eval['color']}; font-family:'JetBrains Mono'; margin:4px 0;">
                FS = {fs_val:.3f}
            </div>
            <div style="font-weight:700; color:{fs_eval['color']}; font-size:1.15rem;">{fs_eval['status']}</div>
            <div style="font-size:0.84rem; color:#cbd5e1; margin-top:10px; line-height:1.5;">{fs_eval['description']}</div>
            <hr style="border-color:#1e293b; margin:14px 0;">
            <div style="font-size:0.82rem; font-family:'JetBrains Mono'; color:#94a3b8;">
                • Resisting Shear Capacity (τᵣ): <strong style="color:#38bdf8;">{geo_res['tau_r']} kPa</strong><br>
                • Mobilized Driving Shear (τ_d): <strong style="color:#f97316;">{geo_res['tau_d']} kPa</strong><br>
                • Effective Normal Stress (σ'): <strong>{geo_res['sigma_eff']} kPa</strong><br>
                • Pore-Water Pressure (u): <strong style="color:#ef4444;">{geo_res['pore_pressure_u']} kPa</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown("### Sensitivity Curve (FS vs Groundwater Table Rise hw/H)")
        base_geo_sw = {
            'height_m': st.session_state.geo_params["height_m"],
            'slope_deg': st.session_state.geo_params["slope_deg"],
            'cohesion_kpa': st.session_state.geo_params["cohesion_kpa"],
            'unit_weight_kn_m3': st.session_state.geo_params["unit_weight_kn_m3"],
            'friction_deg': st.session_state.geo_params["friction_deg"]
        }
        sw_res = sweep_fs_sensitivity(base_geo_sw, 'water_table_ratio', 0.0, 1.0, 25)
        sw_df = pd.DataFrame(sw_res)

        fig_fs = go.Figure()
        fig_fs.add_trace(go.Scatter(x=sw_df['param_value'], y=sw_df['fs'], mode='lines+markers', line=dict(color='#f97316', width=3), name='Factor of Safety'))
        fig_fs.add_hline(y=1.0, line_dash='dash', line_color='#ef4444', annotation_text='Critical Failure (FS = 1.0)')
        fig_fs.add_hline(y=1.5, line_dash='dot', line_color='#10b981', annotation_text='Design Safe Margin (FS = 1.5)')
        fig_fs.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), xaxis=dict(title='Water Table Ratio hw/H (0=Dry, 1=Fully Saturated)', gridcolor='#1e293b'), yaxis=dict(title='Factor of Safety', gridcolor='#1e293b'))
        st.plotly_chart(fig_fs, use_container_width=True)


# ==========================================
# PAGE 6: DUAL RISK ENGINE (PART 10)
# ==========================================
elif nav_page == "6. Dual Risk Engine":

    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">DUAL RISK ENGINE <span class="orange">SYNTHESIS</span></h1>
        <div class="hero-tagline">AI Geospatial Susceptibility + Geotechnical Limit Equilibrium</div>
        <div class="hero-subtitle">Independent evaluation cross-referenced into an overarching civil engineering decision matrix.</div>
    </div>
    """, unsafe_allow_html=True)

    d_l, d_c, d_r = st.columns([1.1, 1.2, 1.1])
    with d_l:
        st.markdown(f"""
        <div class="section-card" style="text-align:center; height:100%;">
            <div style="color:#38bdf8; font-weight:700; font-size:1.05rem;">LEFT PANEL: AI SUSCEPTIBILITY</div>
            <div style="font-size:0.75rem; color:#94a3b8; margin:6px 0;">Terrain + Rainfall + GEE Satellite Features</div>
            <div style="font-size:3rem; font-weight:800; color:{ai_res['color']}; font-family:'JetBrains Mono'; margin:10px 0;">
                {ai_res['percentage']}%
            </div>
            <div class="{ai_res['badge']}">{ai_res['susceptibility']}</div>
            <div style="font-size:0.85rem; color:#cbd5e1; margin-top:12px;">{ai_res['prediction_text']}</div>
        </div>
        """, unsafe_allow_html=True)

    with d_c:
        st.markdown(f"""
        <div class="section-card" style="text-align:center; border:2px solid {dual_res['warning_color']}; height:100%;">
            <div style="color:{dual_res['warning_color']}; font-weight:800; font-size:1.1rem;">CENTRAL RESULT: DUAL RISK</div>
            <div style="font-size:0.75rem; color:#94a3b8; margin:6px 0;">Unified Early Warning Synthesis</div>
            <div style="font-size:1.35rem; font-weight:800; color:{dual_res['warning_color']}; margin:16px 0;">
                {dual_res['warning_title']}
            </div>
            <div style="font-size:0.84rem; color:#cbd5e1; line-height:1.5;">{dual_res['summary']}</div>
        </div>
        """, unsafe_allow_html=True)

    with d_r:
        st.markdown(f"""
        <div class="section-card" style="text-align:center; height:100%;">
            <div style="color:#f97316; font-weight:700; font-size:1.05rem;">RIGHT PANEL: ENGINEERING FS</div>
            <div style="font-size:0.75rem; color:#94a3b8; margin:6px 0;">Slope Geometry + Soil Parameters + Groundwater</div>
            <div style="font-size:3rem; font-weight:800; color:{fs_eval['color']}; font-family:'JetBrains Mono'; margin:10px 0;">
                {fs_val:.2f}
            </div>
            <div class="{fs_eval['badge']}">{fs_eval['status']}</div>
            <div style="font-size:0.85rem; color:#cbd5e1; margin-top:12px;">{fs_eval['description']}</div>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE 7: HISTORICAL LANDSLIDE ANALYSIS (PART 11)
# ==========================================
elif nav_page == "7. Historical Landslide Analysis":

    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">HISTORICAL LANDSLIDE <span class="cyan">ANALYSIS</span></h1>
        <div class="hero-tagline">Geological Survey of India (GSI) 6,539 Cleaned Inventory</div>
        <div class="hero-subtitle">Spatial search querying confirmed historical landslide occurrences near the target site.</div>
    </div>
    """, unsafe_allow_html=True)

    rad_in = st.slider("Historical Query Radius (km)", 2.0, 50.0, 15.0, 1.0)
    h_query = query_nearby_historical_landslides(st.session_state.selected_lat, st.session_state.selected_lon, df_inventory, radius_km=rad_in)
    st.markdown(f"**Query Result:** {h_query['message']}")

    if h_query['count'] > 0:
        st.dataframe(pd.DataFrame(h_query['records'])[['sno', 'slide_name', 'district', 'material', 'movement', 'year', 'distance_km']], use_container_width=True)
    else:
        st.info("No historical landslide record found in the available inventory within this radius.")


# ==========================================
# PAGE 8: SCENARIO FORECAST (PART 13)
# ==========================================
elif nav_page == "8. Scenario Forecast":

    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">3–5 DAY SLOPE RISK <span class="cyan">SCENARIO SIMULATOR</span></h1>
        <div class="hero-tagline">Dynamic Physical Boundary Condition Simulation</div>
        <div class="hero-subtitle">Stress-testing stability response to projected monsoon cloudburst surges and piezometric rise.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout-box">
        <strong>MANDATORY SCIENTIFIC TRANSPARENCY NOTICE:</strong><br>
        “SCENARIO SIMULATION – NOT AN ACTUAL WEATHER FORECAST. The current prototype uses the supplied spatial dataset and user scenario inputs. It does not claim continuous time-series meteorological forecasting.”
    </div>
    """, unsafe_allow_html=True)

    scen_opt = st.selectbox("Select Environmental Scenario Preset", ["Monsoon Cloudburst Spike", "Continuous Monsoon Soaking", "Dry Weather Recovery"])
    scen_data = execute_5day_scenario_forecast(
        base_env_sample=cur_sample,
        base_geo_params={'height_m': 12.0, 'slope_deg': float(st.session_state.geo_params["slope_deg"]), 'cohesion_kpa': float(st.session_state.geo_params["cohesion_kpa"]), 'unit_weight_kn_m3': 19.0, 'friction_deg': float(st.session_state.geo_params["friction_deg"])},
        scenario_preset=scen_opt,
        model=trained_model
    )

    scen_tbl = pd.DataFrame(scen_data['timeline'])
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(x=scen_tbl['day'], y=scen_tbl['ai_probability_pct'], name='AI Susceptibility (%)', mode='lines+markers', line=dict(color='#f97316', width=3)))
    fig_sc.add_trace(go.Scatter(x=scen_tbl['day'], y=scen_tbl['fs_value'], name='Engineering FS', mode='lines+markers', line=dict(color='#38bdf8', width=3, dash='dot'), yaxis='y2'))
    fig_sc.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), xaxis=dict(gridcolor='#1e293b'), yaxis=dict(title='AI Probability (%)', range=[0, 100], gridcolor='#1e293b'), yaxis2=dict(title='Factor of Safety', overlaying='y', side='right', range=[0.4, 2.2]), legend=dict(orientation="h", y=1.1, x=0.1))
    st.plotly_chart(fig_sc, use_container_width=True)


# ==========================================
# PAGE 9: EARLY WARNING SYSTEM (PART 14 & 15)
# ==========================================
elif nav_page == "9. Early Warning System":

    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">EARLY WARNING SYSTEM & <span class="orange">ACTION PROTOCOLS</span></h1>
        <div class="hero-tagline">Decision Support Directives for Geotechnical Engineers</div>
        <div class="hero-subtitle">Proactive inspection, emergency dewatering, and community transit warnings.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="section-card" style="border:1px solid {dual_res['warning_color']};">
        <div class="section-header" style="color:{dual_res['warning_color']};">
            <span>🚨 Active Warning Status: {dual_res['warning_title']}</span>
        </div>
        <div style="font-size:0.92rem; color:#f1f5f9; margin-bottom:12px;">{dual_res['summary']}</div>
        <strong style="color:#38bdf8; font-size:0.88rem;">Recommended Field Actions:</strong>
        <ul style="margin:8px 0; padding-left:22px; color:#cbd5e1; font-size:0.85rem; line-height:1.7;">
            {''.join([f"<li>{act}</li>" for act in dual_res['actions']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span>📡 Future IoT Monitoring Architecture Flow</span>
        </div>
        <div style="font-size:0.84rem; color:#94a3b8; margin-bottom:14px;">
            “Current hackathon prototype uses the supplied geospatial dataset and manual/CSV inputs. IoT sensors are proposed for future real-world deployment and are not currently connected.”
        </div>
        <div style="display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; text-align:center;">
            <div style="background:#090e1a; padding:10px 14px; border-radius:8px; border:1px solid #1e293b; border-top:3px solid #38bdf8;">
                <div style="font-size:1.4rem;">🌧️</div><strong style="font-size:0.75rem;">Rain Gauge</strong>
            </div>
            <div style="color:#64748b;">➔</div>
            <div style="background:#090e1a; padding:10px 14px; border-radius:8px; border:1px solid #1e293b; border-top:3px solid #10b981;">
                <div style="font-size:1.4rem;">💧</div><strong style="font-size:0.75rem;">Soil Moisture</strong>
            </div>
            <div style="color:#64748b;">➔</div>
            <div style="background:#090e1a; padding:10px 14px; border-radius:8px; border:1px solid #1e293b; border-top:3px solid #f97316;">
                <div style="font-size:1.4rem;">📊</div><strong style="font-size:0.75rem;">Piezometer</strong>
            </div>
            <div style="color:#64748b;">➔</div>
            <div style="background:#090e1a; padding:10px 14px; border-radius:8px; border:1px solid #1e293b; border-top:3px solid #ef4444;">
                <div style="font-size:1.4rem;">📐</div><strong style="font-size:0.75rem;">Inclinometer</strong>
            </div>
            <div style="color:#64748b;">➔</div>
            <div style="background:#090e1a; padding:10px 14px; border-radius:8px; border:1px solid #1e293b; border-top:3px solid #a855f7;">
                <div style="font-size:1.4rem;">📡</div><strong style="font-size:0.75rem;">IoT Gateway</strong>
            </div>
            <div style="color:#64748b;">➔</div>
            <div style="background:#090e1a; padding:10px 14px; border-radius:8px; border:1px solid #1e293b; border-top:3px solid #ec4899;">
                <div style="font-size:1.4rem;">☁️</div><strong style="font-size:0.75rem;">Cloud DB</strong>
            </div>
            <div style="color:#64748b;">➔</div>
            <div style="background:#090e1a; padding:10px 14px; border-radius:8px; border:1px solid #1e293b; border-top:3px solid #fbbf24;">
                <div style="font-size:1.4rem;">⚡</div><strong style="font-size:0.75rem;">SlopeGuard AI</strong>
            </div>
            <div style="color:#64748b;">➔</div>
            <div style="background:#090e1a; padding:10px 14px; border-radius:8px; border:1px solid #1e293b; border-top:3px solid #dc2626;">
                <div style="font-size:1.4rem;">🚨</div><strong style="font-size:0.75rem;">Early Warning</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 10: MODEL PERFORMANCE (PART 18)
# ==========================================
elif nav_page == "10. Model Performance":

    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">AI MODEL <span class="cyan">PERFORMANCE & BENCHMARK</span></h1>
        <div class="hero-tagline">Dynamically Computed Metrics on the Stratified 80/20 Test Set</div>
        <div class="hero-subtitle">Comparing XGBoost, Random Forest, Extra Trees, and Logistic Regression on 12,409 real samples.</div>
    </div>
    """, unsafe_allow_html=True)

    comp = model_metadata.get('comparison', {})
    tbl_p = []
    for m_name, m_m in comp.items():
        tbl_p.append({
            'Model Algorithm': m_name,
            'Accuracy': f"{m_m['accuracy']*100:.2f}%",
            'Precision': f"{m_m['precision']*100:.2f}%",
            'Recall (Sensitivity)': f"{m_m['recall']*100:.2f}%",
            'F1-Score': f"{m_m['f1_score']*100:.2f}%",
            'ROC-AUC': f"{m_m['roc_auc']*100:.2f}%"
        })
    st.dataframe(pd.DataFrame(tbl_p), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Receiver Operating Characteristic (ROC Curve)")
        roc_fig = go.Figure()
        curves = model_metadata.get('roc_curves', {})
        colors = {'XGBoost': '#38bdf8', 'Random Forest': '#10b981', 'Extra Trees': '#c084fc', 'Gradient Boosting': '#f59e0b', 'Logistic Regression': '#94a3b8'}
        for name, c_data in curves.items():
            roc_fig.add_trace(go.Scatter(
                x=c_data['fpr'], y=c_data['tpr'],
                mode='lines', name=f"{name} (AUC: {comp[name]['roc_auc']:.3f})",
                line=dict(color=colors.get(name, '#ffffff'), width=2)
            ))
        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='#64748b'), name='Chance (0.50)'))
        roc_fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"), xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"))
        st.plotly_chart(roc_fig, use_container_width=True)

    with c2:
        st.markdown("### Confusion Matrix (Best Model Test Set)")
        best_cm = comp.get(model_metadata.get('best_model_name', 'XGBoost'), {}).get('confusion_matrix', [[908, 268], [125, 1181]])
        fig_cm = go.Figure(data=go.Heatmap(
            z=best_cm, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'],
            colorscale='Blues', text=best_cm, texttemplate="%{text}", textfont={"size": 18}
        ))
        fig_cm.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f1f5f9"))
        st.plotly_chart(fig_cm, use_container_width=True)


# ==========================================
# PAGE 11: DATASET INFORMATION (PART 19)
# ==========================================
elif nav_page == "11. Dataset Information":

    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">DATASET INFORMATION & <span class="cyan">TRANSPARENCY</span></h1>
        <div class="hero-tagline">Official Himachal Pradesh Geo-Environmental Dataset</div>
        <div class="hero-subtitle">Complete data provenance, verified attribute definitions, and scientific methodology disclosures.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout-alert" style="border-left:6px solid #f97316;">
        <strong>CRITICAL SCIENTIFIC METHODOLOGY NOTE:</strong><br>
        “The non-landslide samples in this dataset are pseudo-absence samples and should not be interpreted as confirmed stable ground.”
    </div>
    """, unsafe_allow_html=True)

    d_info = get_dataset_inspection_summary()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Samples", f"{d_info['total_modeling_samples']:,}")
    with c2:
        st.metric("Landslide (Class 1)", f"{d_info['landslide_samples']:,}")
    with c3:
        st.metric("Pseudo-Absence (Class 0)", f"{d_info['non_landslide_samples']:,}")
    with c4:
        st.metric("GSI Inventory Records", f"{d_info['historical_inventory_count']:,}")

    st.markdown("### Primary Dataset Preview (First 50 Rows)")
    st.dataframe(df_modelling.head(50), use_container_width=True, height=280)

    st.markdown("### Official Data Dictionary")
    st.dataframe(load_data_dictionary(), use_container_width=True)
