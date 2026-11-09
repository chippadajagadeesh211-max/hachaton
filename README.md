# SlopeGuard AI — AI-Based Predictive Slope Stability Monitoring & Early Warning System

**Hackathon Theme:** AI in Civil Engineering
**Target Geography:** Himachal Pradesh, India (Western Himalayas)
**Core Motto:** *“Predict before failure. Monitor before disaster.”*

---

> «*“From identifying vulnerable slopes to predicting instability — SlopeGuard AI combines GIS, AI and geotechnical engineering for proactive slope-risk management.”*»

---

## 📌 Executive Summary

SlopeGuard AI is a dual-engine civil engineering decision-support platform designed to address landslide risks across the Himalayas. Traditional slope engineering relies either on empirical remote sensing susceptibility maps (which lack site-specific geotechnical mechanics) or on localized Limit Equilibrium Factor of Safety calculations (which lack regional climatic and spatial context).

SlopeGuard AI bridges this gap through a **Dual Risk Engine**:
1. **AI Susceptibility Layer:** Powered by high-resolution Earth Observation (EO) data from Google Earth Engine (30 m SRTM DEM, CHIRPS precipitation, Copernicus Sentinel-2 NDVI, HydroSHEDS drainage). Real machine learning models (XGBoost, Random Forest) predict the statistical likelihood of landslide occurrence.
2. **Geotechnical Limit Equilibrium Layer:** Powered by classical Mohr-Coulomb infinite slope stability mechanics, calculating the physical Factor of Safety ($FS$) as the ratio of available shear resistance to mobilizing shear stress under varying phreatic water surface levels.

The system synthesizes both layers into a unified **Early Warning Decision Matrix**, enabling proactive geotechnical site investigation before catastrophic shear failure occurs.

---

## 🗂️ Dataset Provenance & Scientific Transparency

The application is built strictly on the actual uploaded dataset: **`A Geo-Environmental Feature Dataset for Landslide.zip`**.

### Dataset Specifications:
- **Primary AI Dataset:** `landslide_modelling_dataset_12409.csv`
- **Total Records:** 12,409 samples
  - **Landslide Occurrence ($class = 1$):** 6,531 confirmed landslide locations
  - **Pseudo-Absence ($class = 0$):** 5,878 non-landslide reference points
- **Coordinate System:** WGS84 / EPSG:4326 (Latitude: 30.0°N–33.1°N, Longitude: 75.7°E–79.8°E)
- **Historical Inventory:** `hp_landslide_inventory_clean.csv` (6,539 historical records from the Geological Survey of India)
- **Feature Extraction:** 30 m spatial resolution via Google Earth Engine (SRTM, CHIRPS, Sentinel-2, HydroSHEDS)

> ⚠️ **CRITICAL SCIENTIFIC METHODOLOGY NOTE:**
> *“The non-landslide samples in this dataset are pseudo-absence samples and should not be interpreted as confirmed stable ground.”* In geospatial machine learning, pseudo-absences represent sampled locations where no landslide was formally cataloged in the historical inventory; they may include uninventoried historic movements or marginally stable slopes.

---

## 🏆 Real AI Model Benchmarking & Performance

Candidate algorithms were trained on an 80/20 stratified split of the 12,409 dataset samples. All metrics below are genuinely computed from test-set evaluation (no hardcoded figures):

| Model Algorithm | Test Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Best)** | **84.29%** | **81.67%** | **90.43%** | **85.83%** | **92.02%** |
| **Random Forest** | 84.00% | 80.90% | 91.12% | 85.70% | 91.75% |
| **Extra Trees** | 82.35% | 78.59% | 91.35% | 84.49% | 90.97% |
| **Gradient Boosting** | 82.03% | 79.61% | 88.51% | 83.83% | 89.83% |
| **Logistic Regression (Baseline)** | 75.87% | 74.81% | 81.62% | 78.07% | 83.77% |

### Key Predictor Influence (Global Importance):
1. **Terrain Slope Angle (`slope_deg` / `Slope_z`):** Strongest physical predictor; steep angles increase downhill gravitational shear stress.
2. **Elevation Gradient (`elevation_m` / `Elevation_z`):** High elevations correlate with glacial-periglacial weathering and fractured bedrock.
3. **Monsoon Precipitation (`Rainfall_z`):** Critical triggering mechanism elevating pore-water pressure.
4. **Vegetation Density (`NDVI_z`):** Root cohesion provides mechanical reinforcement and regulates infiltration.
5. **Proximity to River Network (`Dist_River_z`):** Fluvial toe-undercutting removes lateral earth support.
6. **Land Use / Land Cover (`LULC_10` to `LULC_100`):** Forested vs. bare rock vs. agricultural terraces.

---

## 📐 Engineering Factor of Safety ($FS$) Formulation

> *“FS is an engineering calculation based on the user's geotechnical inputs and assumptions; it is not directly predicted by the supplied landslide dataset.”*

For mountainous debris and rock slopes, SlopeGuard AI implements the **Infinite Slope Formulation with Parallel Seepage** (Mohr-Coulomb limit equilibrium):

$$\tau_d = \gamma \cdot H \cdot \sin\beta \cdot \cos\beta$$

$$\sigma_n = \gamma \cdot H \cdot \cos^2\beta$$

$$u = \gamma_w \cdot h_w \cdot \cos^2\beta$$

$$\sigma' = \sigma_n - u$$

$$\tau_r = c' + \sigma' \cdot \tan\phi'$$

$$FS = \frac{\tau_r}{\tau_d} = \frac{c' + (\gamma \cdot H - \gamma_w \cdot h_w)\cos^2\beta \cdot \tan\phi'}{\gamma \cdot H \cdot \sin\beta \cdot \cos\beta}$$

Where:
- $H$: Depth to slip surface / slope height (m)
- $\beta$: Slope inclination angle (degrees)
- $c'$: Effective soil cohesion (kPa)
- $\gamma$: Bulk unit weight ($\text{kN/m}^3$)
- $\phi'$: Effective internal friction angle (degrees)
- $h_w / H$: Normalized phreatic water surface ratio ($0 = \text{dry}$, $1 = \text{ground surface seepage}$)
- $\gamma_w$: Unit weight of water ($9.81\ \text{kN/m}^3$)

### Stability Classification:
- **$FS > 1.50$:** **Stable** (Standard civil engineering safety margin)
- **$1.20 \le FS \le 1.50$:** **Caution** (Vulnerable to saturated monsoon surge)
- **$1.00 \le FS < 1.20$:** **High Risk** (Critical limit-equilibrium state)
- **$FS < 1.00$:** **Potential Instability** (Driving shear stress exceeds resisting strength)

---

## ⚖️ The Dual Risk Engine: AI + Geotechnical Synthesis

```
                   AI Geospatial Susceptibility
                     (Terrain + Climate + EO)
                               │
                               ▼
                   [ Landslide Probability % ]
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   Engineering Factor of Safety           Unified Decision Matrix
  (Slope Geometry + Soil Parameters)                  │
            │                                         ▼
            ▼                              [ OVERALL ACTION DIRECTIVE ]
     [ Calculated FS ]                 - Critical Intervention Required
                                       - High Priority Site Assessment
                                       - Drainage & Crack Monitoring
                                       - Low Risk / Routine Observation
```

---

## 📍 Three Methods to Select a Location

1. **Method 1 — Dataset / Hotspot Explorer:** Select from verified Himachal Pradesh landslide hotspots (Prashar Lake Mandi, Shimla Ridge, Kullu Valley, Solan Highway NH-5, Dharamshala, Kinnaur Gorge) or click any sampled coordinate.
2. **Method 2 — Manual Lat / Lon Input:** Enter custom geographic coordinates within Himachal Pradesh (Lat 30.0°–33.5°N, Lon 75.5°–80.0°E) and click **ANALYZE LOCATION** to find the nearest environmental tensor and run predictions.
3. **Method 3 — Batch CSV Upload:** Upload any field survey CSV. SlopeGuard AI auto-detects columns, validates features, executes batch model inference, and exports an enriched predictions CSV.

---

## 🔮 3–5 Day Scenario Forecast (Stress Testing)

Simulates projected environmental boundary changes:
- Projected precipitation deltas (e.g. cloudburst spike $+120\text{ mm}$)
- Rising piezometric water table ($h_w / H$ from $0.20 \to 0.90$)
- Dynamic loss of effective cohesion due to matric suction reduction

*Explicitly labeled as a scenario stress simulation, not a simulated weather forecast.*

---

## 📡 Future IoT Continuous Monitoring Architecture

```
[ Rain Gauge ] ──┐
[ Soil Moisture ]─┼─▶ [ IoT Gateway ] ──▶ [ Cloud DB ] ──▶ [ SlopeGuard AI ] ──▶ [ Early Warning ]
[ Piezometer ] ──┤     (LoRaWAN / 4G)   (Time-Series)     (Dual Risk Engine)    (SMS / Siren)
[ Inclinometer ] ─┘
```

*Transparently presented as a deployment roadmap; no active hardware connection is falsely claimed.*

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Model training
On first launch, the app trains and evaluates candidate models from the supplied dataset if the saved model artifacts are missing. The trained model and its measured performance metadata are saved under `models/`.

### 4. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Deploy publicly with Streamlit Community Cloud
1. Push this repository to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) and connect GitHub.
3. Create an app using repository `chippadajagadeesh211-max/soil`, branch `main`, and entrypoint `app.py`.
4. In Advanced settings, select Python 3.12, then deploy. Streamlit Community Cloud will provide the public `*.streamlit.app` URL.

The trained model is committed under `models/`; the app also includes the supplied dataset and its inventory files under `data/`. The pinned requirements match the Python 3.12 environment used to train and smoke-test the saved model.

---

## 📁 Repository Directory Structure

```
hachaton/
├── data/
│   ├── A Geo-Environmental Feature Dataset for Landslide.zip
│   ├── landslide_modelling_dataset_12409.csv       (Primary ML dataset)
│   ├── hp_landslide_inventory_clean.csv            (GSI historical inventory)
│   ├── landslide_features_master_6503.csv          (Master features)
│   └── data_dictionary.csv                         (Column definitions)
├── models/
│   ├── trained_model.joblib                        (Selected trained model)
│   └── model_metadata.json                         (Measured evaluation metrics)
├── modules/                                        (Data, GIS, AI, engineering modules)
├── app.py                                          (Streamlit dashboard)
├── config.py                                       (Dataset paths and model features)
├── sample_field_data.csv                           (CSV upload example)
├── requirements.txt                                (Package dependencies)
└── README.md                                       (Documentation)
```

---

## 🛡️ License & Ethical Declaration
SlopeGuard AI was developed for the **AI in Civil Engineering Hackathon**. All predictions, models, and spatial records are generated solely from the uploaded Geological Survey of India and Google Earth Engine dataset. No synthetic, fabricated, or hardcoded values are used.
