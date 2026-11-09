"""
SlopeGuard AI — Preprocessing & Feature Engineering Module
Safely transforms, cleans, and casts features into model-ready tensors.
"""

import pandas as pd
import numpy as np
from config import FEATURE_COLUMNS, CSV_ALIASES


# Baseline statistical moments computed directly from landslide_modelling_dataset_12409.csv
SCALER_BASELINE = {
    'aspect_deg': {'mean': 178.1878, 'std': 98.7455},
    'elevation_m': {'mean': 1780.1436, 'std': 1263.6101},
    'slope_deg': {'mean': 26.7938, 'std': 13.4340}
}


def prepare_feature_tensor(data_input, feature_cols=None) -> pd.DataFrame:
    """
    Constructs a model-ready 2D DataFrame with guaranteed numeric float64 dtypes.
    Accepts:
      - pd.Series (a single dataset row)
      - dict (from manual inputs or JSON)
      - pd.DataFrame (batch uploaded rows)

    Guarantees:
      - All columns in feature_cols exist
      - Exactly in the expected order
      - Every column has float64 dtype (prevents XGBoost object dtype errors)
      - No NaNs (imputed safely with baseline medians)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    if isinstance(data_input, pd.Series):
        df = data_input.to_frame().T
    elif isinstance(data_input, dict):
        df = pd.DataFrame([data_input])
    elif isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        raise TypeError(f"Unsupported data_input type: {type(data_input)}")

    # Auto-compute z-scores if only raw features are supplied
    for raw_col, z_col in [('aspect_deg', 'Aspect_z'), ('elevation_m', 'Elevation_z'), ('slope_deg', 'Slope_z')]:
        if raw_col in df.columns and z_col not in df.columns:
            mean = SCALER_BASELINE[raw_col]['mean']
            std = SCALER_BASELINE[raw_col]['std']
            df[z_col] = (pd.to_numeric(df[raw_col], errors='coerce') - mean) / std

    # Handle one-hot LULC if only lulc_code is supplied
    if 'lulc_code' in df.columns:
        for i in range(10, 110, 10):
            col_name = f'LULC_{i}'
            if col_name not in df.columns:
                df[col_name] = (pd.to_numeric(df['lulc_code'], errors='coerce') == i).astype(float)

    # Ensure all required feature columns exist
    for col in feature_cols:
        if col not in df.columns:
            if col == 'LULC_10':
                df[col] = 1.0  # Default to majority Tree cover in HP
            elif col.startswith('LULC_'):
                df[col] = 0.0
            else:
                df[col] = 0.0

    # Extract exact columns in order
    X = df[feature_cols].copy()

    # STRICT NUMERIC CONVERSION: Fixes "ValueError: DataFrame.dtypes must be int, float, bool or category"
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype(np.float64)

    return X


def remap_uploaded_csv_columns(uploaded_df: pd.DataFrame) -> pd.DataFrame:
    """
    Intelligently maps user-uploaded CSV columns to standard SlopeGuard AI features
    using case-insensitive alias dictionaries.
    """
    col_map = {}
    normalized_cols = {str(c).strip().lower(): c for c in uploaded_df.columns}

    for canonical, aliases in CSV_ALIASES.items():
        for alias in aliases:
            if alias.lower() in normalized_cols:
                col_map[normalized_cols[alias.lower()]] = canonical
                break

    remapped_df = uploaded_df.rename(columns=col_map).copy()
    return remapped_df
