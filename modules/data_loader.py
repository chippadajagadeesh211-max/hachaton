"""
SlopeGuard AI — Data Loader Module
Inspects, validates, and loads all dataset files from the uploaded ZIP.
"""

import os
import zipfile
import pandas as pd
import streamlit as st
from config import (
    PRIMARY_DATASET_PATH,
    HISTORICAL_INVENTORY_PATH,
    DATA_DICTIONARY_PATH,
    ZIP_DATASET_PATH,
    DATA_DIR
)


def ensure_dataset_extracted():
    """
    Verifies that all required CSV files exist in data/.
    If any are missing and the ZIP is present, automatically extracts them.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    required_files = [
        PRIMARY_DATASET_PATH,
        HISTORICAL_INVENTORY_PATH,
        DATA_DICTIONARY_PATH
    ]

    all_exist = all(p.exists() for p in required_files)
    if not all_exist and ZIP_DATASET_PATH.exists():
        with zipfile.ZipFile(ZIP_DATASET_PATH, 'r') as zf:
            for member in zf.namelist():
                filename = os.path.basename(member)
                if filename:  # Skip directories
                    target = DATA_DIR / filename
                    with zf.open(member) as source, open(target, 'wb') as dest:
                        dest.write(source.read())


@st.cache_data(show_spinner=False)
def load_primary_dataset() -> pd.DataFrame:
    """
    Loads and validates the primary 12,409 sample modeling dataset.
    """
    ensure_dataset_extracted()
    if not PRIMARY_DATASET_PATH.exists():
        raise FileNotFoundError(f"Primary dataset not found at {PRIMARY_DATASET_PATH}")

    df = pd.read_csv(PRIMARY_DATASET_PATH)

    # Enforce numeric types on coordinates
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['class'] = pd.to_numeric(df['class'], errors='coerce').astype(int)

    return df


@st.cache_data(show_spinner=False)
def load_historical_inventory() -> pd.DataFrame:
    """
    Loads and validates the 6,539 Geological Survey of India historical landslide records.
    """
    ensure_dataset_extracted()
    if not HISTORICAL_INVENTORY_PATH.exists():
        raise FileNotFoundError(f"Historical inventory not found at {HISTORICAL_INVENTORY_PATH}")

    df = pd.read_csv(HISTORICAL_INVENTORY_PATH)
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)

    return df


@st.cache_data(show_spinner=False)
def load_data_dictionary() -> pd.DataFrame:
    """
    Loads the official data dictionary definitions.
    """
    ensure_dataset_extracted()
    if not DATA_DICTIONARY_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DATA_DICTIONARY_PATH)


def get_dataset_inspection_summary():
    """
    Returns verified row counts, feature types, and validation metadata.
    """
    df_mod = load_primary_dataset()
    df_inv = load_historical_inventory()

    return {
        'total_modeling_samples': len(df_mod),
        'landslide_samples': int((df_mod['class'] == 1).sum()),
        'non_landslide_samples': int((df_mod['class'] == 0).sum()),
        'modeling_columns': len(df_mod.columns),
        'historical_inventory_count': len(df_inv),
        'coordinate_system': 'WGS84 / EPSG:4326',
        'geographic_region': 'Himachal Pradesh, India'
    }
