"""
SlopeGuard AI — Prediction Engine
Performs real machine learning inference for individual samples and batch CSVs.
"""

import joblib
import pandas as pd
import numpy as np
import streamlit as st
from config import TRAINED_MODEL_PATH, MODEL_METADATA_PATH, FEATURE_COLUMNS
from modules.preprocessing import prepare_feature_tensor


@st.cache_resource(show_spinner=False)
def load_prediction_model():
    """
    Loads the trained model pipeline into memory.
    """
    if not TRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {TRAINED_MODEL_PATH}. Run training first.")
    model = joblib.load(TRAINED_MODEL_PATH)
    return model


def classify_susceptibility(probability: float):
    """
    Categorizes AI Landslide Probability into standard hazard classes:
      Prob >= 0.75: VERY HIGH (Red)
      0.55 <= Prob < 0.75: HIGH (Orange)
      0.35 <= Prob < 0.55: MODERATE (Yellow)
      Prob < 0.35: LOW (Green)
    """
    if probability >= 0.75:
        return {
            'level': 'VERY HIGH',
            'color': '#ef4444',
            'badge': 'badge-critical',
            'prediction_text': 'Potential Landslide-Prone Area',
            'is_prone': True
        }
    elif probability >= 0.55:
        return {
            'level': 'HIGH',
            'color': '#f97316',
            'badge': 'badge-high',
            'prediction_text': 'Potential Landslide-Prone Area',
            'is_prone': True
        }
    elif probability >= 0.35:
        return {
            'level': 'MODERATE',
            'color': '#eab308',
            'badge': 'badge-moderate',
            'prediction_text': 'Lower Landslide Susceptibility',
            'is_prone': False
        }
    else:
        return {
            'level': 'LOW',
            'color': '#10b981',
            'badge': 'badge-low',
            'prediction_text': 'Lower Landslide Susceptibility',
            'is_prone': False
        }


def predict_susceptibility(data_input, model=None):
    """
    Predicts landslide susceptibility probability for a single sample.
    Guarantees strict float64 numeric casting.
    """
    if model is None:
        model = load_prediction_model()

    # Prepares tensor with strictly float64 dtypes (resolves XGBoost object dtype errors)
    X = prepare_feature_tensor(data_input)

    proba = float(model.predict_proba(X)[0, 1])
    pred_class = int(model.predict(X)[0])
    classification = classify_susceptibility(proba)

    return {
        'probability': proba,
        'percentage': round(proba * 100.0, 1),
        'predicted_class': pred_class,
        'susceptibility': classification['level'],
        'color': classification['color'],
        'badge': classification['badge'],
        'prediction_text': classification['prediction_text'],
        'is_prone': classification['is_prone'],
        'feature_vector': X.iloc[0].to_dict()
    }


def predict_batch_csv(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Executes batch inference over an entire DataFrame of locations.
    """
    if model is None:
        model = load_prediction_model()

    X = prepare_feature_tensor(df)
    probs = model.predict_proba(X)[:, 1]

    results = []
    for p in probs:
        c = classify_susceptibility(float(p))
        results.append({
            'ai_probability': round(float(p), 4),
            'ai_probability_pct': round(float(p) * 100.0, 1),
            'ai_susceptibility': c['level'],
            'ai_prediction': c['prediction_text']
        })

    res_df = pd.DataFrame(results)
    return pd.concat([df.reset_index(drop=True), res_df], axis=1)
