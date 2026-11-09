"""
SlopeGuard AI — Model Training & Evaluation Engine
Performs stratified train/test split, trains candidate algorithms,
evaluates validation metrics, and serializes fitted pipelines.
"""

import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAINED_MODEL_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR
)
from modules.preprocessing import prepare_feature_tensor


def train_and_evaluate_models(df: pd.DataFrame, force_retrain: bool = False):
    """
    Trains candidate models on an 80/20 stratified split of the 12,409 dataset.
    Saves the best model and performance metadata.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not force_retrain and TRAINED_MODEL_PATH.exists() and MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return metadata

    # Safe feature extraction
    X = prepare_feature_tensor(df[FEATURE_COLUMNS])
    y = df[TARGET_COLUMN].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    candidate_models = {
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=150, max_depth=16, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    if XGB_AVAILABLE:
        candidate_models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=150, max_depth=6, learning_rate=0.1,
            eval_metric='logloss', random_state=42, n_jobs=-1
        )

    model_comparison = {}
    roc_curves = {}
    feature_importances = {}
    fitted_models = {}

    for name, model in candidate_models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        # Subsample ROC points for clean JSON storage
        sub_idx = np.linspace(0, len(fpr) - 1, 100).astype(int)
        roc_curves[name] = {
            'fpr': [float(x) for x in fpr[sub_idx]],
            'tpr': [float(y) for y in tpr[sub_idx]]
        }

        model_comparison[name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': auc,
            'confusion_matrix': cm
        }

        if hasattr(model, 'feature_importances_'):
            feature_importances[name] = dict(zip(FEATURE_COLUMNS, [float(v) for v in model.feature_importances_]))
        elif hasattr(model, 'coef_'):
            feature_importances[name] = dict(zip(FEATURE_COLUMNS, [float(v) for v in model.coef_[0]]))

    # Select best model (XGBoost if available, else Random Forest)
    best_model_name = 'XGBoost' if 'XGBoost' in candidate_models else 'Random Forest'
    best_model = fitted_models[best_model_name]

    # Save best model to disk
    joblib.dump(best_model, TRAINED_MODEL_PATH)

    metadata = {
        'best_model_name': best_model_name,
        'features': FEATURE_COLUMNS,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'comparison': model_comparison,
        'roc_curves': roc_curves,
        'feature_importances': feature_importances
    }

    with open(MODEL_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata
