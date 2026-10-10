from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_cleaning import (
    FEATURE_COLS,
    PATIENT_COL,
    TIME_COL,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_lstm.keras"
)

PREPROCESSOR_PATH = (
    PROJECT_ROOT
    / "models"
    / "preprocessing.joblib"
)

SEQUENCE_LENGTH = 6


def load_artifacts():
    """Load the existing model and preprocessing artifacts."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing file not found: "
            f"{PREPROCESSOR_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    preprocessing = joblib.load(
        PREPROCESSOR_PATH
    )

    expected_keys = {
        "imputer",
        "scaler",
        "clinical_cols",
        "sequence_length",
    }

    missing_keys = expected_keys - set(
        preprocessing.keys()
    )

    if missing_keys:
        raise ValueError(
            "Saved preprocessing is missing keys: "
            f"{missing_keys}"
        )

    return model, preprocessing


def predict_patient(patient_window, threshold=0.5):
    """
    Predict using six consecutive hourly observations.

    patient_window must contain:
    patient_id, hour_from_admission, and the 14 clinical features.
    """

    required_cols = [
        PATIENT_COL,
        TIME_COL,
        *FEATURE_COLS,
    ]

    missing_cols = [
        col for col in required_cols
        if col not in patient_window.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing input columns: {missing_cols}"
        )

    patient_window = (
        patient_window[required_cols]
        .copy()
        .sort_values(TIME_COL)
    )

    if len(patient_window) != SEQUENCE_LENGTH:
        raise ValueError(
            "Exactly six hourly observations are required."
        )

    if (
        patient_window[PATIENT_COL].isna().any()
        or patient_window[PATIENT_COL].nunique() != 1
    ):
        raise ValueError(
            "All six rows must belong to the same patient."
        )

    hours = pd.to_numeric(
        patient_window[TIME_COL],
        errors="coerce",
    ).to_numpy()

    if (
        np.isnan(hours).any()
        or not np.all(np.diff(hours) == 1)
    ):
        raise ValueError(
            "Observations must be six consecutive hours."
        )

    model, preprocessing = load_artifacts()

    if preprocessing["clinical_cols"] != FEATURE_COLS:
        raise ValueError(
            "Feature order differs from the saved preprocessing."
        )

    if preprocessing["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError(
            "Sequence length differs from the saved preprocessing."
        )

    # Use the SAME fitted imputer and scaler from the experiment.
    X = (
        patient_window[FEATURE_COLS]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=float)
    )

    X = preprocessing["imputer"].transform(X)
    X = preprocessing["scaler"].transform(X)

    X = X.reshape(
        1,
        SEQUENCE_LENGTH,
        len(FEATURE_COLS),
    )

    # Inference only; no model training.
    probability = float(
        model.predict(X, verbose=0).ravel()[0]
    )

    predicted_class = int(
        probability >= threshold
    )

    return {
        "risk_probability": round(probability, 4),
        "predicted_class": predicted_class,
        "prediction": (
            "Positive deterioration prediction"
            if predicted_class == 1
            else "Negative deterioration prediction"
        ),
    }


if __name__ == "__main__":
    print("Prediction module ready.")
    print("Use predict_patient(patient_window) to make predictions.")