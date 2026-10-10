import numpy as np
import pandas as pd

PATIENT_COL = "patient_id"
TIME_COL = "hour_from_admission"
TARGET_COL = "deterioration_next_12h"

FEATURE_COLS = [
    "heart_rate",
    "respiratory_rate",
    "spo2_pct",
    "temperature_c",
    "systolic_bp",
    "diastolic_bp",
    "oxygen_flow",
    "wbc_count",
    "lactate",
    "creatinine",
    "crp_level",
    "hemoglobin",
    "age",
    "comorbidity_index",
]


def clean_data(df):
    """Validate columns and clean invalid records."""

    required_cols = [
        PATIENT_COL,
        TIME_COL,
        TARGET_COL,
        *FEATURE_COLS,
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns: {missing_cols}"
        )

    df = df[required_cols].copy()

    numeric_cols = [
        TIME_COL,
        TARGET_COL,
        *FEATURE_COLS,
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=[
            PATIENT_COL,
            TIME_COL,
            TARGET_COL,
        ]
    )

    df = df.drop_duplicates(
        subset=[
            PATIENT_COL,
            TIME_COL,
        ],
        keep="last",
    )

    df = df[
        df[TARGET_COL].isin([0, 1])
    ].copy()

    df[TIME_COL] = df[TIME_COL].astype(int)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    df = df.sort_values(
        [PATIENT_COL, TIME_COL]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid records remain after cleaning."
        )

    return df


if __name__ == "__main__":
    from src.data_ingestion import load_data

    df = clean_data(load_data())

    print(df.head())
    print("\nTarget distribution:")
    print(df[TARGET_COL].value_counts())