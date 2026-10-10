from src.data_cleaning import FEATURE_COLS


def select_features(df):
    """Select the clinical features in the expected order."""

    missing_cols = [
        col for col in FEATURE_COLS
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing feature columns: {missing_cols}"
        )

    return df[FEATURE_COLS].copy()