import numpy as np

from src.data_cleaning import (
    FEATURE_COLS,
    PATIENT_COL,
    TIME_COL,
    TARGET_COL,
)

SEQUENCE_LENGTH = 6


def create_sequences(df, patient_ids=None):
    """Create six-hour sequences for each individual patient."""

    if patient_ids is not None:
        patient_ids = set(patient_ids)
        df = df[df[PATIENT_COL].isin(patient_ids)]

    X_sequences = []
    y_labels = []

    df = df.sort_values(
        [PATIENT_COL, TIME_COL]
    )

    for _, group in df.groupby(
        PATIENT_COL,
        sort=False,
    ):
        group = group.sort_values(
            TIME_COL
        ).reset_index(drop=True)

        features = group[FEATURE_COLS].to_numpy(
            dtype=np.float32
        )

        labels = group[TARGET_COL].to_numpy(
            dtype=np.int32
        )

        hours = group[TIME_COL].to_numpy()

        for end in range(
            SEQUENCE_LENGTH - 1,
            len(group),
        ):
            start = end - SEQUENCE_LENGTH + 1

            if not np.all(
                np.diff(hours[start:end + 1]) == 1
            ):
                continue

            X_sequences.append(
                features[start:end + 1]
            )

            y_labels.append(labels[end])

    if not X_sequences:
        raise ValueError(
            "No consecutive six-hour sequences found."
        )

    return (
        np.asarray(X_sequences, dtype=np.float32),
        np.asarray(y_labels, dtype=np.int32),
    )