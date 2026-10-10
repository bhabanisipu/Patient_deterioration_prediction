from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_lstm.keras"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test_sequences.npz"
)


def evaluate_model():
    """Evaluate the saved model on held-out test sequences."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test sequences not found: {TEST_PATH}. "
            "Save X_test and y_test from your experiment first."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    with np.load(TEST_PATH) as data:
        X_test = data["X_test"]
        y_test = data["y_test"]

    probabilities = model.predict(
        X_test,
        verbose=0,
    ).ravel()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print(
        "Test accuracy:",
        round(
            accuracy_score(y_test, predictions),
            4,
        ),
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    evaluate_model()