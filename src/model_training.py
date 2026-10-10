from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_lstm.keras"
)


def load_saved_model():
    """Load the previously trained LSTM. No training occurs."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}\n"
            "Copy your trained model into the models folder "
            "or update MODEL_PATH."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Existing trained model loaded.")
    model.summary()

    return model


if __name__ == "__main__":
    load_saved_model()