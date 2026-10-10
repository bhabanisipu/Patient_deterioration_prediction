from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hospital_deterioration_hourly_panel.csv"
)


def load_data(path=DATA_PATH):
    """Load the existing dataset."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)

    return df


if __name__ == "__main__":
    df = load_data()
    print(df.head())