
from pathlib import Path
import sys

import pandas as pd
from flask import Flask, render_template, request

# Allow Python to import modules from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction import predict_patient


app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "template"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)

# Limit uploaded CSV files to 5 MB.
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("file")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Please select a CSV file."
        elif not uploaded_file.filename.lower().endswith(".csv"):
            error = "Please upload a CSV file only."
        else:
            try:
                # Read the six-hour patient observation window.
                patient_window = pd.read_csv(uploaded_file)

                # Use the existing saved LSTM model through prediction.py.
                result = predict_patient(patient_window)

            except (ValueError, KeyError) as exc:
                error = str(exc)
            except FileNotFoundError:
                error = (
                    "A required model or preprocessing file was not found. "
                    "Check the files in your models/ directory."
                )
            except Exception:
                app.logger.exception("Prediction failed")
                error = (
                    "Prediction failed. Check the CSV columns, "
                    "model files, and application logs."
                )

    return render_template(
        "index.html",
        result=result,
        error=error,
    )


@app.errorhandler(413)
def file_too_large(_error):
    return (
        "File too large. Please upload a CSV file smaller than 5 MB.",
        413,
    )


if __name__ == "__main__":
    app.run(debug=True)