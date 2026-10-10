# Patient Deterioration LSTM

Patient Deterioration LSTM is a machine learning project that predicts early signs of patient decline using Long Short-Term Memory (LSTM) networks. The system analyzes six consecutive hourly observations from a patient and identifies patterns in vital signs that may indicate deterioration over time. It is designed for healthcare monitoring, early-warning systems, and clinical decision support, where timely detection can improve outcomes and help healthcare teams intervene sooner.

This project includes the full workflow for a time-series predictive model: data ingestion, data cleaning, feature engineering, sequence creation, model training, evaluation, and prediction. The model is trained on structured patient monitoring data and learns dependencies across time, enabling it to detect subtle changes in physiological indicators that may not be obvious from single-point measurements.

A simple Flask-based web application is included so users can upload a CSV file containing six hourly observations and receive a prediction result through an interactive interface. This makes the project useful for demonstrations, learning, and prototype deployment in healthcare analytics.

The application emphasizes that it is an educational and research-oriented prototype, and that predictions are not a substitute for professional clinical judgment or medical assessment. It should be used only as a decision-support tool alongside standard clinical care.

Overall, Patient Deterioration LSTM showcases how deep learning can be applied to early detection of patient deterioration, combining time-series analysis, healthcare relevance, and an accessible deployment interface in a single project.

## Features
- LSTM-based patient deterioration prediction
- Time-series processing of six hourly observations
- Data preprocessing and sequence generation
- Model training and evaluation pipeline
- Flask web app for CSV-based prediction
- Educational and research-focused design

## Project Structure
- `src/` – data processing, feature engineering, model training, and prediction logic
- `app/` – Flask web application and UI templates
- `data/` – sample and raw datasets
- `models/` – trained model and preprocessing artifacts
- `notebooks/` – experimentation and tuning notebooks

## Usage
1. Prepare a CSV file with six consecutive hourly patient observations.
2. Run the Flask app.
3. Upload the file through the web interface.
4. View the prediction result.

## Note
This project is for educational and experimental use only and should not be used as an independent medical diagnosis tool.