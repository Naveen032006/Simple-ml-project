# Simple ML Model Deployment using DagsHub

## Project Overview
This project predicts student marks based on study hours using a Linear Regression machine learning model. The project demonstrates an end-to-end MLOps pipeline including model training, evaluation, remote experiment tracking on DagsHub with MLflow, local artifact persistence, CLI inference, and a Streamlit web application.

## Technologies Used
- **Python** (Core runtime)
- **Pandas** & **NumPy** (Data handling)
- **Scikit-learn** (Linear Regression model & metrics)
- **MLflow** (Experiment tracking, metrics, and model logging)
- **DagsHub** (Remote MLflow tracking server and repository hosting)
- **Joblib** (Model serialization)
- **Streamlit** (Interactive web application)

## Workflow
1. **Dataset Creation**: Generate sample study hours and exam marks dataset.
2. **Train/Test Split**: Split data into training and evaluation sets.
3. **Model Training**: Fit a Linear Regression model.
4. **Evaluation**: Compute regression metrics (MAE, MSE, $R^2$).
5. **Experiment Tracking**: Log parameters, metrics, and model artifacts to MLflow tracked remotely on DagsHub.
6. **Artifact Persistence**: Save the serialized model locally to `artifacts/model.pkl`.
7. **Inference**:
   - **CLI Mode**: Interactive terminal tool via `predict.py`.
   - **Web UI Mode**: Interactive dashboard with real-time predictions and charts via `app.py` (Streamlit).

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

This will train the model, save `artifacts/model.pkl`, and log the run to DagsHub MLflow.

## Predict Marks (CLI)

```bash
python predict.py
```

Enter study hours (0–24) when prompted, or enter `q` to quit.

## Streamlit Web App

To launch the web interface:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501` to test the interactive marks predictor.

## Project Objective
Demonstrate a complete, production-ready beginner MLOps workflow through model training, evaluation, experiment tracking, artifact storage, model reuse, and interactive web deployment.
