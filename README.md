
# Simple ML Model Deployment using DagsHub

## Project Overview
This project predicts student marks based on study hours
using a Linear Regression machine learning model.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- MLflow
- DagsHub
- Joblib

## Workflow
1. Create a sample dataset.
2. Split data into training and testing sets.
3. Train a Linear Regression model.
4. Evaluate the model using MAE, MSE, and R2.
5. Track parameters, metrics, and the model using MLflow.
6. Store experiment results remotely on DagsHub.
7. Load the saved model and predict new marks.

## Installation

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python train.py
```

## Predict Marks

```bash
python predict.py
```

## Project Objective
Demonstrate a basic MLOps workflow through
model training, evaluation, experiment tracking,
artifact storage, and model reuse.
