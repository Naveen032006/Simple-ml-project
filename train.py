
import os
import sys
import joblib
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn

# Configure UTF-8 streams on Windows to avoid UnicodeEncodeError when MLflow logs Unicode emojis
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import DAGSHUB_USERNAME, DAGSHUB_REPO_NAME, MODEL_PATH


def main():
    # 1. Connect to DagsHub and configure MLflow tracking
    dagshub.init(
        repo_owner=DAGSHUB_USERNAME,
        repo_name=DAGSHUB_REPO_NAME,
        mlflow=True
    )

    # 2. Create a sample dataset
    data = pd.DataFrame({
        "hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                  2, 4, 6, 8, 10],
        "marks": [12, 20, 29, 38, 46, 55, 64, 73, 82, 91,
                  21, 39, 56, 74, 92]
    })

    X = data[["hours"]]
    y = data["marks"]

    # 3. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Start an MLflow experiment run
    with mlflow.start_run(run_name="student-marks-linear-regression"):

        # 5. Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 6. Make test predictions
        predictions = model.predict(X_test)

        # 7. Evaluate the model
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        # 8. Log parameters and metrics to MLflow
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("MAE", float(mae))
        mlflow.log_metric("MSE", float(mse))
        mlflow.log_metric("R2", float(r2))

        # 9. Save model locally
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        # 10. Log model to DagsHub's MLflow artifact store
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model"
        )

        print("\nModel training completed successfully!")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"R2 Score: {r2:.4f}")
        print(f"Local model saved at: {MODEL_PATH}")
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
