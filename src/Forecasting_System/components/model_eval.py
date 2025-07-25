import os
import sys
from urllib.parse import urlparse
import numpy as np
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv
from src.Forecasting_System.utils.utils import load_object

import pickle

# Load credentials
load_dotenv()

class ModelEvaluation:
    def __init__(self):
        # Set DagsHub MLflow Tracking URI
        mlflow.set_tracking_uri("https://dagshub.com/BhaveshNikam09/Retail-demand-forecasting.mlflow")

        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")

        if not username or not password:
            raise Exception("❌ Missing DagsHub credentials in .env")

        mlflow.set_experiment("Retail-Demand-Forecasting")

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_evaluation(self, X_train, X_test, y_train, y_test):
        try:
            model_path = os.path.join("artifacts", "catboost_main_model.cbm")
            model = load_object(model_path)

            with mlflow.start_run():
                predicted_qualities = model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # ✅ Save model manually and log it as an artifact
                os.makedirs("mlruns_artifacts", exist_ok=True)
                local_model_path = os.path.join("mlruns_artifacts", "catboost_model.cbm")
                model.save_model(local_model_path)

                mlflow.log_artifact(local_model_path, artifact_path="model")

                print("✅ Model saved and logged manually to DagsHub.")

        except Exception as e:
            print(f"❌ Exception during model evaluation: {e}")
            raise e
