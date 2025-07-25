import os
import sys
import pandas as pd
import mlflow
import pickle
from Forecasting_System.logger import logging
from Forecasting_System.exception import CustomException
from Forecasting_System.utils.common import save_object
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(true, predicted):
    mse = mean_squared_error(true, predicted)
    mae = mean_absolute_error(true, predicted)
    rmse = mean_squared_error(true, predicted, squared=False)
    r2 = r2_score(true, predicted)

    return {"mse": mse, "mae": mae, "rmse": rmse, "r2_score": r2}


def initiate_model_evaluation(X_train, X_test, y_train, y_test):
    try:
        from Forecasting_System.components.data_transformation import DataTransformation
        from Forecasting_System.components.model_trainer import ModelTrainer

        # Load preprocessor
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

        model_trainer = ModelTrainer()
        model = model_trainer.get_trained_model()

        X_test_transformed = preprocessor.transform(X_test)
        X_train_transformed = preprocessor.transform(X_train)

        y_pred = model.predict(X_test_transformed)
        y_train_pred = model.predict(X_train_transformed)

        test_metrics = evaluate_model(y_test, y_pred)
        train_metrics = evaluate_model(y_train, y_train_pred)

        logging.info(f"Train Metrics: {train_metrics}")
        logging.info(f"Test Metrics: {test_metrics}")

        # Set tracking URI
        mlflow.set_tracking_uri("https://dagshub.com/BhaveshNikam09/Retail-demand-forecasting.mlflow")
        mlflow.set_experiment("Retail Forecasting Experiment")

        with mlflow.start_run():
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(test_metrics)

            # Save model and preprocessor locally
            os.makedirs("mlruns_artifacts", exist_ok=True)
            model_path = "mlruns_artifacts/model.pkl"
            pre_path = "mlruns_artifacts/preprocessor.pkl"

            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            with open(pre_path, "wb") as f:
                pickle.dump(preprocessor, f)

            # Log manually as artifacts
            mlflow.log_artifact(model_path, artifact_path="model")
            mlflow.log_artifact(pre_path, artifact_path="preprocessor")

            logging.info("Logged model and preprocessor to MLflow.")

    except Exception as e:
        logging.error("Error during model evaluation.")
        raise CustomException(e, sys)
