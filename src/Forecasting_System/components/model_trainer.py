import os
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from src.Forecasting_System.logger import logging
from src.Forecasting_System.exception import custom_exception
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    main_model_path: str = os.path.join("artifacts", "catboost_main_model.cbm")
    model_10_path: str = os.path.join("artifacts", "catboost_model_10.cbm")
    model_90_path: str = os.path.join("artifacts", "catboost_model_90.cbm")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def sanitize_data(self, X, y):
        """Ensure input data is dense, numeric, and cleaned of inf/nan."""
        try:
        # Convert sparse to dense if needed
            if hasattr(X, "toarray"):
                X = X.toarray()

            # Convert to DataFrame to use .isnull()
            df = pd.DataFrame(X)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Ensure y is a numpy array and remove non-finite values
            y = np.array(y)
            valid_idx = ~df.isnull().any(axis=1) & np.isfinite(y)

            if valid_idx.sum() == 0:
                raise ValueError("No valid rows remain after sanitizing.")

            return df.loc[valid_idx].values, y[valid_idx]
    
        except Exception as e:
            raise custom_exception(e, sys)

    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            logging.info("Starting data sanitization")
            X_train, y_train = self.sanitize_data(X_train, y_train)
            X_test, y_test = self.sanitize_data(X_test, y_test)

            logging.info("Training CatBoost quantile models")
            model_90 = CatBoostRegressor(loss_function='Quantile:alpha=0.9',
                                         iterations=1000, depth=6,
                                         learning_rate=0.1, verbose=100)
            model_10 = CatBoostRegressor(loss_function='Quantile:alpha=0.1',
                                         iterations=1000, depth=6,
                                         learning_rate=0.1, verbose=100)
            model_main = CatBoostRegressor(loss_function='RMSE',
                                           iterations=1000, depth=6,
                                           learning_rate=0.1, verbose=100)

            model_main.fit(X_train, y_train)
            model_90.fit(X_train, y_train)
            model_10.fit(X_train, y_train)
            
            

            y_pred_log = model_main.predict(X_test)
            y_pred_90_log = model_90.predict(X_test)
            y_pred_10_log = model_10.predict(X_test)

            # Inverse-transform
            y_pred = np.expm1(y_pred_log)
            y_pred_90 = np.expm1(y_pred_90_log)
            y_pred_10 = np.expm1(y_pred_10_log)
            y_true = np.expm1(y_test)

            r2 = r2_score(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mae = mean_absolute_error(y_true, y_pred)

            interval_covered = ((y_true >= y_pred_10) & (y_true <= y_pred_90)).mean()


            print(f"\n📊 MAE: {mae:.2f}")
            print(f"📉 RMSE: {rmse:.2f}")
            print(f"📈 R² Score: {r2:.4f}")
            print(f"📦 Interval coverage within 10%-90%: {interval_covered*100:.2f}%")

            # Save models
            model_main.save_model(self.model_trainer_config.main_model_path)
            model_10.save_model(self.model_trainer_config.model_10_path)
            model_90.save_model(self.model_trainer_config.model_90_path)

            logging.info("All CatBoost models saved successfully.")

        except Exception as e:
            raise custom_exception(e, sys)
