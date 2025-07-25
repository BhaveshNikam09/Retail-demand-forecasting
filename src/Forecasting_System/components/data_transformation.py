import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.Forecasting_System.logger import logging
from src.Forecasting_System.exception import custom_exception
from src.Forecasting_System.utils.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def feature_engineering(self, df):
        try:
            logging.info("Starting feature engineering...")

            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')

            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            df['day'] = df['Date'].dt.day
            df['dayofweek'] = df['Date'].dt.dayofweek
            df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

            df['lag_1'] = df['Order_Demand'].shift(1)
            df['rolling_mean_3'] = df['Order_Demand'].shift(1).rolling(3).mean()
            df['lag_7'] = df['Order_Demand'].shift(7)
            df['rolling_mean_7'] = df['Order_Demand'].shift(1).rolling(7).mean()
            df['rolling_std_7'] = df['Order_Demand'].shift(1).rolling(7).std()
            df['sma_7'] = df['Order_Demand'].rolling(7).mean()
            df['sma_14'] = df['Order_Demand'].rolling(14).mean()
            df['ema_7'] = df['Order_Demand'].ewm(span=7, adjust=False).mean()
            df['ema_14'] = df['Order_Demand'].ewm(span=14, adjust=False).mean()

            df.dropna(inplace=True)
            df.drop(columns=['Date'], inplace=True)

            logging.info("Feature engineering completed.")
            return df

        except Exception as e:
            logging.error("Error in feature engineering.")
            raise custom_exception(e, sys)

    def get_preprocessor(self, numeric_features, categorical_features):
        try:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", drop='first', sparse_output=False))

            ])

            preprocessor = ColumnTransformer([
                ("num", num_pipeline, numeric_features),
                ("cat", cat_pipeline, categorical_features)
            ])

            return preprocessor

        except Exception as e:
            logging.error("Error in creating preprocessor.")
            raise custom_exception(e, sys)

    def initiate_data_transformation(self, raw_path: str):
        try:
            df = pd.read_csv(raw_path)
            logging.info("Raw data loaded.")

            df = self.feature_engineering(df)

            X = df.drop(columns=['Order_Demand'])
            y = np.log1p(df['Order_Demand'])

            categorical_cols = ['Warehouse', 'Product_Category', 'StateHoliday', 'Product_Code']
            numerical_cols = X.drop(columns=categorical_cols).columns.tolist()

            preprocessor = self.get_preprocessor(numerical_cols, categorical_cols)

            # Time-based split (80% train, 20% test)
            split_index = int(len(X) * 0.8)
            X_train_raw, X_test_raw = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

            # Fit only on training set
            X_train = preprocessor.fit_transform(X_train_raw)
            X_test = preprocessor.transform(X_test_raw)

            # Save preprocessor
            save_object(self.config.preprocessor_obj_file_path, preprocessor)

            logging.info("Data transformation completed with TimeSeriesSplit.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error during data transformation pipeline.")
            raise custom_exception(e, sys)


    def transform_new_data(self, df, preprocessor):
        try:
            # Feature Engineering
            df = self.feature_engineering(df)
            X = df.drop(columns=['Order_Demand'], errors='ignore')

            # Transform using loaded preprocessor
            X_transformed = preprocessor.transform(X)
            return X_transformed, X

        except Exception as e:
            raise custom_exception(e, sys)

        