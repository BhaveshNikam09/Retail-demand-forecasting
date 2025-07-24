from src.Forecasting_System.components.data_ingestion import Data_Ingestion
from src.Forecasting_System.components.data_transformation import DataTransformation
from src.Forecasting_System.components.model_trainer import ModelTrainer

from src.Forecasting_System.logger import logging
from src.Forecasting_System.exception import custom_exception


data_inject=Data_Ingestion()
raw_data_path = data_inject.data_insert()

data_transform_obj = DataTransformation()
X_train, X_test, y_train, y_test= data_transform_obj.initiate_data_transformation(raw_data_path)

model_trainer_obj = ModelTrainer()
model_trainer_obj.initiate_model_training(X_train, X_test, y_train, y_test)