import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.Forecasting_System.logger import logging
from src.Forecasting_System.exception import custom_exception

@dataclass
class DataIngestion:
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class Data_Ingestion:
    def __init__(self):
        self.ingestion_config = DataIngestion()

    def data_insert(self):
        logging.info("Data ingestion started")

        try:
            # Step 1: Load raw data
            df = pd.read_csv(os.path.join("notebooks/data", "Retail_Dataset2.csv"))
            

            # Step 6: Save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")

            return (
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise custom_exception(e, sys)
