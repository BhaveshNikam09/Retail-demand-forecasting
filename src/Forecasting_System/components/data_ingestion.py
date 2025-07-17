import pandas as pd 
import numpy as np
from src.Forecasting_System.logger import logging
from src.Forecasting_System.exception import custom_exception

import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path


class DataIngestion:
    # joining the path to save csv data files
    raw_data_path: str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
@dataclass

class Data_Ingestion:
    def __init__(self):
        # creating object of the dataingestionconfig class
        self.ingestion_config = DataIngestion()
    
    def data_inset(self):
        logging.info("Data ingestion started")
        
        try:
            # reading the data from csv file
            data = pd.read_csv(os.path.join("notebooks/data", "Retail_Dataset2.csv"))
            logging.info("Csv file read as df")
            
            # making the dir to save the csv files
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)), exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Raw data saved")
            
            #splitting the data
            logging.info("Splitting the data")
            Train_data, Test_data = train_test_split(data, test_size=0.30)
            logging.info("Data splitting completed")
            
            #saving data to the csv files
            Train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            Test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data ingestion part completed")
            # returning the path of the datas both training and the testing
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )       
        except Exception as e:
            raise custom_exception(e, sys)            
    
