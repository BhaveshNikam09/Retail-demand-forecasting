import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.Forecasting_System.logger import logging
from src.Forecasting_System.exception import custom_exception

from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise custom_exception(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,model):
    try:
            # Train model
        model.fit(X_train,y_train)

            

            # Predict Testing data
        y_test_pred =model.predict(X_test)
        y_pred=np.expm1(y_test_pred)
        y_test=np.expm1(y_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
        test_model_score = r2_score(y_test,y_pred)


        return  f'The r2_score is : {test_model_score}'
 
    except Exception as e:
        logging.info('Exception occured during model training')
        raise custom_exception(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise custom_exception(e,sys)