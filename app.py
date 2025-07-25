import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from src.Forecasting_System.components.data_transformation import DataTransformation
from src.Forecasting_System.utils.utils import load_object

# Load paths
MAIN_MODEL_PATH = os.path.join("artifacts", "catboost_main_model.cbm")
MODEL_10_PATH = os.path.join("artifacts", "catboost_model_10.cbm")
MODEL_90_PATH = os.path.join("artifacts", "catboost_model_90.cbm")
PREPROCESSOR_PATH = os.path.join("artifacts", "preprocessor.pkl")

# Load models and preprocessor
model = CatBoostRegressor()
model.load_model(MAIN_MODEL_PATH)

model_10 = CatBoostRegressor()
model_10.load_model(MODEL_10_PATH)

model_90 = CatBoostRegressor()
model_90.load_model(MODEL_90_PATH)

preprocessor = load_object(PREPROCESSOR_PATH)

st.title("📦 Retail Demand Forecasting System")
st.markdown("Upload your retail dataset and get forecast predictions with confidence intervals.")

uploaded_file = st.file_uploader("📁 Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Take last 20 rows to preserve rolling features
        latest_df = df.tail(20).copy()

        transformer = DataTransformation()

        # Apply transformation
        transformed_X, _ = transformer.transform_new_data(latest_df, preprocessor)

        if transformed_X.shape[0] == 0:
            st.error("❌ Not enough valid rows after transformation. Please upload more data.")
            st.stop()

        # Take last valid row for prediction
        X_input = transformed_X[-1:]  # Shape = (1, 32)

        # Predict
        y_pred = np.expm1(model.predict(X_input))[0]
        y_pred_10 = np.expm1(model_10.predict(X_input))[0]
        y_pred_90 = np.expm1(model_90.predict(X_input))[0]

        st.markdown("### 🔮 Forecast for Next Period")
        st.success(f"📈 Predicted Demand: **{y_pred:.2f}**")
        st.info(f"📦 Confidence Interval (10%-90%): **{y_pred_10:.2f} - {y_pred_90:.2f}**")

        # Plot 30-day forecast
        future_dates = pd.date_range(start=pd.to_datetime("today"), periods=30)
        forecast = [y_pred] * 30
        lower = [y_pred_10] * 30
        upper = [y_pred_90] * 30

        plt.figure(figsize=(10, 5))
        plt.plot(future_dates, forecast, label="Forecast", color="blue")
        plt.fill_between(future_dates, lower, upper, color='blue', alpha=0.2, label="Confidence Interval")
        plt.xlabel("Date")
        plt.ylabel("Demand")
        plt.title("📊 30-Day Demand Forecast")
        plt.grid(True)
        plt.legend()
        st.pyplot(plt)

    except Exception as e:
        st.error(f"❌ Prediction Failed: {str(e)}")
