import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from src.Forecasting_System.components.data_transformation import DataTransformation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page config
st.set_page_config(page_title="Retail Forecasting", layout="wide")

st.title("📦 Retail Demand Forecasting System")
st.markdown("Upload your retail dataset and get forecast predictions with confidence intervals.")

# Load trained models
@st.cache_resource
def load_models():
    model_main = CatBoostRegressor()
    model_main.load_model("artifacts/catboost_main_model.cbm")

    model_10 = CatBoostRegressor()
    model_10.load_model("artifacts/catboost_model_10.cbm")

    model_90 = CatBoostRegressor()
    model_90.load_model("artifacts/catboost_model_90.cbm")

    return model_main, model_10, model_90

model, model_10, model_90 = load_models()

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    uploaded_file_path = "artifacts/uploaded_data.csv"
    df.to_csv(uploaded_file_path, index=False)

    st.success("✅ File uploaded and saved!")

    # Transform input data
    transformer = DataTransformation()
    X_train, X_test, y_train, y_test = transformer.initiate_data_transformation(uploaded_file_path)

    # Combine transformed data
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.hstack([y_train, y_test])

    # Run predictions
    y_pred_log = model.predict(X_combined)
    y_pred_10_log = model_10.predict(X_combined)
    y_pred_90_log = model_90.predict(X_combined)

    y_pred = np.expm1(y_pred_log)
    y_pred_10 = np.expm1(y_pred_10_log)
    y_pred_90 = np.expm1(y_pred_90_log)
    actual_demand = np.expm1(y_combined)

    # Merge predictions into original df
    df = df.iloc[-len(y_pred):].copy()
    df["Prediction"] = y_pred
    df["Lower_Bound_10%"] = y_pred_10
    df["Upper_Bound_90%"] = y_pred_90
    df["Actual"] = actual_demand

    # Evaluation
    mae = mean_absolute_error(df["Actual"], df["Prediction"])
    rmse = mean_squared_error(df["Actual"], df["Prediction"], squared=False)
    r2 = r2_score(df["Actual"], df["Prediction"])
    interval_coverage = ((df["Actual"] >= df["Lower_Bound_10%"]) & (df["Actual"] <= df["Upper_Bound_90%"])).mean() * 100

    st.subheader("📊 Forecast Summary")
    st.metric("📉 MAE", f"{mae:.2f}")
    st.metric("📉 RMSE", f"{rmse:.2f}")
    st.metric("📈 R² Score", f"{r2:.4f}")
    st.metric("📦 Interval coverage within 10%-90%", f"{interval_coverage:.2f}%")

    # Plot
    st.subheader("📈 Forecast Chart")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Prediction"], label="Prediction", color='blue')
    ax.plot(df["Date"], df["Actual"], label="Actual", color='green', linestyle='--')
    ax.fill_between(df["Date"], df["Lower_Bound_10%"], df["Upper_Bound_90%"],
                    color='lightblue', alpha=0.4, label='Confidence Interval')
    ax.set_xlabel("Date")
    ax.set_ylabel("Demand")
    ax.legend()
    st.pyplot(fig)

    # Display DataFrame
    st.subheader("🔍 Forecast Table")
    st.dataframe(df.tail(50))

    # Download
    st.download_button("⬇️ Download Results as CSV",
                       df.to_csv(index=False).encode("utf-8"),
                       "retail_forecast_results.csv",
                       "text/csv")
