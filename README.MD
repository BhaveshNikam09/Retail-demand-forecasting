# 📦 Retail Demand Forecasting System

![Retail Forecasting Dashboard](image-1.png)

A production-ready demand forecasting system that uses **CatBoost** regression models with **quantile estimation** to predict future order demand, including **confidence intervals** (10% - 90%).

---

## 🚀 Key Features

- 📈 Predict retail demand using historical sales data
- 🎯 Confidence intervals using quantile regression (10th and 90th percentiles)
- 🧠 Automated feature engineering (lags, rolling means, SMAs, EMAs)
- 🧪 Time series-based train/test split
- 📊 Model performance metrics: R², MAE, RMSE, Interval Coverage
- 🌐 Streamlit UI for uploading CSVs and getting instant forecasts
- 🛠 Modular ML pipeline with data ingestion, transformation, training, and inference

---

## 📊 Example Forecast Output

| Metric | Value |
|--------|--------|
| **MAE** | `2046.80` |
| **RMSE** | `17983.73` |
| **R² Score** | `0.6670` |
| **Interval Coverage (10%-90%)** | `79.85%` |

---

---

## 🧪 Data & Feature Engineering

Features created from the input `Date` and `Order_Demand`:

- `lag_1`, `lag_7`
- `rolling_mean_3`, `rolling_mean_7`, `rolling_std_7`
- `sma_7`, `sma_14` (Simple Moving Averages)
- `ema_7`, `ema_14` (Exponential Moving Averages)
- Date-based features: `year`, `month`, `day`, `dayofweek`, `is_weekend`

---

## 💻 Streamlit UI

- Upload a `.csv` file
- Visualize the forecast chart
- Download the forecast with intervals

> Example UI:

![UI Screenshot](image.png)

---

## 📦 How to Run Locally

1. **Clone the Repo**
   ```bash
   git clone https://github.com/bhaveshnikam09/retail-demand-forecasting.git
   cd retail-demand-forecasting


