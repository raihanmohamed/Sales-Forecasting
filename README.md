# 📊 Retail Sales Forecasting & Demand Intelligence

End-to-end analysis of retail sales data — from exploratory analysis and time series forecasting to anomaly detection and product demand segmentation — packaged into a 4-page interactive Streamlit app.

**Author:** Raihan Valiyakath Mohamed

---

## 🧭 Project Overview

This project analyzes historical retail order data (`train.csv`) to help a retail business:

- Understand past sales performance
- Forecast future demand
- Catch unusual sales activity early
- Plan inventory more intelligently

It was built as a **6-part analysis (Tasks 1–6)** in a Jupyter notebook, then turned into a **Streamlit app** for interactive exploration.

---

## ✅ Tasks Covered

| Task | Description |
|------|-------------|
| **1** | Data loading, cleaning, and deep exploration (missing values, duplicates, date features, seasonal tagging, weekly/monthly aggregation) |
| **2** | Time series decomposition (trend/seasonality/residual) and stationarity testing (ADF test) |
| **3** | Sales forecasting using three models — **SARIMA**, **Facebook Prophet**, and **XGBoost** — compared on MAE, RMSE, and MAPE, with the best model selected automatically |
| **4** | Category-level and region-level forecasting (Furniture, Technology, Office Supplies, West, East) |
| **5** | Anomaly detection on weekly sales using **Isolation Forest** and **rolling Z-score**, with flagged weeks and likely explanations |
| **6** | Product demand segmentation via **K-Means clustering** on sub-categories, with auto-generated segment labels and stocking recommendations |

---

## 🖥️ Streamlit App

A 4-page interactive app built on top of the Task 1–6 analysis:

1. **Sales Overview** — Total sales by year, monthly trend, region × category filters
2. **Forecast Explorer** — Prophet forecasts by Category/Region, 1–3 month horizon, with MAE/RMSE
3. **Anomaly Report** — Isolation Forest + Z-score anomalies on weekly sales, with a table of flagged weeks
4. **Product Demand Segments** — K-Means clustering of sub-categories, with auto-generated segment labels

---

## 🔑 Key Results

- **Best forecasting model:** Prophet (RMSE ≈ 14,050 vs. SARIMA ≈ 22,539 and XGBoost ≈ 18,894)
- **Sales trend:** Stationary with a clear upward trend and yearly seasonality (ADF p < 0.001)
- **Anomalies:** 16 unusual weeks flagged across both methods, with Isolation Forest catching more subtle deviations than Z-score
- **Demand segments:** 4 clusters — *High Value Products*, *Low Volume/Stable*, *High Volume/Stable*, and *Growing Demand* — each with a tailored stocking strategy

---

## 🛠️ Tech Stack

**Analysis:** Python, pandas, NumPy, statsmodels, Prophet, XGBoost, scikit-learn

**Visualization:** Matplotlib

**App:** Streamlit
