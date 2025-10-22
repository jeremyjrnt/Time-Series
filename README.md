# 🕒 Delhi Air Quality — Time Series Forecasting (2015–2020)

> *Technion – Israel Institute of Technology*
> *Course – Time Series
> **Authors:** Naomie Melloul · Jeremy Jornet

---

## 🌍 Project Overview

This project analyzes and forecasts the **Air Quality Index (AQI)** in **Delhi, India** from **2015–2020**.  
The study explores the temporal structure of AQI, fits forecasting models, integrates exogenous meteorological data, and detects structural changes in air quality trends.

### Objectives
1. Identify **trends, seasonality, and autocorrelation** in AQI data.  
2. Fit and compare classical and modern **forecasting models**.  
3. Incorporate **meteorological exogenous variables** to improve predictions.  
4. Apply **change-point detection** methods to spot long-term shifts.

---

## 📦 Data & Preprocessing

- **Source:** Central Pollution Control Board (CPCB) – via Kaggle  
- **Period:** 2015–2019  
- **Frequency:** Daily → aggregated **weekly** for modeling  
- **City:** Delhi  
- **Exogenous variables:**  
  `avg_temperature`, `avg_humidity`, `avg_dew_point`, `avg_wind_speed`, `avg_pressure`

Missing values were imputed using **previous-day interpolation** and **Gaussian noise (5%)** to maintain realistic variability.

---

## 📊 Exploratory Data Analysis (EDA)

The AQI series exhibits **strong seasonality**, **short-term volatility**, and **moderate autocorrelation**.

### Key insights
- **Trend:** Flat from 2015–2017 → downward shift in early 2017 → stabilization post-2018.  
- **Seasonality:** Pronounced yearly cycle — peaks in **winter (Nov–Jan)**, lows in **monsoon (Jun–Sep)**.  
- **Weekly pattern:** AQI marginally higher on weekdays vs weekends.  
- **ACF/PACF:** Seasonal correlation at lag ≈ **52 weeks** confirms annual periodicity.

**Main plots included:**  
- Seasonal decomposition  
- Month–year heatmap  
- Boxplots by weekday  
- Rolling statistics (30, 100, 300 days)  
- ACF/PACF up to 400 lags  

---

## ⚙️ Models Implemented

### 1️⃣ SARIMA — Seasonal ARIMA

We modeled weekly AQI using **SARIMA(p,d,q) × (P,D,Q, 52)**.  
Two scenarios were evaluated:

| Scenario | Model | BIC | RMSE | MAE |
|-----------|--------|------|-------|------|
| 1 | (2,1,1) × (1,1,2)[52] | 567.8 | 103.1 | 86.3 |
| 2 | **(0,1,1) × (0,1,2)[52]** | **561.8** | **74.1** | **57.9** |

✅ **Selected model:** `(0,1,1) × (0,1,2)[52]` — simplest, best-performing, stable residuals.

**Residual diagnostics**
- Mean ≈ 0  
- No significant autocorrelation  
- Residuals ≈ Gaussian  

> 📘 **References**  
> Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (2015). *Time Series Analysis: Forecasting and Control.*  
> Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice.*

---

### 2️⃣ Prophet (Meta/Facebook)

Facebook’s **Prophet** was applied for its scalability and auto-seasonality detection.

- Configuration: default (`yearly_seasonality=True`)  
- Train/Test split: 80% / 20%  
- Performance: **RMSE ≈ 90**, **MAE ≈ 70**

Prophet captured overall seasonality and trends, though it slightly **overestimated high peaks**.

> Taylor, S.J., & Letham, B. (2018). *Forecasting at Scale.* *The American Statistician*, 72(1).

---

### 3️⃣ Neural Network (MLP)

A **multi-layer perceptron** was trained for nonlinear pattern capture.

**Features**
- Cyclic encodings (`sin`, `cos` of month/day)
- Normalization
- Global index (time position)

**Architecture**
Input → Dense(256) → Dense(128) → Dense(128) → Dense(64) → Dense(32) → Output
- Activation: ReLU  
- Optimizer: Adam (lr = 1e-3)  
- Loss: MSE  
- Regularization: Dropout  

🧠 Captured long-term trend and annual pattern but smoothed short-term spikes.

> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning.* MIT Press.

---

## 🌡️ Incorporating Exogenous Variables (SARIMAX)

We added meteorological variables as regressors to the SARIMA model.  
**Dew point** was the most correlated feature, supported by literature (Xu & Zhu, 2023).

Model used: **SARIMAX(0,1,1) × (0,1,2,52)** with dew point.

| Metric | SARIMA | **SARIMAX (dew point)** |
|:------:|:------:|:-----------------------:|
| MAE | 57.86 | **49.12** |
| MSE | 5496.1 | **4580.4** |

✅ **Improvement:** Better short-term prediction, reduced overestimation.

> Xu, Y., & Zhu, X. (2023). *Recognizing Dew as an Indicator and an Improver of Near-Surface Air Quality.*

---

## 🔍 Change-Point Detection

Two methods were applied:

### 🧭 Shewhart Control Chart
- Applied on raw and decomposed series (±3σ bounds)
- Detected recurring high-AQI spikes (e.g., Nov–Dec pollution)
- Mainly flagged **seasonal outliers**, not structural changes.

### ✂️ Segmentation (STL-based)
- Applied on **trend** and **residuals** components
- Found a **major change-point around early 2017**, matching the visual drop.

> Shewhart, W.A. (1931). *Economic Control of Quality of Manufactured Product.*  
> Killick, R. et al. (2012). *Optimal Detection of Changepoints with a Linear Computational Cost.*

---

## 📈 Results Summary

| Model | RMSE | MAE | Key Notes |
|-------|:----:|:---:|------------|
| SARIMA | 74.1 | 57.9 | Robust, interpretable baseline |
| Prophet | 90.0 | 70.0 | Fast, good trend capture |
| Neural Net | 85.0 | 68.0 | Nonlinear, smooth output |
| **SARIMAX** | **67.0** | **49.1** | **Best overall accuracy** |

---

## 🧰 Tools & Libraries

`python`, `pandas`, `numpy`, `matplotlib`, `seaborn`,  
`statsmodels` (SARIMA/SARIMAX),  
`prophet`, `scikit-learn`,  
`tensorflow/keras`, `scipy`, `stl`/`seasonal_decompose`

---

## 📁 Repo Structure

.
├── README.md                     # Project documentation and overview
├── delhi-air-quality.ipynb       # Main Jupyter notebook with full analysis and modeling
└── delhi-air-quality.pdf         # Final project report (Technion submission)

---

## 💡 Insights & Conclusions

- AQI shows **strong annual seasonality** and moderate **weekday effect**.  
- **SARIMA** and **SARIMAX** captured trends effectively, outperforming Prophet and NN in RMSE/MAE.  
- Adding **dew point** as an exogenous feature improved predictions.  
- A **structural shift** around 2017 indicates potential environmental or policy-driven change.

---

## 📚 References

- Box, G.E.P., Jenkins, G.M., Reinsel, G.C. (2015). *Time Series Analysis: Forecasting and Control.*  
- Hyndman, R.J., Athanasopoulos, G. (2021). *Forecasting: Principles and Practice.*  
- Taylor, S.J., Letham, B. (2018). *Forecasting at Scale.* *The American Statistician.*  
- Xu, Y., Zhu, X. (2023). *Recognizing Dew as an Indicator and an Improver of Near-Surface Air Quality.*  
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning.* MIT Press.
