# 📈 Linear Regression Explorer

An interactive Streamlit app to visualize how linear regression works — from raw data to gradient descent convergence.

## Features

The app is organized into 6 tabs:

| Tab | Description |
|-----|-------------|
| 📊 Data & Line Fit | Manually adjust slope & intercept and compare against the OLS best fit |
| 📉 Error & Residuals | Visualize residuals and per-point squared errors with MSE/RMSE/MAE metrics |
| 🗺️ Loss Landscape | Explore the MSE loss surface as a 3D plot or contour map |
| 🚀 Gradient Descent | Watch gradient descent animate across the loss surface in real time |
| ⚡ Learning Rate | Compare how different learning rates affect convergence speed |
| 🌪️ Noise & Outliers | See how noise and outliers impact the regression line |

## Setup

```bash
git clone https://github.com/<your-username>/streamlit-demo.git
cd streamlit-demo

python -m venv env
source env/bin/activate

pip install streamlit plotly numpy
```

## Run

```bash
streamlit run app.py
```

## Dependencies

- `streamlit`
- `plotly`
- `numpy`

## Reference

See [`streamlit_functions.md`](./streamlit_functions.md) for a full reference of Streamlit functions used and available.
