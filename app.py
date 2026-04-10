import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

st.title("📈 Stock Price Prediction App")
st.markdown("Analyze stock trends and forecast future prices using ARIMA model.")

# -------------------------------
# LOAD DATA (SAFE METHOD)
# -------------------------------
@st.cache_data
def load_data():
    file_path = "stocks_data.csv"
    
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        return None

df = load_data()

# -------------------------------
# FILE UPLOAD OPTION (IMPORTANT)
# -------------------------------
if df is None:
    st.warning("⚠️ File not found! Please upload your dataset.")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# -------------------------------
# DATA PREPROCESSING
# -------------------------------
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Sidebar
st.sidebar.header("⚙️ Settings")

stocks = df['stock'].unique()
selected_stock = st.sidebar.selectbox("Select Stock", stocks)

# Filter data
st_data = df[df['stock'] == selected_stock][['Close']]

# -------------------------------
# SHOW DATA
# -------------------------------
st.subheader(f"📊 Data for {selected_stock}")
st.dataframe(st_data.tail())

# -------------------------------
# PLOT ACTUAL PRICES
# -------------------------------
st.subheader("📉 Stock Price Trend")
st.line_chart(st_data['Close'])

# -------------------------------
# RETURNS
# -------------------------------
st_data['Returns'] = st_data['Close'].pct_change()
st_data.dropna(inplace=True)

# -------------------------------
# STATIONARITY CHECK
# -------------------------------
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    return result[1]

st.subheader("📌 Stationarity Check")

p_value = check_stationarity(st_data['Close'])
st.write(f"P-value: {p_value}")

if p_value < 0.05:
    st.success("Data is Stationary ✅")
else:
    st.warning("Data is NOT Stationary ❌")

# Differencing
st_data['Close_Diff'] = st_data['Close'].diff()
p_value_diff = check_stationarity(st_data['Close_Diff'].dropna())

st.write(f"Differenced P-value: {p_value_diff}")

# -------------------------------
# MODEL TRAINING
# -------------------------------
st.subheader("🤖 Model Training")

model = ARIMA(st_data['Close'], order=(5,1,0))
model_fit = model.fit()

st.success("Model Trained Successfully!")

# -------------------------------
# FORECAST
# -------------------------------
forecast_steps = st.slider("Select Forecast Days", 1, 30, 10)

forecast = model_fit.forecast(steps=forecast_steps)

dates = pd.date_range(start=st_data.index[-1], periods=forecast_steps+1, freq='B')[1:]

# -------------------------------
# PLOT FORECAST
# -------------------------------
st.subheader("🔮 Forecasted Prices")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(st_data['Close'], label='Actual Prices')
ax.plot(dates, forecast, label='Predicted Prices', linestyle='dashed')
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()

st.pyplot(fig)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("✨ Built with Streamlit + ARIMA")
