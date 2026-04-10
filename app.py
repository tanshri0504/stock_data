import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Title
st.title("📈 Stock Price Prediction App")
st.markdown("Analyze stock trends and forecast future prices using ARIMA model.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('stocks_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

df = load_data()

# Sidebar
st.sidebar.header("⚙️ Settings")

stocks = df['stock'].unique()
selected_stock = st.sidebar.selectbox("Select Stock", stocks)

# Filter data
st_data = df[df['stock'] == selected_stock][['Close']]

# Show data
st.subheader(f"📊 Data for {selected_stock}")
st.dataframe(st_data.tail())

# Plot actual prices
st.subheader("📉 Stock Price Trend")
st.line_chart(st_data['Close'])

# Returns
st_data['Returns'] = st_data['Close'].pct_change()
st_data.dropna(inplace=True)

# Stationarity check
def check_stationarity(timeseries):
    result = adfuller(timeseries.dropna())
    return result[1]

st.subheader("📌 Stationarity Check")

p_value = check_stationarity(st_data['Close'])
st.write(f"Close Price P-value: {p_value}")

if p_value < 0.05:
    st.success("Data is Stationary ✅")
else:
    st.warning("Data is NOT Stationary ❌")

# Differencing
st_data['Close_Diff'] = st_data['Close'].diff()
p_value_diff = check_stationarity(st_data['Close_Diff'].dropna())

st.write(f"Differenced P-value: {p_value_diff}")

# Model training
st.subheader("🤖 Model Training")

model = ARIMA(st_data['Close'], order=(5,1,0))
model_fit = model.fit()

st.success("Model Trained Successfully!")

# Forecast
forecast_steps = st.slider("Select Forecast Days", 1, 30, 10)

forecast = model_fit.forecast(steps=forecast_steps)

dates = pd.date_range(start=st_data.index[-1], periods=forecast_steps+1, freq='B')[1:]

# Plot forecast
st.subheader("🔮 Forecasted Prices")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(st_data['Close'], label='Actual Prices')
ax.plot(dates, forecast, label='Predicted Prices', linestyle='dashed')
ax.set_xlabel("Date")
ax.set_ylabel("Stock Price")
ax.legend()

st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("✨ Built with Streamlit | ARIMA Model")
