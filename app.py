import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Visual Stock Dashboard", layout="wide")

st.title("📊 Visual Stock Analysis Dashboard")
st.markdown("Clean + Attractive + Easy to Understand")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    file = "stocks_data.csv"
    if os.path.exists(file):
        return pd.read_csv(file)
    return None

df = load_data()

if df is None:
    st.warning("Upload your dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
    else:
        st.stop()

# -------------------------------
# PREPROCESS
# -------------------------------
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df.set_index('Date', inplace=True)

# Sidebar
st.sidebar.header("⚙️ Settings")
stock = st.sidebar.selectbox("Select Stock", df['stock'].unique())

# Filter data
data = df[df['stock'] == stock][['Close']].copy()

# 🔥 IMPORTANT FIX
data = data.sort_index()
data.index = pd.to_datetime(data.index)

# -------------------------------
# METRICS
# -------------------------------
latest = data['Close'].iloc[-1]
prev = data['Close'].iloc[-2]
change = latest - prev

returns = data['Close'].pct_change().dropna()

col1, col2, col3 = st.columns(3)
col1.metric("Price", f"{latest:.2f}")
col2.metric("Change", f"{change:.2f}")
col3.metric("Volatility", f"{returns.std():.4f}")

# -------------------------------
# AREA CHART
# -------------------------------
st.subheader("📈 Price Trend")

fig1 = px.area(data, y='Close', title="Stock Price Trend")
fig1.update_layout(template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# ✅ FIXED RESAMPLE
# -------------------------------
st.subheader("📊 Monthly Average Price")

monthly = data.resample('ME').mean()   # 🔥 FIX: use 'ME' instead of 'M'

fig2 = px.bar(monthly, y='Close', title="Monthly Average Price", color='Close')
fig2.update_layout(template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# PIE CHART
# -------------------------------
st.subheader("🥧 Gain vs Loss")

gain = (returns > 0).sum()
loss = (returns < 0).sum()

fig3 = px.pie(values=[gain, loss], names=['Gain', 'Loss'])
fig3.update_layout(template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# FUNNEL CHART
# -------------------------------
st.subheader("🔻 Funnel Analysis")

fig4 = go.Figure(go.Funnel(
    y=["Total Days", "Gain Days", "High Gain (>2%)"],
    x=[len(returns), gain, (returns > 0.02).sum()]
))
fig4.update_layout(template="plotly_white")
st.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# HISTOGRAM
# -------------------------------
st.subheader("📉 Returns Distribution")

fig5 = px.histogram(returns, nbins=40)
fig5.update_layout(template="plotly_white")
st.plotly_chart(fig5, use_container_width=True)

# -------------------------------
# STATIONARITY
# -------------------------------
st.subheader("📌 Stationarity")

p = adfuller(data['Close'])[1]

if p < 0.05:
    st.success(f"Stationary (p={p:.4f})")
else:
    st.warning(f"Not Stationary (p={p:.4f})")

# -------------------------------
# FORECAST
# -------------------------------
st.subheader("🔮 Forecast")

model = ARIMA(data['Close'], order=(5,1,0))
fit = model.fit()

steps = st.slider("Forecast Days", 5, 20, 10)
forecast = fit.forecast(steps=steps)

future_dates = pd.date_range(start=data.index[-1], periods=steps+1, freq='B')[1:]

fig6 = go.Figure()
fig6.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Actual"))
fig6.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(dash='dash')))

fig6.update_layout(template="plotly_white")
st.plotly_chart(fig6, use_container_width=True)

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("🧠 Insights")

trend = "Uptrend 📈" if change > 0 else "Downtrend 📉"

st.write(f"Trend: {trend}")
st.write(f"Avg Return: {returns.mean():.4f}")
st.write(f"Risk: {returns.std():.4f}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Clean Dashboard | Fixed Errors ✅")
