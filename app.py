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
st.set_page_config(page_title="Stock Dashboard", layout="wide")

# -------------------------------
# TITLE
# -------------------------------
st.title("📊 Stock Market Analysis Dashboard")
st.markdown("Clean, interactive and professional stock insights")

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
    st.warning("⚠️ Please upload your dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
    else:
        st.stop()

# -------------------------------
# PREPROCESSING
# -------------------------------
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Sidebar
st.sidebar.header("⚙️ Settings")
stock = st.sidebar.selectbox("Select Stock", df['stock'].unique())

data = df[df['stock'] == stock][['Close']]

# -------------------------------
# KPIs
# -------------------------------
latest = data['Close'].iloc[-1]
prev = data['Close'].iloc[-2]
change = latest - prev
percent = (change / prev) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Price", f"{latest:.2f}")
col2.metric("Change", f"{change:.2f}")
col3.metric("Percent Change", f"{percent:.2f}%")
col4.metric("Volatility", f"{data['Close'].pct_change().std():.4f}")

# -------------------------------
# LINE CHART
# -------------------------------
st.subheader("📈 Stock Price Trend")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    name="Price",
    line=dict(color='blue', width=2)
))

# Moving averages
data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()

fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name="MA20", line=dict(color='orange')))
fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name="MA50", line=dict(color='green')))

fig.update_layout(
    template="plotly_white",
    xaxis_title="Date",
    yaxis_title="Price",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# RETURNS DISTRIBUTION
# -------------------------------
st.subheader("📊 Returns Distribution")

returns = data['Close'].pct_change().dropna()

fig2 = px.histogram(
    returns,
    nbins=40,
    title="Returns Histogram",
    color_discrete_sequence=['skyblue']
)

fig2.update_layout(template="plotly_white")

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# STATIONARITY CHECK
# -------------------------------
st.subheader("📌 Stationarity Test")

p_value = adfuller(data['Close'].dropna())[1]

if p_value < 0.05:
    st.success(f"Data is Stationary (p={p_value:.4f})")
else:
    st.warning(f"Data is NOT Stationary (p={p_value:.4f})")

# -------------------------------
# MODEL
# -------------------------------
st.subheader("🤖 Forecast (ARIMA)")

model = ARIMA(data['Close'], order=(5,1,0))
model_fit = model.fit()

steps = st.slider("Forecast Days", 5, 30, 10)
forecast = model_fit.forecast(steps=steps)

future_dates = pd.date_range(
    start=data.index[-1],
    periods=steps+1,
    freq='B'
)[1:]

# -------------------------------
# FORECAST CHART
# -------------------------------
fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    name="Actual",
    line=dict(color='blue')
))

fig3.add_trace(go.Scatter(
    x=future_dates,
    y=forecast,
    name="Forecast",
    line=dict(color='red', dash='dash')
))

fig3.update_layout(
    template="plotly_white",
    xaxis_title="Date",
    yaxis_title="Price",
    height=500
)

st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# KEY INSIGHTS
# -------------------------------
st.subheader("🧠 Key Insights")

trend = "Uptrend 📈" if change > 0 else "Downtrend 📉"
avg_return = returns.mean()
risk = returns.std()

st.write(f"- Trend: {trend}")
st.write(f"- Average Return: {avg_return:.4f}")
st.write(f"- Risk (Volatility): {risk:.4f}")

# -------------------------------
# BUY/SELL SIGNAL
# -------------------------------
st.subheader("📢 Trading Signal")

if latest > data['MA20'].iloc[-1]:
    st.success("BUY Signal (Price above MA20)")
else:
    st.error("SELL Signal (Price below MA20)")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Built with Streamlit + Plotly + ARIMA")
