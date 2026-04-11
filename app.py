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
st.set_page_config(page_title="🌈 Smart Stock Dashboard", layout="wide")

# -------------------------------
# CUSTOM CSS (COLORFUL UI)
# -------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #00F5A0;
}
.card {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🚀 Smart Stock Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown("### 📊 AI Insights • 📈 Trends • 🔮 Forecast")

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
    st.warning("⚠️ Upload your dataset")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
    else:
        st.stop()

# -------------------------------
# PREPROCESS
# -------------------------------
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Sidebar
st.sidebar.header("⚙️ Controls")
stock = st.sidebar.selectbox("Select Stock", df['stock'].unique())

data = df[df['stock'] == stock][['Close']]

# -------------------------------
# KPIs (COLORFUL)
# -------------------------------
latest = data['Close'].iloc[-1]
prev = data['Close'].iloc[-2]
change = latest - prev
percent = (change / prev) * 100

col1, col2, col3, col4 = st.columns(4)

col1.markdown(f"<div class='card'>💰 <b>Price</b><br>{latest:.2f}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'>📈 <b>Change</b><br>{change:.2f}</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'>📊 <b>% Change</b><br>{percent:.2f}%</div>", unsafe_allow_html=True)
col4.markdown(f"<div class='card'>📉 <b>Volatility</b><br>{data['Close'].pct_change().std():.4f}</div>", unsafe_allow_html=True)

# -------------------------------
# MULTICOLOR LINE CHART
# -------------------------------
st.subheader("🌈 Multicolor Stock Trend")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name='Price',
    line=dict(color='cyan', width=3)
))

# Moving averages
data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()

fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name='MA20', line=dict(color='yellow')))
fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name='MA50', line=dict(color='magenta')))

fig.update_layout(
    template="plotly_dark",
    title="Stock Price with Moving Averages",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# RETURNS HEATMAP STYLE
# -------------------------------
st.subheader("🔥 Returns Distribution")

returns = data['Close'].pct_change().dropna()

fig2 = px.histogram(
    returns,
    nbins=50,
    color_discrete_sequence=['#00F5A0']
)
fig2.update_layout(template="plotly_dark")

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# STATIONARITY
# -------------------------------
st.subheader("📌 Stationarity Test")

p_value = adfuller(data['Close'].dropna())[1]

if p_value < 0.05:
    st.success(f"Stationary ✅ (p={p_value:.4f})")
else:
    st.warning(f"Not Stationary ❌ (p={p_value:.4f})")

# -------------------------------
# ARIMA MODEL
# -------------------------------
st.subheader("🤖 Forecast Engine")

model = ARIMA(data['Close'], order=(5,1,0))
model_fit = model.fit()

steps = st.slider("Forecast Days", 5, 30, 10)
forecast = model_fit.forecast(steps=steps)

future_dates = pd.date_range(start=data.index[-1], periods=steps+1, freq='B')[1:]

# -------------------------------
# COLORFUL FORECAST GRAPH
# -------------------------------
st.subheader("🔮 Future Prediction")

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    name="Actual",
    line=dict(color='cyan')
))

fig3.add_trace(go.Scatter(
    x=future_dates,
    y=forecast,
    name="Forecast",
    line=dict(color='red', dash='dash')
))

fig3.update_layout(template="plotly_dark", height=500)

st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# KEY INSIGHTS PANEL
# -------------------------------
st.subheader("🧠 AI Insights")

trend = "📈 Uptrend" if change > 0 else "📉 Downtrend"
risk = returns.std()
avg_return = returns.mean()

col1, col2, col3 = st.columns(3)

col1.success(f"Trend: {trend}")
col2.info(f"Average Return: {avg_return:.4f}")
col3.warning(f"Risk Level: {risk:.4f}")

# -------------------------------
# BUY/SELL SIGNAL
# -------------------------------
st.subheader("📢 Trading Signal")

if latest > data['MA20'].iloc[-1]:
    st.success("🟢 BUY Signal (Above MA20)")
else:
    st.error("🔴 SELL Signal (Below MA20)")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("🌈 Made with ❤️ | Streamlit + Plotly + ARIMA")
