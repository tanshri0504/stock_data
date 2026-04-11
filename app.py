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
st.markdown("Easy to understand insights with multiple visualizations")

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
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Sidebar
st.sidebar.header("⚙️ Settings")
stock = st.sidebar.selectbox("Select Stock", df['stock'].unique())

data = df[df['stock'] == stock][['Close']]

# -------------------------------
# BASIC METRICS
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
# LINE + AREA CHART
# -------------------------------
st.subheader("📈 Price Trend")

fig1 = px.area(data, y='Close', title="Stock Price Trend (Area View)")
fig1.update_layout(template="plotly_white")
st.plotly_chart(fig1, use_container_width=True)

# -------------------------------
# BAR CHART (MONTHLY AVG)
# -------------------------------
st.subheader("📊 Monthly Average Price")

monthly = data.resample('M').mean()

fig2 = px.bar(monthly, y='Close', title="Monthly Average Price", color='Close')
fig2.update_layout(template="plotly_white")
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# PIE CHART (GAIN vs LOSS DAYS)
# -------------------------------
st.subheader("🥧 Gain vs Loss Days")

gain_days = (returns > 0).sum()
loss_days = (returns < 0).sum()

fig3 = px.pie(
    values=[gain_days, loss_days],
    names=['Gain Days', 'Loss Days'],
    title="Market Movement Distribution"
)
fig3.update_layout(template="plotly_white")
st.plotly_chart(fig3, use_container_width=True)

# -------------------------------
# FUNNEL CHART (SIMPLIFIED FLOW)
# -------------------------------
st.subheader("🔻 Stock Movement Funnel")

stages = ["Total Days", "Positive Days", "High Gain Days"]
values = [
    len(returns),
    gain_days,
    (returns > 0.02).sum()
]

fig4 = go.Figure(go.Funnel(
    y=stages,
    x=values
))
fig4.update_layout(template="plotly_white")
st.plotly_chart(fig4, use_container_width=True)

# -------------------------------
# HISTOGRAM (RETURNS)
# -------------------------------
st.subheader("📉 Returns Distribution")

fig5 = px.histogram(returns, nbins=40, title="Returns Histogram")
fig5.update_layout(template="plotly_white")
st.plotly_chart(fig5, use_container_width=True)

# -------------------------------
# STATIONARITY
# -------------------------------
st.subheader("📌 Stationarity")

p_value = adfuller(data['Close'].dropna())[1]

if p_value < 0.05:
    st.success(f"Stationary Data (p={p_value:.4f})")
else:
    st.warning(f"Not Stationary (p={p_value:.4f})")

# -------------------------------
# FORECAST
# -------------------------------
st.subheader("🔮 Forecast")

model = ARIMA(data['Close'], order=(5,1,0))
model_fit = model.fit()

steps = st.slider("Forecast Days", 5, 20, 10)
forecast = model_fit.forecast(steps=steps)

future_dates = pd.date_range(start=data.index[-1], periods=steps+1, freq='B')[1:]

fig6 = go.Figure()

fig6.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Actual"))
fig6.add_trace(go.Scatter(x=future_dates, y=forecast, name="Forecast", line=dict(dash='dash')))

fig6.update_layout(template="plotly_white", title="Forecast vs Actual")
st.plotly_chart(fig6, use_container_width=True)

# -------------------------------
# SIMPLE INSIGHTS
# -------------------------------
st.subheader("🧠 Key Insights")

trend = "Uptrend 📈" if change > 0 else "Downtrend 📉"

st.write(f"- Trend: {trend}")
st.write(f"- Avg Return: {returns.mean():.4f}")
st.write(f"- Risk Level: {returns.std():.4f}")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("Simple Visual Dashboard | Streamlit + Plotly")
