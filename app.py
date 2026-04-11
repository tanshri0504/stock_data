import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Page config
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main {background-color: #0E1117; color: white;}
    h1, h2, h3 {color: #00ADB5;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Smart Stock Price Prediction Dashboard")
st.markdown("AI-powered insights with interactive visualization 🚀")

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    file_path = "stocks_data.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_data()

if df is None:
    st.warning("⚠️ Upload your stock dataset to continue")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

# -------------------------------
# PREPROCESSING
# -------------------------------
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Sidebar
st.sidebar.header("⚙️ Controls")
stocks = df['stock'].unique()
selected_stock = st.sidebar.selectbox("Select Stock", stocks)

st_data = df[df['stock'] == selected_stock][['Close']]

# -------------------------------
# METRICS
# -------------------------------
latest_price = st_data['Close'].iloc[-1]
prev_price = st_data['Close'].iloc[-2]
change = latest_price - prev_price

col1, col2, col3 = st.columns(3)
col1.metric("💰 Latest Price", f"{latest_price:.2f}")
col2.metric("📈 Change", f"{change:.2f}")
col3.metric("📊 % Change", f"{(change/prev_price)*100:.2f}%")

# -------------------------------
# TREND GRAPH
# -------------------------------
st.subheader("📉 Stock Trend")

fig, ax = plt.subplots(figsize=(10,5))
sns.lineplot(data=st_data, x=st_data.index, y='Close', ax=ax)
ax.set_title("Stock Price Trend", fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Price")
st.pyplot(fig)

# -------------------------------
# RETURNS
# -------------------------------
st_data['Returns'] = st_data['Close'].pct_change()
st_data.dropna(inplace=True)

# -------------------------------
# STATIONARITY
# -------------------------------
def check_stationarity(series):
    return adfuller(series.dropna())[1]

p_value = check_stationarity(st_data['Close'])

st.subheader("📌 Stationarity Analysis")
if p_value < 0.05:
    st.success(f"Stationary Data ✅ (p={p_value:.4f})")
else:
    st.warning(f"Not Stationary ❌ (p={p_value:.4f})")

# -------------------------------
# MODEL
# -------------------------------
model = ARIMA(st_data['Close'], order=(5,1,0))
model_fit = model.fit()

st.success("🤖 Model trained successfully!")

# -------------------------------
# FORECAST
# -------------------------------
steps = st.slider("Forecast Days", 1, 30, 10)
forecast = model_fit.forecast(steps=steps)

dates = pd.date_range(start=st_data.index[-1], periods=steps+1, freq='B')[1:]

# -------------------------------
# FORECAST GRAPH
# -------------------------------
st.subheader("🔮 Forecast Visualization")

fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(st_data['Close'], label='Actual', linewidth=2)
ax2.plot(dates, forecast, linestyle='dashed', label='Forecast', linewidth=2)
ax2.fill_between(dates, forecast, alpha=0.3)
ax2.set_title("Future Price Prediction")
ax2.legend()
st.pyplot(fig2)

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("🧠 Key Insights")

trend = "Upward 📈" if change > 0 else "Downward 📉"
volatility = st_data['Returns'].std()

st.write(f"""
- Current Trend: **{trend}**
- Volatility (Risk Level): **{volatility:.4f}**
- Average Return: **{st_data['Returns'].mean():.4f}**
- Prediction suggests possible future movement based on past data.
""")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.markdown("✨ Designed with Streamlit | Enhanced UI + Insights")
