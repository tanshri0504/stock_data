# -------------------------------
# IMPORTS (MUST BE FIRST)
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Visual Stock Dashboard", layout="wide")

st.title("📊 Visual Stock Analysis Dashboard")

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

stock = st.selectbox("Select Stock", df['stock'].unique())

data = df[df['stock'] == stock][['Close']].copy()
data = data.sort_index()

returns = data['Close'].pct_change().dropna()

# -------------------------------
# ✅ SUBPLOTS
# -------------------------------
st.subheader("📊 Combined Dashboard (Subplots)")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Price", "Monthly", "Gain/Loss", "Returns"),
    specs=[[{"type": "scatter"}, {"type": "bar"}],
           [{"type": "domain"}, {"type": "histogram"}]]
)

# Price
fig.add_trace(
    go.Scatter(x=data.index, y=data['Close'], fill='tozeroy', name="Price"),
    row=1, col=1
)

# Monthly
monthly = data.resample('ME').mean()
fig.add_trace(
    go.Bar(x=monthly.index, y=monthly['Close'], name="Monthly"),
    row=1, col=2
)

# Pie
gain = (returns > 0).sum()
loss = (returns < 0).sum()
fig.add_trace(
    go.Pie(labels=["Gain", "Loss"], values=[gain, loss]),
    row=2, col=1
)

# Histogram
fig.add_trace(
    go.Histogram(x=returns, nbinsx=40),
    row=2, col=2
)

fig.update_layout(height=700, template="plotly_white")

st.plotly_chart(fig, use_container_width=True)
