st.subheader("📊 Combined Dashboard (Subplots)")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Price Trend",
        "Monthly Avg",
        "Gain vs Loss",
        "Returns Distribution"
    ),
    specs=[
        [{"type": "scatter"}, {"type": "bar"}],
        [{"type": "domain"}, {"type": "histogram"}]
    ]
)

# -------------------------------
# 1. Price Trend (Area)
# -------------------------------
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data['Close'],
        fill='tozeroy',
        name="Price"
    ),
    row=1, col=1
)

# -------------------------------
# 2. Monthly Avg (Bar)
# -------------------------------
monthly = data.resample('ME').mean()

fig.add_trace(
    go.Bar(
        x=monthly.index,
        y=monthly['Close'],
        name="Monthly Avg"
    ),
    row=1, col=2
)

# -------------------------------
# 3. Pie Chart (Gain vs Loss)
# -------------------------------
gain = (returns > 0).sum()
loss = (returns < 0).sum()

fig.add_trace(
    go.Pie(
        labels=["Gain", "Loss"],
        values=[gain, loss],
        name="Gain/Loss"
    ),
    row=2, col=1
)

# -------------------------------
# 4. Histogram (Returns)
# -------------------------------
fig.add_trace(
    go.Histogram(
        x=returns,
        nbinsx=40,
        name="Returns"
    ),
    row=2, col=2
)

# Layout
fig.update_layout(
    height=700,
    showlegend=False,
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
