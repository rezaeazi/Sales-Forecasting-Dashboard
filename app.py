import streamlit as st
import pandas as pd
from src.preprocessing import load_sales_data, preprocess_sales_data
from src.modeling import train_test_split_ts, create_prophet_model, make_forecast
from src.evaluation import mae, mape
import plotly.graph_objects as go

# 1. Load and preprocess data

@st.cache_data
def load_data():
    df_raw = load_sales_data("data/sales.csv")
    df_clean = preprocess_sales_data(df_raw)
    return df_clean

df = load_data()

# 2. Sidebar input

st.sidebar.header("Settings")

forecast_days = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=30,
    max_value=180,
    value=90,
    step=10
)

start_date = st.sidebar.date_input("Start date", df.index.min())
end_date = st.sidebar.date_input("End date", df.index.max())

# Filter data by selected dates
df_filtered = df.loc[start_date:end_date]

# 3. Train / Test split

train, test = train_test_split_ts(df_filtered, test_days=forecast_days)

# 4. Fit Prophet model

model = create_prophet_model(train)
forecast = make_forecast(model, periods=forecast_days)

# 5. Evaluation (MAE / MAPE)

# Align forecast with test dates
forecast_test = forecast.set_index("ds").loc[test.index]

y_true = test["sales"].values
y_pred = forecast_test["yhat"].values

mae_value = mae(y_true, y_pred)
mape_value = mape(y_true, y_pred)

# 6. Metrics display

st.subheader("Model Evaluation")

col1, col2 = st.columns(2)
col1.metric("MAE", f"{mae_value:.2f}")
col2.metric("MAPE (%)", f"{mape_value:.2f}")

st.info(
    "This model accounts for promotinal events and hoidays, "
    "allowing more realistic sales forecasts during abnormal periods."
)

# 7. Plotly interactive plot

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=train.index,
    y=train["sales"],
    mode="lines",
    name="Train"
))

fig.add_trace(go.Scatter(
    x=test.index,
    y=test["sales"],
    mode="lines",
    name="Test"
))

fig.add_trace(go.Scatter(
    x=forecast["ds"],
    y=forecast["yhat"],
    mode="lines",
    name="Forecast"
))

fig.update_layout(
    title="Sales Forecast Dashboard",
    xaxis_title="Date",
    yaxis_title="Sales",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# 8. Summary statistics

st.subheader("Summary Statistics")

st.write("Train data stats:")
st.write(train["sales"].describe())

st.write("Test data stats:")
st.write(test["sales"].describe())

# 9. Download forecast CSV

st.subheader("Download Forecast")

forecast_csv = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False)

st.download_button(
    label="Download forecast CSV",
    data=forecast_csv,
    file_name="forecast.csv",
    mime="text/csv"
)