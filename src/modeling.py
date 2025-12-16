import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def train_test_split_ts(df: pd.DataFrame, test_days: int = 90):
    """
    Split time-series data into train and test based on date.
    """
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    return train, test

def create_prophet_model(train_df, events_path="data/events.csv"):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    # Load events if exists
    try:
        events = pd.read_csv(events_path)
        events["ds"] = pd.to_datetime(events["ds"])
        model = Prophet(holidays=events)
    except FileNotFoundError:
        pass

    prophet_df = train_df.reset_index().rename(
        columns={"date": "ds", "sales": "y"}
    )

    model.fit(prophet_df)
    return model

def make_forecast(model: Prophet, periods: int = 90) -> pd.DataFrame:
    """
    Forecast future sales foe a given number of periods.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast

def plot_forecast(df_train: pd.DataFrame, df_forecast: pd.DataFrame):
    """
    Plot actual vs forecast
    """
    plt.figure(figsize=(12,6))
    plt.plot(df_train.index, df_train["sales"], lable="Actual")
    plt.plot(df_forecast["ds"], df_forecast["yhat"], lable="Forecast")
    plt.fill_between(df_forecast["ds"], df_forecast["yhat_lower"], df_forecast["yhat_upper"], alpha=0.2)
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.title("Sales Forecast")
    plt.show()