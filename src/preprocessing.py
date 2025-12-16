import pandas as pd

def load_sales_data(filepath: str) -> pd.DataFrame:
    """
    Load sales data from CSV file.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare sales time-series data.
    Steps:
    - Convert date column to datetime
    - Sort by date
    - Set date as index
    - Enforce daily frequency
    - Handle missing values
    """


    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Sort by time
    df = df.sort_values("date")

    # Set index 
    df = df.set_index("date")

    # Enforce daily frequency
    df = df.asfreq("D")

    # Handle missing values using time interpolation
    df["sales"] = df["sales"].interpolate(method="time")

    # Safety check (no negative sales)
    df["sales"] = df["sales"].clip(lower=0)

    return df

def validate_time_series(df: pd.DataFrame) -> None:
    """
    validate cleaned time-series data.
    Raises error if data is invalid.
    """

    if df.isnull().sum().any():
        raise ValueError("Missing values remain after preprocessing")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Index is not DatetimeIndex")
    
    if df.index.freq is None:
        raise ValueError("Time frequency is not set")
    
    print("Time-series data validation passed")