import numpy as np
import pandas as pd

np.random.seed(42)

# 1. date range

dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")
n = len(dates)

# 2. Base trend (business growth)

trend = np.linspace(50, 200, n)

# 3. Yearly seasonality

seasonality = 20 * np.sin(2 * np.pi * dates.dayofyear / 365)

# 4. weekly seasonality

weekly_effect = np.where(dates.weekday >= 5, 15, 0)

# 5. Random noise

noise = np.random.normal(0, 8, n)

# 6. Promotional campaigns

promo = np.zeros(n)
promo_days = np.random.choice(n, size=40, replace=False)
promo[promo_days] = np.random.uniform(30, 60, size=40)

# 7. Final sales

sales = trend + seasonality + weekly_effect + noise + promo
sales = np.maximum(sales, 5)

# 8. Create Dataframe

df = pd.DataFrame({
    "date": dates,
    "sales": sales.round(0)
})

# 9. Introduce missing values

missing_idx = np.random.choice(df.index, size=15, replace=False)
df.loc[missing_idx, "sales"] = np.nan

# 10. Save to CSV

df.to_csv("data/sales.csv", index=False)

print("sales.csv generated successfully")