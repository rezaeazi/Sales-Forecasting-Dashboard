# Sales Forecasting Dashboard

A Streamlit dashboard that forecasts daily sales using Prophet, accounting for holidays and promotional events.

## Features

- Interactive dashboard with Plotly charts
- Forecast horizon adjustable by user
- Model evaluation metrics (MAE, MAPE)
- Event-aware forecasting

## Installation

```bash
git clone <your-repo-url>
cd sales-forecasting-dashboard
conda create -n sales_ts python=3.11 -y
conda activate sales_ts
pip install -r requirements.txt
streamlit run app.py


Screenshots of the dashboard are available here: [screenshots/README.md](screenshots/README.md)
