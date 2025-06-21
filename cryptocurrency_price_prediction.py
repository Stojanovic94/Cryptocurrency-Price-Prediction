# Install required libraries
!pip install -q yfinance autots matplotlib ipywidgets statsmodels prophet tensorflow

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autots import AutoTS
from datetime import timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import ipywidgets as widgets
from IPython.display import display, clear_output

# Define currency options
crypto_options = {
    "Bitcoin (BTC)": "BTC",
    "Ethereum (ETH)": "ETH",
    "Dogecoin (DOGE)": "DOGE",
    "Litecoin (LTC)": "LTC",
    "Cardano (ADA)": "ADA",
    "Ripple (XRP)": "XRP",
}

fiat_options = ["USD", "EUR", "GBP", "JPY", "RUB"]
forecast_length = 90
start_date = '2023-06-01'
end_date = '2025-06-01'

"""'''**Select cryptocurrency and fiat currency**
User selects:

- One cryptocurrency (e.g., BTC)
- One fiat currency (e.g., USD)

This enables dynamic downloading of data for a specific pair, e.g., BTC-USD.
'''
"""

# Dropdown to select cryptocurrency
crypto_select = widgets.Dropdown(
    options=crypto_options.keys(),
    description='Cryptocurrency:',
    value='Bitcoin (BTC)',
    style={'description_width': 'initial'}
)

# Dropdown to select fiat currency
fiat_select = widgets.Dropdown(
    options=fiat_options,
    description='Fiat Currency:',
    value='USD'
)

# Button to run the analysis
run_button = widgets.Button(description="📊 Run Analysis", button_style='success')

"""**Downloading historical data (June 1, 2023 – June 1, 2025)**
We use the "Close" column (closing price) as the target variable (y) for prediction.
"""

# Function triggered when the button is clicked
# Executes everything: fetches data, trains models, displays results

def on_button_click(b):
    clear_output(wait=True)
    display(crypto_select, fiat_select, run_button)

    selected_crypto = crypto_options[crypto_select.value]
    selected_fiat = fiat_select.value
    ticker = f"{selected_crypto}-{selected_fiat}"

    # Download data from Yahoo Finance
    df_raw = yf.download(ticker, start=start_date, end=end_date)
    if df_raw.empty or 'Close' not in df_raw.columns:
        print("⚠️ No data available.")
        return

    df = df_raw[['Close']].reset_index()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])

    # Plot historical prices
    plt.figure(figsize=(12, 4))
    plt.plot(df['ds'], df['y'], label='Historical Prices')
    plt.title(f"{ticker} - Historical Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid()
    plt.legend()
    plt.show()

    # Linear regression on historical trend
    df['Date_ordinal'] = df['ds'].map(pd.Timestamp.toordinal)
    X = df[['Date_ordinal']]
    y_reg = df['y']
    linear_model = LinearRegression().fit(X, y_reg)
    y_pred = linear_model.predict(X)

    plt.figure(figsize=(14, 4))
    plt.plot(df['ds'], df['y'], label='History', color='blue')
    plt.plot(df['ds'], y_pred, label='Linear Regression', color='red', linestyle='--')
    plt.title(f"{ticker} - Trend + Linear Regression")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Prepare training data
    df_train = df.copy()
    df_train = df_train[df_train['y'].notnull()]

    last_date = df_train['ds'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_length)

    results = {}

    # ARIMA model (AutoTS)
    arima_model = AutoTS(forecast_length=forecast_length, frequency='D', model_list=['ARIMA'], ensemble=None)
    arima_model = arima_model.fit(df_train, date_col='ds', value_col='y', id_col=None)
    arima_forecast = arima_model.predict().forecast
    arima_forecast.index = future_dates
    results['ARIMA'] = arima_forecast['y'].values

    # Holt-Winters model
    hw_model = ExponentialSmoothing(df_train['y'], trend='add', seasonal=None).fit()
    hw_forecast = hw_model.forecast(forecast_length)
    results['Holt-Winters'] = hw_forecast.values

    # Prophet model
    prophet_model = Prophet()
    prophet_model.fit(df_train)
    future_df = pd.DataFrame({'ds': future_dates})
    prophet_forecast = prophet_model.predict(future_df)
    results['Prophet'] = prophet_forecast['yhat'].values

    # LSTM model – normalization
    lstm_df = df_train.set_index('ds').copy()
    scaler_min = lstm_df['y'].min()
    scaler_max = lstm_df['y'].max()
    lstm_df['y'] = (lstm_df['y'] - scaler_min) / (scaler_max - scaler_min)

    window_size = 30
    X, y_lstm = [], []
    for i in range(window_size, len(lstm_df)):
        X.append(lstm_df['y'].values[i-window_size:i])
        y_lstm.append(lstm_df['y'].values[i])
    X, y_lstm = np.array(X), np.array(y_lstm)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    lstm_model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X, y_lstm, epochs=10, verbose=0)

    # Recursive prediction
    last_sequence = lstm_df['y'].values[-window_size:]
    predictions = []
    for _ in range(forecast_length):
        input_seq = np.reshape(last_sequence, (1, window_size, 1))
        pred = lstm_model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        last_sequence = np.append(last_sequence[1:], pred)

    # Denormalization
    lstm_predictions = np.array(predictions) * (scaler_max - scaler_min) + scaler_min
    results['LSTM'] = lstm_predictions

    # Visualization of all forecasts + history
    plt.figure(figsize=(14, 6))
    plt.plot(df['ds'], df['y'], label='History', color='black')
    for model_name, forecast in results.items():
        plt.plot(future_dates, forecast, label=model_name)
    plt.title(f"Model Comparison - {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid()
    plt.legend()
    plt.show()

    # Forecasts only (no history)
    pred_df = pd.DataFrame(results, index=future_dates)
    pred_df.plot(figsize=(12, 4), title="Forecast Comparison")
    plt.xlabel("Date")
    plt.ylabel("Predicted Price")
    plt.grid()
    plt.show()

# Launch interactive widgets
run_button.on_click(on_button_click)
display(crypto_select, fiat_select, run_button)

"""'''**Conclusion**

This project demonstrates how various time series and deep learning models can be used to predict cryptocurrency prices. Each model has unique strengths:

- *ARIMA* – suitable for short-term stable patterns.
- *Holt-Winters* – strong for data with trend components.
- *Prophet* – ideal for automatic detection of trend and seasonality.
- *LSTM* – powerful for capturing complex, nonlinear relationships.
'''
"""