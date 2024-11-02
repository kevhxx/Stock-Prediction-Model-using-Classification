import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def fetch_data():
    atkr = yf.Ticker("ATKR")
    data = atkr.history(period="max")
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = np.log(data["Tomorrow"] / data["Close"])  # Log returns as target for regression
    data = data.loc["1990-01-01":].copy()
    return data  

def preprocess_data(data):
    data["log_ret"] = np.log(data.Close) - np.log(data.Close.shift(1))
    data["RSILR"] = ta.rsi(data.log_ret, length=15)
    data["EMAFLR"] = ta.ema(data.log_ret, length=25)
    data["EMAMLR"] = ta.ema(data.log_ret, length=100)
    data["EMASLR"] = ta.ema(data.log_ret, length=150)
    data.dropna(inplace=True)
    data.reset_index(inplace=True)

    dates = data["Date"]
    columns_to_drop = ["Volume", "Date", "Open", "High", "Low"]
    data.drop(columns_to_drop, axis=1, inplace=True)
    
    return data, dates

def add_features_for_regression(data):
    horizons = [2, 5, 10, 60, 120, 250, 500]
    new_predictors = []

    for horizon in horizons:
        rolling_log_ret = data["log_ret"].rolling(horizon).mean()
        ratio_column = f"LogRet_Ratio_{horizon}"
        data[ratio_column] = data["log_ret"] / rolling_log_ret
        new_predictors.append(ratio_column)

        rsi_trend_column = f"RSI_Trend_{horizon}"
        data[rsi_trend_column] = data["RSILR"].shift(1).rolling(horizon).mean()
        new_predictors.append(rsi_trend_column)

        volatility_column = f"Volatility_{horizon}"
        data[volatility_column] = data["log_ret"].rolling(horizon).std()
        new_predictors.append(volatility_column)

        rolling_mean_column = f"RollingMean_LogRet_{horizon}"
        data[rolling_mean_column] = rolling_log_ret
        new_predictors.append(rolling_mean_column)

    data.dropna(inplace=True)
    return data, new_predictors

def clean_data(data, predictors):
    data = data.copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(subset=predictors, inplace=True)
    return data

def predict_one_year_ahead(data, model, predictors, steps=252, base_mean_reversion_weight=0.5):
    current_data = data.iloc[-1:].copy()
    future_predictions = []
    simulated_data = []

    long_term_mean = data["log_ret"].mean()
    historical_std = data["log_ret"].std()

    for step in range(steps):
        latest_data = current_data[predictors]
        next_day_log_ret = model.predict(latest_data)[0]
        
        # Clip log returns to prevent extreme values
        next_day_log_ret = np.clip(next_day_log_ret, -0.005, 0.005)
        
        # Apply mean reversion with an adaptive weight
        deviation_from_mean = abs(next_day_log_ret - long_term_mean)
        mean_reversion_weight = min(base_mean_reversion_weight + deviation_from_mean, 0.8)  # Cap weight at 0.8
        next_day_log_ret = (1 - mean_reversion_weight) * next_day_log_ret + mean_reversion_weight * long_term_mean
        
        # Add random noise based on historical variance
        random_noise = np.random.normal(0, historical_std * 0.2)  # Increased randomness
        next_day_log_ret += random_noise
        
        # Occasionally force a downward correction (10% chance)
        if np.random.rand() < 0.1:
            next_day_log_ret -= abs(np.random.normal(0, historical_std * 0.5))
        
        future_predictions.append(next_day_log_ret)
        
        new_row = current_data.copy()
        new_row["log_ret"] = next_day_log_ret
        
        # Update the dataset to keep continuity
        data = pd.concat([data, new_row], ignore_index=True)
        simulated_data.append(new_row.iloc[0].to_dict())

        current_data = new_row

    future_data = pd.DataFrame(simulated_data)
    return future_predictions, future_data

def plot_regression_predictions(data, future_data, dates, initial_price):
    plt.figure(figsize=(14, 7))

    valid_dates = dates.iloc[-len(data):]  
    historical_prices = initial_price * np.exp(data["log_ret"].cumsum())
    plt.plot(valid_dates, historical_prices, label='Historical Stock Prices', color='blue')

    future_dates = pd.date_range(start=valid_dates.iloc[-1] + pd.Timedelta(days=1), periods=len(future_data), freq='B')
    future_prices = historical_prices.iloc[-1] * np.exp(future_data["log_ret"].cumsum())
    
    plt.plot(future_dates, future_prices, label='Simulated Future Stock Prices', color='orange')
    plt.title('Historical and Simulated Future Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def main():
    data = fetch_data()
    data, dates = preprocess_data(data)
    print(f"Total rows in data: {len(data)}")

    data, horizon_predictors = add_features_for_regression(data)
    predictors = ["log_ret", "RSILR", "EMAFLR", "EMAMLR", "EMASLR"] + horizon_predictors
    data = clean_data(data, predictors)

    train = data.iloc[:-100]
    test = data.iloc[-100:]

    model = RandomForestRegressor(n_estimators=100, min_samples_split=50, random_state=1)
    model.fit(train[predictors], train["Target"])

    predictions = model.predict(test[predictors])
    mae = mean_absolute_error(test["Target"], predictions)
    mse = np.mean((test["Target"] - predictions) ** 2)

    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")

    future_predictions, future_data = predict_one_year_ahead(data, model, predictors, steps=252)

    initial_price = fetch_data()["Close"].iloc[-len(data)]
    plot_regression_predictions(data, future_data, dates, initial_price)

if __name__ == "__main__":
    main()
