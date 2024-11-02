import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import pandas_ta as ta

def fetch_data():
    msft = yf.Ticker("ATKR")
    data = msft.history(interval="1d", period="max")
    return data

def preprocess_data(data):
    data["log_ret"] = np.log(data.Close) - np.log(data.Close.shift(1))
    data["RSILR"] = ta.rsi(data.log_ret, length=15)
    data["EMAFLR"] = ta.ema(data.log_ret, length=25)
    data["EMAMLR"] = ta.ema(data.log_ret, length=100)
    data["EMASLR"] = ta.ema(data.log_ret, length=150)
    data["log_ret_next"] = data["log_ret"].shift(-1)
    data["Target"] = (data["log_ret_next"] > data["log_ret"]).astype(int)
    data.dropna(inplace=True)
    data.reset_index(inplace=True)

    dates = data["Date"]
    columns_to_drop = ["Volume", "Close", "Date", "Open", "High", "Low"]
    data.drop(columns_to_drop, axis=1, inplace=True)
    
    return data, dates

def add_features(data):
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
        
        target_trend_column = f"Target_Trend_{horizon}"
        data[target_trend_column] = data["Target"].shift(1).rolling(horizon).sum()
        new_predictors.append(target_trend_column)
    
    data.dropna(inplace=True)
    return data, new_predictors

def simulate_log_ret(current_data, next_day_prediction, data, horizon=252):
    historical_log_rets = data["log_ret"].iloc[-horizon:]
    mean_log_ret = historical_log_rets.mean()
    std_log_ret = historical_log_rets.std()

    if next_day_prediction == 1:
        simulated_log_ret = np.random.normal(loc=mean_log_ret + std_log_ret, scale=std_log_ret)
    else:
        simulated_log_ret = np.random.normal(loc=mean_log_ret - std_log_ret, scale=std_log_ret)

    return simulated_log_ret

def predict_one_year_ahead(data, model, predictors, steps=252, horizon=252):
    current_data = data.iloc[-1:].copy()
    future_predictions = []
    future_data = []

    for _ in range(steps):
        latest_data = current_data[predictors]
        
        next_day_prediction = model.predict(latest_data)[0]
        future_predictions.append(next_day_prediction)
        
        simulated_log_ret = simulate_log_ret(current_data, next_day_prediction, data, horizon=horizon)
        
        new_row = current_data.copy()
        new_row["log_ret"] = simulated_log_ret
        
        new_row["RSILR"] = ta.rsi(pd.concat([data["log_ret"], pd.Series([simulated_log_ret])]), length=15).iloc[-1]
        new_row["EMAFLR"] = ta.ema(pd.concat([data["log_ret"], pd.Series([simulated_log_ret])]), length=25).iloc[-1]
        new_row["EMAMLR"] = ta.ema(pd.concat([data["log_ret"], pd.Series([simulated_log_ret])]), length=100).iloc[-1]
        new_row["EMASLR"] = ta.ema(pd.concat([data["log_ret"], pd.Series([simulated_log_ret])]), length=150).iloc[-1]
        
        for horizon in [2, 5, 10, 60, 120, 250, 500]:
            rolling_log_ret = pd.concat([data["log_ret"], pd.Series([simulated_log_ret])]).rolling(horizon).mean().iloc[-1]
            ratio_column = f"LogRet_Ratio_{horizon}"
            new_row[ratio_column] = simulated_log_ret / rolling_log_ret
            
            rsi_trend_column = f"RSI_Trend_{horizon}"
            new_row[rsi_trend_column] = pd.concat([data["RSILR"], pd.Series([new_row["RSILR"]])]).shift(1).rolling(horizon).mean().iloc[-1]
            
            target_trend_column = f"Target_Trend_{horizon}"
            new_row[target_trend_column] = pd.concat([data["Target"], pd.Series([new_row["Target"]])]).shift(1).rolling(horizon).sum().iloc[-1]
        
        # Convert the new row to a dictionary and append it to the future_data list
        future_data.append(new_row.iloc[0].to_dict())
        
        # Add the new row back to the dataset
        data = pd.concat([data, new_row])
        current_data = new_row

    # Convert future_data to a DataFrame after the loop
    future_data_df = pd.DataFrame(future_data)
    
    return future_predictions, future_data_df


def plot_bullish_predictions(data, future_data, dates, initial_price):
    plt.figure(figsize=(14, 7))
    
    # Ensure dates and prices have the same length
    valid_dates = dates.iloc[-len(data):]  # Adjust dates to match the length of data after preprocessing
    
    # Convert cumulative log returns to actual prices
    historical_prices = initial_price * np.exp(data["log_ret"].cumsum())
    
    plt.plot(valid_dates, historical_prices, label='Historical Stock Prices', color='blue')

    # Adjust future log returns to show a bullish bias
    # Adding a small bullish bias to the log returns
    future_data["log_ret"] += 0.0009  # Small adjustment to bias upward

    # Plot the simulated future prices
    future_dates = pd.date_range(start=valid_dates.iloc[-1] + pd.Timedelta(days=1), periods=len(future_data), freq='B')
    future_prices = historical_prices.iloc[-1] * np.exp(future_data["log_ret"].cumsum())
    plt.plot([valid_dates.iloc[-1], future_dates[0]], [historical_prices.iloc[-1], future_prices.iloc[0]], color='blue', linestyle='--')
    plt.plot(future_dates, future_prices, label='Predicted Future Stock Prices', color='green')

    plt.title('Historical and Bullish Simulated Future Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

def clean_data(data, predictors):
        """
        Cleans the data by replacing infinities with NaNs and then dropping any rows with NaNs in the predictors.
        """
        # Replace infinities with NaNs
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Drop rows with NaNs in the predictors
        data.dropna(subset=predictors, inplace=True)
        return data

def main():
    data = fetch_data()
    data, dates = preprocess_data(data)
    data, new_predictors = add_features(data)
    
    predictors = ["log_ret", "RSILR", "EMAFLR", "EMAMLR", "EMASLR"] + new_predictors
    data = clean_data(data, predictors)  # Clean the data after adding features

    train = data.iloc[:-100]
    test = data.iloc[-100:]

    model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
    model.fit(train[predictors], train["Target"])

    precision = precision_score(test["Target"], model.predict(test[predictors]))
    print(f"Precision Score: {precision}")

    future_predictions, future_data = predict_one_year_ahead(data, model, predictors, steps=252, horizon=252)
    
    # Get the initial stock price (last price from the preprocessed data)
    initial_price = fetch_data()["Close"].iloc[-len(data)]
    
    plot_bullish_predictions(data, future_data, dates, initial_price)

if __name__ == "__main__":
    main()