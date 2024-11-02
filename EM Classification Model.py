import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

def fetch_data():
    stock = yf.Ticker("ATKR")
    data = stock.history(period="max")
    data["Tomorrow"] = data["Close"].shift(-1)  # Target tomorrow's Close price
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)  # Binary classification
    data = data.loc["1990-01-01":].copy()  # Start data from 1990 onward
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

    # Store the dates column
    dates = data["Date"]

    # Drop only the columns that are irrelevant for the model
    columns_to_drop = ["Volume", "Date", "Open", "High", "Low"]
    data.drop(columns_to_drop, axis=1, inplace=True)
    
    return data, dates

def plot_accuracy(estimators, train_accuracies, test_accuracies, xlabel, ylabel):
    plt.figure()
    plt.plot(estimators, train_accuracies.values(), label="Training Accuracy")
    plt.plot(estimators, test_accuracies.values(), label="Testing Accuracy")
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plot_correlation_heatmap(data, columns):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.show()

def plot_enhanced_backtest(data, predictions, dates):
    plt.figure(figsize=(14, 7))
    
    # Calculate cumulative log returns and plot
    cumulative_log_returns = data["log_ret"].cumsum()
    
    # Align dates with cumulative_log_returns by slicing the dates
    dates_aligned = dates[-len(cumulative_log_returns):]
    plt.plot(dates_aligned, cumulative_log_returns, label='Cumulative Log Returns', color='blue')
    
    # Ensure predictions is a Series, if it's a DataFrame, assume the relevant data is in the first column
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.iloc[:, 0]  # Adjust as necessary to select the correct predictions column

    # Make sure dates and predictions are properly aligned
    prediction_dates = dates_aligned[predictions.index]  # Align dates with the predictions' index
    
    # Filter signals
    buy_signals = prediction_dates[predictions == 1]
    sell_signals = prediction_dates[predictions == 0]

    # Plot signals
    plt.scatter(buy_signals, cumulative_log_returns[buy_signals.index], color='green', marker='^', alpha=0.7, label='Buy Signals')
    plt.scatter(sell_signals, cumulative_log_returns[sell_signals.index], color='red', marker='v', alpha=0.7, label='Sell Signals')
    
    plt.title('Enhanced Backtest: Model Predictions vs. Market Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()


def plot_predictions(data, future_data, dates):
    plt.figure(figsize=(14, 7))

    # Plot the historical cumulative log returns
    cumulative_log_returns = data["log_ret"].cumsum()
    plt.plot(dates, cumulative_log_returns, label='Historical Cumulative Log Returns', color='blue')

    # Plot the simulated future log returns (cumulative)
    future_dates = pd.date_range(start=dates.iloc[-1] + pd.Timedelta(days=1), periods=len(future_data), freq='B')
    future_cumulative_log_returns = future_data["log_ret"].cumsum() + cumulative_log_returns.iloc[-1]
    plt.plot(future_dates, future_cumulative_log_returns, label='Simulated Future Cumulative Log Returns', color='orange')
    plt.title('Historical and Simulated Future Stock Movements')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Log Returns')
    plt.legend()
    plt.show()

def clean_data(data, predictors):
    # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    data = data.copy()
    # Replace infinities and -infinities with NaNs across the entire DataFrame
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop rows where any of the specified predictors contain NaNs
    data.dropna(subset=predictors, inplace=True)
    return data

def main():
    data = fetch_data()
    data, dates = preprocess_data(data)
    print(f"Total rows in data: {len(data)}")

    predictors = ["log_ret", "RSILR", "EMAFLR", "EMAMLR", "EMASLR"]
    train = data.iloc[:-100]
    test = data.iloc[-100:]

    # Testing different numbers of estimators
    estimators = np.arange(25, 250, 25)
    train_accuracies = {}
    test_accuracies = {}
    for estimator in estimators:
        model = RandomForestClassifier(n_estimators=estimator, min_samples_split=100, random_state=1)
        model.fit(train[predictors], train["Target"])
        train_accuracies[estimator] = model.score(train[predictors], train["Target"])
        test_accuracies[estimator] = model.score(test[predictors], test["Target"])

    #plot_accuracy(estimators, train_accuracies, test_accuracies, "Number of estimators", "Accuracy")

    # Testing different sample splits
    samplesplits = np.arange(25, 250, 25)
    train_accuracies = {}
    test_accuracies = {}
    for samplesplit in samplesplits:
        model = RandomForestClassifier(n_estimators=25, min_samples_split=samplesplit, random_state=1)
        model.fit(train[predictors], train["Target"])
        train_accuracies[samplesplit] = model.score(train[predictors], train["Target"])
        test_accuracies[samplesplit] = model.score(test[predictors], test["Target"])

    #plot_accuracy(samplesplits, train_accuracies, test_accuracies, "Min Sample Split", "Accuracy")

    # Final model setup and evaluation
    model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    precision = precision_score(test["Target"], preds)
    print(f"Precision Score: {precision}")

    # Backtesting function
    def predict(train, test, predictors, model):
        model.fit(train[predictors], train["Target"])
        preds = model.predict_proba(test[predictors])[:, 1]
        preds[preds >= 0.6] = 1
        preds[preds < 0.6] = 0
        preds = pd.Series(preds, index=test.index, name="Predictions")
        combined = pd.concat([test["Target"], preds], axis=1)
        return combined

    def backtest(data, model, predictors, start=200, step=50):
        all_predictions = []

        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i].copy()
            test = data.iloc[i:(i + step)].copy()
            predictions = predict(train, test, predictors, model)
            all_predictions.append(predictions)

        return pd.concat(all_predictions)

    predictions = backtest(data, model, predictors)
    print(predictions["Predictions"].value_counts())
    backtest_precision = precision_score(predictions["Target"], predictions["Predictions"])
    print(f"Backtest Precision Score: {backtest_precision}")

    horizons = [2, 5, 10, 60, 120, 250, 500]
    new_predictors = []
    

    for horizon in horizons:
    # 1. Rolling average using log returns
        rolling_log_ret = data["log_ret"].rolling(horizon).mean()
        ratio_column = f"LogRet_Ratio_{horizon}"
        data[ratio_column] = data["log_ret"] / rolling_log_ret
        new_predictors.append(ratio_column)
        
        # 2. Momentum-based trend using RSI
        rsi_trend_column = f"RSI_Trend_{horizon}"
        data[rsi_trend_column] = data["RSILR"].shift(1).rolling(horizon).mean()
        new_predictors.append(rsi_trend_column)
        
        # 3. Outcome-based trend using Target frequency
        target_trend_column = f"Target_Trend_{horizon}"
        data[target_trend_column] = data["Target"].shift(1).rolling(horizon).sum()
        new_predictors.append(target_trend_column)

    updated_predictors = new_predictors + predictors

    data = clean_data(data, updated_predictors)
    
    train_second = data.iloc[:-100]
    test_second = data.iloc[-100:]

    # Clean the training and testing datasets
    train_second = clean_data(train_second, updated_predictors)
    test_second = clean_data(test_second, updated_predictors)

    # Testing different numbers of estimators
    estimators = np.arange(25, 250, 25)
    train_accuracies = {}
    test_accuracies = {}
    for estimator in estimators:
        model = RandomForestClassifier(n_estimators=estimator, min_samples_split=100, random_state=1)
        model.fit(train_second[updated_predictors], train_second["Target"])
        train_accuracies[estimator] = model.score(train_second[updated_predictors], train_second["Target"])
        test_accuracies[estimator] = model.score(test_second[updated_predictors], test_second["Target"])

    #plot_accuracy(estimators, train_accuracies, test_accuracies, "Number of estimators", "Accuracy")

    # Testing different sample splits
    samplesplits = np.arange(25, 250, 25)
    train_accuracies = {}
    test_accuracies = {}
    for samplesplit in samplesplits:
        model = RandomForestClassifier(n_estimators=25, min_samples_split=samplesplit, random_state=1)
        model.fit(train_second[updated_predictors], train_second["Target"])
        train_accuracies[samplesplit] = model.score(train_second[updated_predictors], train_second["Target"])
        test_accuracies[samplesplit] = model.score(test_second[updated_predictors], test_second["Target"])

    #plot_accuracy(samplesplits, train_accuracies, test_accuracies, "Min Sample Split", "Accuracy")

    model_second = RandomForestClassifier(n_estimators=150, min_samples_split=100, random_state=1)
    model_second.fit(train_second[updated_predictors], train_second["Target"])
    preds_second = model_second.predict(test_second[updated_predictors])
    precision_second = precision_score(test_second["Target"], preds_second)
    print(f"Precision Score: {precision_second}")

    predictions_second = backtest(data, model_second, updated_predictors)
    print(predictions_second["Predictions"].value_counts())
    backtest_precision_second = precision_score(predictions_second["Target"], predictions_second["Predictions"])
    print(f"Backtest Precision Score: {backtest_precision_second}")

    latest_data = data.iloc[-1:][updated_predictors]
    next_day_prediction = model.predict(latest_data)[0]

    if next_day_prediction == 1:
        print("The model predicts that ATKR will go UP tommorrow.")
    else:
        print("The model predicts that ATKR will go DOWN tommorrow.}")
    
    #backtest visuallisation
    #plot_enhanced_backtest(data=data, predictions=predictions_second, dates=dates)

    #Ploting heatmaps 
    selected_columns = ['log_ret', 'RSILR', 'EMAMLR', 'EMASLR', 'LogRet_Ratio_250', 'LogRet_Ratio_500', 'RSI_Trend_250', 'RSI_Trend_500', 'Target_Trend_250', 'Target_Trend_500']
    plot_correlation_heatmap(data, selected_columns)



if __name__ == "__main__":
    main()