import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

def fetch_data(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    data = stock.history(interval="1d", period="max")
    return data

def preprocess_data(data):
    # Calculate log returns and technical indicators
    data["log_ret"] = np.log(data.Close) - np.log(data.Close.shift(1))
    data["RSILR"] = ta.rsi(data.log_ret, length=15)
    data["EMAFLR"] = ta.ema(data.log_ret, length=25)
    data["EMAMLR"] = ta.ema(data.log_ret, length=100)
    data["EMASLR"] = ta.ema(data.log_ret, length=150)
    
    # Add the up/down mechanism based on the next day's price
    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)  # 1 if the next day's close is higher than today's

    data.dropna(inplace=True)
    data.reset_index(inplace=True)

    # Store the dates column
    dates = data["Date"]

    # Drop only the columns that are irrelevant for the model
    columns_to_drop = ["Volume", "Close", "Date", "Open", "High", "Low"]
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

def plot_correlation_heatmap(data):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.show()

def plot_enhanced_backtest(data, predictions, dates):
    plt.figure(figsize=(14, 7))
    
    # Calculate cumulative log returns and plot
    cumulative_log_returns = data["log_ret"].cumsum()
    plt.plot(dates, cumulative_log_returns, label='Cumulative Log Returns', color='blue')
    
    # Ensure predictions is a Series
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions["Predictions"]

    # Align dates and predictions properly
    prediction_dates = dates[predictions.index]
    
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

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:, 1]
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
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

def main(stock_symbol):
    # Fetch and preprocess the data
    data = fetch_data(stock_symbol)
    data, dates = preprocess_data(data)
    print(f"Total rows in data: {len(data)}")

    # Define predictors and split the dataset into training and testing sets
    predictors = ["log_ret", "RSILR", "EMAFLR", "EMAMLR", "EMASLR"]
    train = data.iloc[:-100]
    test = data.iloc[-100:]

    # Initialize the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=25, min_samples_split=100, random_state=1)

    # Testing different numbers of estimators
    estimators = np.arange(25, 250, 25)
    train_accuracies = {}
    test_accuracies = {}
    for estimator in estimators:
        model = RandomForestClassifier(n_estimators=estimator, min_samples_split=100, random_state=1)
        model.fit(train[predictors], train["Target"])
        train_accuracies[estimator] = model.score(train[predictors], train["Target"])
        test_accuracies[estimator] = model.score(test[predictors], test["Target"])

    plot_accuracy(estimators, train_accuracies, test_accuracies, "Number of Estimators", "Accuracy")

    # Testing different sample splits
    samplesplits = np.arange(25, 250, 25)
    train_accuracies = {}
    test_accuracies = {}
    for samplesplit in samplesplits:
        model = RandomForestClassifier(n_estimators=25, min_samples_split=samplesplit, random_state=1)
        model.fit(train[predictors], train["Target"])
        train_accuracies[samplesplit] = model.score(train[predictors], train["Target"])
        test_accuracies[samplesplit] = model.score(test[predictors], test["Target"])

    plot_accuracy(samplesplits, train_accuracies, test_accuracies, "Min Sample Split", "Accuracy")

    # Final model setup and evaluation
    model = RandomForestClassifier(n_estimators=25, min_samples_split=100, random_state=1)
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    precision = precision_score(test["Target"], preds)
    print(f"Precision Score: {precision}")
    print(classification_report(test["Target"], preds))
    print("Confusion Matrix:")
    print(confusion_matrix(test["Target"], preds))

    # Perform backtesting
    predictions = backtest(data, model, predictors)
    print(predictions["Predictions"].value_counts())
    backtest_precision = precision_score(predictions["Target"], predictions["Predictions"])
    print(f"Backtest Precision Score: {backtest_precision}")

    # Backtest visualization
    plot_enhanced_backtest(data=data, predictions=predictions["Predictions"], dates=dates)

    # Predict tomorrow's movement using the last available data
    latest_data = data.iloc[-1:][predictors]
    next_day_prediction = model.predict(latest_data)[0]

    if next_day_prediction == 1:
        print(f"The model predicts that {stock_symbol}'s price will go UP tomorrow.")
    else:
        print(f"The model predicts that {stock_symbol}'s price will go DOWN tomorrow.")

if __name__ == "__main__":
    stock_symbol = "ATKR"  # You can switch to other stock symbols like "AAPL" as needed
    main(stock_symbol)
