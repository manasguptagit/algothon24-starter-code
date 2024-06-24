import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Define the trading strategy
def moving_average_strategy(stock_data, short_window=40, long_window=100):
    # Compute short-term and long-term moving averages
    stock_data['Short_MA'] = stock_data['Close'].rolling(window=short_window, min_periods=1).mean()
    stock_data['Long_MA'] = stock_data['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Create signals
    stock_data['Signal'] = 0
    stock_data['Signal'][short_window:] = np.where(stock_data['Short_MA'][short_window:] > stock_data['Long_MA'][short_window:], 1, 0)
    stock_data['Position'] = stock_data['Signal'].diff()
    
    return stock_data

# Fetch historical stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Plot the data with signals
def plot_trading_strategy(stock_data, ticker):
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Close'], label=f'{ticker} Close Price', alpha=0.5)
    plt.plot(stock_data['Short_MA'], label='Short-term Moving Average', alpha=0.75)
    plt.plot(stock_data['Long_MA'], label='Long-term Moving Average', alpha=0.75)
    
    # Plot buy signals
    plt.plot(stock_data[stock_data['Position'] == 1].index, 
             stock_data['Short_MA'][stock_data['Position'] == 1], 
             '^', markersize=10, color='g', label='Buy Signal')
    
    # Plot sell signals
    plt.plot(stock_data[stock_data['Position'] == -1].index, 
             stock_data['Short_MA'][stock_data['Position'] == -1], 
             'v', markersize=10, color='r', label='Sell Signal')
    
    plt.title(f'Trading Strategy for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

# Main function
if __name__ == "__main__":
    # Parameters
    ticker = 'AAPL'
    start_date = '2022-01-01'
    end_date = '2023-01-01'
    
    # Get stock data
    stock_data = get_stock_data(ticker, start_date, end_date)
    
    # Apply the strategy
    stock_data = moving_average_strategy(stock_data)
    
    # Plot the strategy
    plot_trading_strategy(stock_data, ticker)
