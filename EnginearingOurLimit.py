import numpy as np
import pandas as pd

# Parameters for the strategy
short_lookback = 10
long_lookback = 30
rsi_period = 14
overbought_threshold = 80
oversold_threshold = 20
commission_rate = 0.0010
position_limit = 10000
fibonacci_lookback = 30
bollinger_band_period = 20
stochastic_period = 14
macd_short_period = 12
macd_long_period = 26
macd_signal_period = 9

# Weights for the signals
momentum_weight = 10
rsi_weight = 3
fibonacci_weight = 2
bollinger_weight = 4
stochastic_weight = 2
macd_weight = 2

# Risk management parameters
stop_loss_threshold = 0.010  # 2% stop loss
take_profit_threshold = 0.035  # 4% take profit

def calculate_moving_average(prices, period):
    if len(prices) < period:
        return np.mean(prices)
    return np.mean(prices[-period:])

def calculate_rsi(prices, period):
    if len(prices) < period:
        return 50  # Neutral RSI if not enough data
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = np.mean(gain[-period:]) if len(gain) >= period else np.mean(gain)
    avg_loss = np.mean(loss[-period:]) if len(loss) >= period else np.mean(loss)
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_fibonacci_levels(prices, period):
    if len(prices) < period:
        return [0, 0, 0, 0, 0]
    highest = np.max(prices[-period:])
    lowest = np.min(prices[-period:])
    diff = highest - lowest
    levels = [
        highest - diff * 0.236,
        highest - diff * 0.382,
        highest - diff * 0.500,
        highest - diff * 0.618,
        highest - diff * 0.764
    ]
    return levels

def calculate_bollinger_bands(prices, period):
    if len(prices) < period:
        return np.mean(prices), np.mean(prices), np.mean(prices)
    sma = calculate_moving_average(prices, period)
    std = np.std(prices[-period:])
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, sma, lower_band

def calculate_stochastic_oscillator(prices, period):
    if len(prices) < period:
        return 50  # Neutral Stochastic Oscillator if not enough data
    low = np.min(prices[-period:])
    high = np.max(prices[-period:])
    k = ((prices[-1] - low) / (high - low)) * 100
    return k

def calculate_macd(prices, short_period, long_period, signal_period):
    if len(prices) < long_period:
        return 0, 0  # Neutral MACD if not enough data
    short_ema = pd.Series(prices).ewm(span=short_period, min_periods=short_period).mean().values
    long_ema = pd.Series(prices).ewm(span=long_period, min_periods=long_period).mean().values
    macd = short_ema - long_ema
    signal = pd.Series(macd).ewm(span=signal_period, min_periods=signal_period).mean().values
    return macd[-1], signal[-1]

def getMyPosition(prcSoFar):
    nInst, t = prcSoFar.shape
    newPos = np.zeros(nInst)
    
    for i in range(nInst):
        prices = prcSoFar[i, :]

        # Calculate indicators
        momentum = prices[-1] - prices[-short_lookback] if t >= short_lookback else 0
        rsi = calculate_rsi(prices, rsi_period)
        fibonacci_levels = calculate_fibonacci_levels(prices, fibonacci_lookback)
        upper_band, sma, lower_band = calculate_bollinger_bands(prices, bollinger_band_period)
        stochastic_k = calculate_stochastic_oscillator(prices, stochastic_period)
        macd, signal = calculate_macd(prices, macd_short_period, macd_long_period, macd_signal_period)

        # Calculate signals
        momentum_signal = np.sign(momentum)
        rsi_signal = -1 if rsi > overbought_threshold else (1 if rsi < oversold_threshold else 0)
        fibonacci_signal = 1 if prices[-1] < fibonacci_levels[2] else -1
        bollinger_signal = 1 if prices[-1] < lower_band else (-1 if prices[-1] > upper_band else 0)
        stochastic_signal = 1 if stochastic_k < 20 else (-1 if stochastic_k > 80 else 0)
        macd_signal = 1 if macd > signal else -1

        # Combine signals with weights
        combined_signal = (
            momentum_weight * momentum_signal +
            rsi_weight * rsi_signal +
            fibonacci_weight * fibonacci_signal +
            bollinger_weight * bollinger_signal +
            stochastic_weight * stochastic_signal +
            macd_weight * macd_signal
        )

        # Determine position
        if combined_signal > 0:
            newPos[i] = 1  # Buy
        elif combined_signal < 0:
            newPos[i] = -1  # Sell
        else:
            newPos[i] = 0  # Hold

        # Apply risk management
        if newPos[i] == 1:  # Long position
            entry_price = prices[-1]
            stop_loss_price = entry_price * (1 - stop_loss_threshold)
            take_profit_price = entry_price * (1 + take_profit_threshold)
        elif newPos[i] == -1:  # Short position
            entry_price = prices[-1]
            stop_loss_price = entry_price * (1 + stop_loss_threshold)
            take_profit_price = entry_price * (1 - take_profit_threshold)

        if newPos[i] != 0:
            for j in range(t):
                if (newPos[i] == 1 and (prices[j] <= stop_loss_price or prices[j] >= take_profit_price)) or \
                   (newPos[i] == -1 and (prices[j] >= stop_loss_price or prices[j] <= take_profit_price)):
                    newPos[i] = 0
                    break

    return newPos

