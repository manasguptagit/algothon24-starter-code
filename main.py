import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

Sequential = keras.models.Sequential
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Adam = keras.optimizers.Adam

# Parameters for the strategy
short_lookback = 10
long_lookback = 30
rsi_period = 14
overbought_threshold = 70
oversold_threshold = 30
commission_rate = 0.0010
position_limit = 5000  # Increased position limit to allow for larger positions
fibonacci_lookback = 30
bollinger_band_period = 20
stochastic_period = 14
macd_short_period = 12
macd_long_period = 26
macd_signal_period = 9

# Adjusted weights for the signals to be more responsive
momentum_weight = 12
rsi_weight = 4
fibonacci_weight = 3
bollinger_weight = 2
stochastic_weight = 3
macd_weight = 3

# Risk management parameters
stop_loss_threshold = 0.02  # Slightly tighter stop loss to reduce risk
take_profit_threshold = 0.04  # Increased take profit threshold

# Helper functions for indicators
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
        return np.zeros(len(prices)), np.zeros(len(prices))  # Neutral MACD if not enough data
    short_ema = pd.Series(prices).ewm(span=short_period, min_periods=short_period).mean().values
    long_ema = pd.Series(prices).ewm(span=long_period, min_periods=long_period).mean().values
    macd = short_ema - long_ema
    signal = pd.Series(macd).ewm(span=signal_period, min_periods=signal_period).mean().values
    return macd, signal

# Load data
file_path = 'prices.txt'
data = pd.read_csv(file_path, sep="\s+", header=None)

# Assuming the data contains stock prices over multiple days
data = data.values  # Convert to numpy array

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Split the data into training and test sets
train_size = 500
train_data, test_data = data_scaled[:train_size], data_scaled[train_size:]

# Create technical indicators
def create_technical_indicators(data):
    df = pd.DataFrame(data, columns=['price'])
    df['SMA_10'] = df['price'].rolling(window=10).mean()
    df['SMA_30'] = df['price'].rolling(window=30).mean()
    df['RSI_14'] = df['price'].rolling(window=14).apply(lambda x: calculate_rsi(x, 14))
    macd, signal = calculate_macd(df['price'].values, 12, 26, 9)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    upper_band, sma_20, lower_band = calculate_bollinger_bands(df['price'].values, 20)
    df['Upper_Band'] = upper_band
    df['SMA_20'] = sma_20
    df['Lower_Band'] = lower_band
    df.dropna(inplace=True)
    return df

# Create lag features for prediction
def create_lag_features(data, lags):
    df = pd.DataFrame(data)
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df[0].shift(lag)
    df.dropna(inplace=True)
    return df

# Create features and target for model training
lags = 5
features = []
targets = []
for i in range(data.shape[1]):
    technical_indicators = create_technical_indicators(data[:, i])
    df = create_lag_features(technical_indicators.values, lags)
    if not df.empty:  # Ensure there are enough data points
        features.append(df.iloc[:, 1:].values)
        targets.append(df.iloc[:, 0].values)

if features and targets:  # Ensure we have enough data to train the model
    features = np.concatenate(features)
    targets = np.concatenate(targets)

    # Reshape data for LSTM
    features = features.reshape((features.shape[0], features.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(features.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # Train the model
    model.fit(features, targets, epochs=50, batch_size=32, verbose=0)
else:
    model = None

# Function to predict future prices using the trained model
def predict_future_prices(prices, model, scaler, lags):
    if model is None:
        return prices[-1]  # Return the last price if the model is not trained
    df = create_lag_features(prices[-lags:], lags)
    if df.empty:
        return prices[-1]  # Return the last price if not enough data for lags
    scaled_features = scaler.transform(df.iloc[:, 1:])
    scaled_features = scaled_features.reshape((scaled_features.shape[0], scaled_features.shape[1], 1))
    predictions = model.predict(scaled_features)
    return predictions[-1]

# Main function to get positions
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

        # Predict future price
        predicted_price = predict_future_prices(prices, model, scaler, lags)

        # Calculate signals
        momentum_signal = np.sign(momentum)
        rsi_signal = -1 if rsi > overbought_threshold else (1 if rsi < oversold_threshold else 0)
        fibonacci_signal = 1 if prices[-1] < fibonacci_levels[2] else -1
        bollinger_signal = 1 if prices[-1] < lower_band else (-1 if prices[-1] > upper_band else 0)
        stochastic_signal = 1 if stochastic_k < 20 else (-1 if stochastic_k > 80 else 0)
        macd_signal = 1 if macd[-1] > signal[-1] else -1  # Use the last value
        prediction_signal = 1 if predicted_price > prices[-1] else -1

        # Combine signals with weights
        combined_signal = (
            momentum_weight * momentum_signal +
            rsi_weight * rsi_signal +
            fibonacci_weight * fibonacci_signal +
            bollinger_weight * bollinger_signal +
            stochastic_weight * stochastic_signal +
            macd_weight * macd_signal +
            prediction_signal * 3  # Adding a weight to the prediction signal
        )

        # Determine position
        if combined_signal > 0:
            newPos[i] = position_limit // prices[-1]  # Buy larger positions
        elif combined_signal < 0:
            newPos[i] = -position_limit // prices[-1]  # Sell larger positions
        else:
            newPos[i] = 0  # Hold

        # Apply risk management
        if newPos[i] > 0:  # Long position
            entry_price = prices[-1]
            stop_loss_price = entry_price * (1 - stop_loss_threshold)
            take_profit_price = entry_price * (1 + take_profit_threshold)
        elif newPos[i] < 0:  # Short position
            entry_price = prices[-1]
            stop_loss_price = entry_price * (1 + stop_loss_threshold)
            take_profit_price = entry_price * (1 - take_profit_threshold)

        if newPos[i] != 0:
            for j in range(t):
                if (newPos[i] > 0 and (prices[j] <= stop_loss_price or prices[j] >= take_profit_price)) or \
                   (newPos[i] < 0 and (prices[j] >= stop_loss_price or prices[j] <= take_profit_price)):
                    newPos[i] = 0
                    break

    return newPos
