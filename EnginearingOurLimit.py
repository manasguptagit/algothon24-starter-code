import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from typing import Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

nInst = 50
currentPos = np.zeros(nInst)

# Parameters
threshold_open = 2.0
threshold_close = 0.5
correlation_threshold = 0.88
position_size = 100
max_position = 10000
min_position = -10000
rsi_period = 14
bollinger_period = 15
macd_short_period = 12
macd_long_period = 26
macd_signal_period = 9

def calculate_rsi(prices: np.array, period: int = 14) -> np.array:
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = np.mean(gain[-period:]) if len(gain) >= period else np.mean(gain)
    avg_loss = np.mean(loss[-period:]) if len(loss) >= period else np.mean(loss)
    rs = avg_gain / avg_loss if avg_loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices: np.array, period: int = 20) -> Tuple[np.array, np.array, np.array]:
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, sma, lower_band

def calculate_macd(prices: np.array, short_period: int = 12, long_period: int = 26, signal_period: int = 9) -> Tuple[np.array, np.array]:
    short_ema = prices.ewm(span=short_period, min_periods=short_period).mean()
    long_ema = prices.ewm(span=long_period, min_periods=long_period).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, min_periods=signal_period).mean()
    return macd, signal

def calculate_features(prices: pd.Series) -> pd.DataFrame:
    features = pd.DataFrame()
    features['rsi'] = calculate_rsi(prices.values, rsi_period)
    upper_band, sma, lower_band = calculate_bollinger_bands(prices, bollinger_period)
    features['bollinger_upper'] = upper_band
    features['bollinger_middle'] = sma
    features['bollinger_lower'] = lower_band
    macd, signal = calculate_macd(prices, macd_short_period, macd_long_period, macd_signal_period)
    features['macd'] = macd
    features['macd_signal'] = signal
    return features

# Read the prices.txt file, convert it to numerical values, and extract the first 250 days of data for 50 stocks
def loadPrices(fn):
    global nt, nInst
    df = pd.read_csv(fn, sep='\s+', header=None, index_col=None)
    (nt, nInst) = df.shape
    return df.values

histPrice = loadPrices("./prices.txt")

# Example feature generation
features_list = []
for i in range(nInst):
    prices = pd.Series(histPrice[:, i])  # Note the change to correctly extract each stock's prices over time
    features = calculate_features(prices)
    features_list.append(features)

# Combine all features into a single DataFrame
all_features = pd.concat(features_list, axis=0)

# Add target variable (future returns)
future_returns = (histPrice[1:, :] - histPrice[:-1, :]) / histPrice[:-1, :]
target_values = future_returns.flatten()

# Align the indices correctly for the 'target' column
all_features = all_features.iloc[:-nInst, :]  # Remove the last 'nInst' rows to match the target_values length
all_features['target'] = target_values

# Split the data into training and testing sets
X = all_features.drop(columns=['target'])
y = all_features['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a predictive model
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
print("Model Score:", model.score(X_test, y_test))

# Implement the predictive model into the trading strategy
def getMyPosition(histPrice: np.array):
    global currentPos
    nInst, nDays = histPrice.shape

    posDelta: np.array = np.zeros(nInst)

    if nDays > 200:
        prices = histPrice
        price_mean = prices.mean(axis=1)
        price_sd = np.std(prices, axis=1)
        price_mean_expanded = np.outer(price_mean, np.ones(nDays))
        price_central = prices - price_mean_expanded
        price_sd_expanded = np.outer(price_sd, np.ones(nDays))
        price_std = price_central / price_sd_expanded

        data = pd.DataFrame(prices).T
        corr_matrix = data.corr()

        max_corr = []
        for index, row in corr_matrix.iterrows():
            for i in range(len(row)):
                corr = row[i]
                if corr > correlation_threshold and i > index:
                    max_corr.append((corr, index, i))

        max_corr.sort(key=lambda x: x[0], reverse=True)

        pairs_coint = []
        for pair in max_corr:
            inst_1 = pair[1]
            inst_2 = pair[2]

            spread = price_std[inst_1] - price_std[inst_2]
            spread_adf = adfuller(spread)
            if spread_adf[1] <= 0.05:
                pairs_coint.append((spread_adf[1], inst_1, inst_2))

        if len(pairs_coint) > 0:
            price_diff = np.ndarray(shape=(len(pairs_coint), nDays))
            for i in range(len(pairs_coint)):
                inst_1 = pairs_coint[i][1]
                inst_2 = pairs_coint[i][2]

                price_diff[i] = price_std[inst_1] - price_std[inst_2]

            diff_mean = price_diff.mean(axis=1)
            diff_sd = np.std(price_diff, axis=1)
            diff_mean_expanded = np.outer(diff_mean, np.ones(nDays))
            diff_central = price_diff - diff_mean_expanded
            diff_sd_expanded = np.outer(diff_sd, np.ones(nDays))
            z_diff = diff_central / diff_sd_expanded

            for i in range(len(z_diff)):
                point = z_diff[i][-1]
                inst_1, inst_2 = get_inst(i, pairs_coint)
                if point > threshold_open:
                    long(inst_1, posDelta)
                    short(inst_2, posDelta)
                elif point < -threshold_open:
                    short(inst_1, posDelta)
                    long(inst_2, posDelta)
                elif abs(point) < threshold_close:
                    return_pos(inst_1, posDelta)
                    return_pos(inst_2, posDelta)

        else:
            for i in range(nInst):
                prices = pd.Series(histPrice[:, i])
                features = calculate_features(prices)
                predicted_return = model.predict(features.tail(1))

                if predicted_return > 0:
                    long(i, posDelta)
                elif predicted_return < 0:
                    short(i, posDelta)
                else:
                    return_pos(i, posDelta)

    pos = currentPos + posDelta
    currentPos = pos
    return currentPos

def get_inst(index, pairs_coint):
    return (pairs_coint[index][1], pairs_coint[index][2])

def long(index, posDelta):
    pos = currentPos[index] + posDelta[index]
    if pos < max_position:
        posDelta[index] += position_size

def short(index, posDelta):
    pos = currentPos[index] + posDelta[index]
    if pos > min_position:
        posDelta[index] -= position_size
        
def return_pos(index, posDelta):
    if currentPos[index] > 0:
        posDelta[index] -= position_size
    elif currentPos[index] < 0:
        posDelta[index] += position_size