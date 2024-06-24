import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

# Lookback periods for moving averages
short_lookback = 10
long_lookback = 30

# Commission rate
commission_rate = 0.0010

# Position limit in dollars
position_limit = 10000

# Minimum price movement threshold to trigger a trade (to reduce trading frequency)
price_threshold = 0.01  # 1% price change

# Holding period to avoid frequent trading
holding_period = 5
last_trade_time = np.zeros(nInst)

# RSI parameters
rsi_period = 14
overbought_threshold = 70
oversold_threshold = 30

def calculate_rsi(prices, period):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = np.mean(gain[-period:])
    avg_loss = np.mean(loss[-period:])
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def getMyPosition(prcSoFar):
    global currentPos, last_trade_time
    (nIns, nt) = prcSoFar.shape
    if (nt < long_lookback + 1):
        return np.zeros(nIns)

    # Calculate short-term and long-term moving averages
    short_mavg = np.mean(prcSoFar[:, -short_lookback:], axis=1)
    long_mavg = np.mean(prcSoFar[:, -long_lookback:], axis=1)

    # Calculate RSI
    rsi = np.array([calculate_rsi(prcSoFar[i, :], rsi_period) for i in range(nIns)])

    # Calculate log returns for the last period
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])

    # Calculate volatility
    volatility = np.std(prcSoFar[:, -long_lookback:], axis=1)

    # Calculate the normalized returns
    lNorm = np.sqrt(lastRet.dot(lastRet))
    normalized_ret = lastRet / lNorm

    # Calculate the trend signal
    trend_signal = np.sign(short_mavg - long_mavg)

    # Adjust positions based on trend signal, RSI, and normalized returns
    trend_scaled_positions = 5000 * normalized_ret * trend_signal / prcSoFar[:, -1]

    # Adjust positions based on RSI (mean-reversion)
    rsi_signal = np.where(rsi > overbought_threshold, -1, np.where(rsi < oversold_threshold, 1, 0))
    rsi_scaled_positions = 5000 * normalized_ret * rsi_signal / prcSoFar[:, -1]

    # Combine trend and mean-reversion signals
    combined_positions = (trend_scaled_positions + rsi_scaled_positions) / 2

    # Scale positions by volatility (to avoid too large positions in volatile instruments)
    combined_positions /= volatility

    # Risk management: Limit maximum position size
    max_shares = position_limit / prcSoFar[:, -1]
    combined_positions = np.clip(combined_positions, -max_shares, max_shares)

    # Convert to integer positions
    rpos = np.array([int(x) for x in combined_positions])

    # Enforce the $10k position limit per stock
    for i in range(nIns):
        current_value = currentPos[i] * prcSoFar[i, -1]
        new_position = currentPos[i] + rpos[i]
        new_value = new_position * prcSoFar[i, -1]
        if abs(new_value) > position_limit:
            allowed_shares = position_limit / prcSoFar[i, -1]
            rpos[i] = int(np.sign(new_position) * allowed_shares - currentPos[i])

    # Apply holding period and price movement threshold to reduce trading frequency
    for i in range(nIns):
        if nt - last_trade_time[i] < holding_period or abs(lastRet[i]) < price_threshold:
            rpos[i] = 0  # No trade if within holding period or price movement is insignificant

    # Update current positions
    currentPos += rpos

    # Update last trade time for instruments where trades were made
    for i in range(nIns):
        if rpos[i] != 0:
            last_trade_time[i] = nt

    return currentPos

# Example usage:
# prcSoFar = np.random.rand(50, 100) * 100  # Random price data for testing
# currentPos = getMyPosition(prcSoFar)
# print("Current Positions:", currentPos)
