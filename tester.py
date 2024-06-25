import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load price data
data = pd.read_csv('/mnt/data/prices.txt', header=None)
prices = data[0].values

# Parameters
future_steps = 50  # Number of future steps to predict
polynomial_degree = 4  # Degree of the polynomial regression

# Prepare data for polynomial regression
X = np.arange(len(prices)).reshape(-1, 1)
y = prices

# Fit polynomial regression model
poly = PolynomialFeatures(degree=polynomial_degree)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# Predict future prices
X_future = np.arange(len(prices), len(prices) + future_steps).reshape(-1, 1)
X_future_poly = poly.transform(X_future)
future_prices = model.predict(X_future_poly)

# Simulate a simple trading strategy
initial_balance = 10000  # Starting balance in dollars
balance = initial_balance
position = 0  # Number of shares held

for i in range(future_steps):
    if i == 0:
        # Initial purchase
        position = balance // future_prices[i]
        balance -= position * future_prices[i]
    elif future_prices[i] > future_prices[i - 1]:
        # Price is going up, buy more shares
        shares_to_buy = balance // future_prices[i]
        balance -= shares_to_buy * future_prices[i]
        position += shares_to_buy
    elif future_prices[i] < future_prices[i - 1] and position > 0:
        # Price is going down, sell all shares
        balance += position * future_prices[i]
        position = 0

# Calculate final balance
final_balance = balance + position * future_prices[-1]
profit = final_balance - initial_balance

# Output results
print(f"Initial balance: ${initial_balance:.2f}")
print(f"Final balance: ${final_balance:.2f}")
print(f"Profit: ${profit:.2f}")

# Plot original and predicted prices
plt.plot(prices, label='Original Prices')
plt.plot(np.arange(len(prices), len(prices) + future_steps), future_prices, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Original and Predicted Prices')
plt.legend()
plt.show()
