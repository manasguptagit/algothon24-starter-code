import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('prices.txt', delimiter='   ')

for column in data.columns[1:]:
    plt.plot(data.index, data[column], label=column)

plt.title('Prices Over Time')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig('price_tracking.png')