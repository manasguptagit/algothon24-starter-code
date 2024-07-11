
import numpy as np
from sklearn.linear_model import LinearRegression

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)
model = [LinearRegression() for _ in range(nInst)]
is_model_trained = [False] * nInst  # Track if the model is trained for each instrument

def getMyPosition(prcSoFar):
    global currentPos
    global model
    global is_model_trained

    nInst, nt = prcSoFar.shape
    window_size = 20  # Size of the rolling window to train the model
    forecast_horizon = 1  # How far ahead we predict

    if nt < window_size + forecast_horizon:
        return np.zeros(nInst)

    for i in range(nInst):
        # Prepare the data for the model
        start_index = max(0, 1 - window_size - forecast_horizon)
        end_index = nt - window_size - forecast_horizon

        # Ensure X and y have matching shapes by trimming the larger array
        X = np.log(prcSoFar[i, start_index:end_index] / prcSoFar[i, start_index + forecast_horizon:end_index + forecast_horizon])
        y = np.log(prcSoFar[i, window_size:] / prcSoFar[i, window_size - forecast_horizon:-forecast_horizon])

        # Ensure X and y have the same length
        min_length = min(len(X), len(y))
        X = X[:min_length]
        y = y[:min_length]

        # Ensure X is 2D and y is 1D
        X = X.reshape(-1, 1)  # Adjust shape as needed

        # Train the model if not already trained
        if not is_model_trained[i]:
            model[i].fit(X, y)
            is_model_trained[i] = True

        # Use the last window of data to predict the future return
        recent_data = np.log(prcSoFar[i, -window_size:] / prcSoFar[i, -window_size - 1:-1]).reshape(1, -1)
        predicted_return = model[i].predict(recent_data)

        # Convert predicted return to position
        rpos = 5000 * predicted_return / prcSoFar[i, -1]
        currentPos[i] = np.clip(currentPos[i] + rpos, -10000 / prcSoFar[i, -1], 10000 / prcSoFar[i, -1])

    return currentPos
