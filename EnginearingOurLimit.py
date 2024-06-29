
import numpy as np

##### TODO #########################################
### RENAME THIS FILE TO YOUR TEAM NAME #############
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50
currentPos = np.zeros(nInst)


def getMyPosition(prcSoFar):
    global currentPos
    nInst, nt = prcSoFar.shape
    if nt < 2:
        return np.zeros(nInst)
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2])

    # Normalize returns
    lNorm = np.linalg.norm(lastRet)
    if lNorm == 0:
        return np.zeros(nInst)
    lastRet /= lNorm

    # Calculate RSI for mean reversion 
    delta = prcSoFar[:, 1:] - prcSoFar[:, :-1]
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = np.mean(gain, axis=1)
    avg_loss = np.mean(loss, axis=1)
    rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
    rsi = 100 - 100 / (1 + rs)

    # Get position based on (overbought/oversold)
    momentum_scale = 5000 / prcSoFar[:, -1]
    rsi_threshold = 50
    rsi_adj = (rsi - rsi_threshold) / rsi_threshold 
    combined_signal = lastRet * rsi_adj

    # Update positions
    rpos = combined_signal * momentum_scale
    newPos = currentPos + np.round(rpos).astype(int)
    currentPos = newPos

    return currentPos
