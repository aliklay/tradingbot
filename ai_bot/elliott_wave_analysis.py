import numpy as np
import pandas as pd
from numba import njit
from scipy.signal import argrelextrema

@njit
def compute_retracements(price_data, alternating_extrema):
    retracements = np.empty(len(alternating_extrema) - 2)
    for i in range(len(alternating_extrema) - 2):
        start, mid, end = alternating_extrema[i], alternating_extrema[i + 1], alternating_extrema[i + 2]
        retracement = abs(price_data[end] - price_data[mid])
        retracement /= abs(price_data[mid] - price_data[start])
        retracements[i] = retracement
    return retracements

def adaptive_window(price_data, volatility_multiplier=1):
    return int(volatility_multiplier * np.std(price_data[-100:]))

def elliott_wave_analysis(price_data, retracement_threshold=0.618, volatility_multiplier=1):

    window = adaptive_window(price_data, volatility_multiplier)

    local_max = argrelextrema(price_data, np.greater_equal, order=window)[0]
    local_min = argrelextrema(price_data, np.less_equal, order=window)[0]

    extrema = np.concatenate((local_max, local_min))
    extrema.sort()

    diffs = np.diff(price_data[extrema])

    alternating_extrema = [extrema[0]]
    for i, diff in enumerate(diffs[:-1]):
        if np.sign(diff) != np.sign(diffs[i + 1]):
            alternating_extrema.append(extrema[i + 1])

    alternating_extrema.append(extrema[-1])
    alternating_extrema = np.array(alternating_extrema)

    retracements = compute_retracements(price_data, alternating_extrema)

    waves = [(alternating_extrema[i], alternating_extrema[i + 1], alternating_extrema[i + 2]) for i, retracement in enumerate(retracements) if retracement >= retracement_threshold]
    corrections = [(alternating_extrema[i], alternating_extrema[i + 1], alternating_extrema[i + 2]) for i, retracement in enumerate(retracements) if retracement < retracement_threshold]

    return waves, corrections
