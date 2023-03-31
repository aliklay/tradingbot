import numpy as np
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

def elliott_wave_analysis(price_data, window=50):

    # Find local maxima and minima
    local_max = argrelextrema(price_data, np.greater_equal, order=window)[0]
    local_min = argrelextrema(price_data, np.less_equal, order=window)[0]

    extrema = np.concatenate((local_max, local_min))
    extrema.sort()

    # Compute price differences
    diffs = np.diff(price_data[extrema])

    # Determine alternating maxima/minima
    alternating_extrema = [extrema[0]]
    for i, diff in enumerate(diffs[:-1]):
        if np.sign(diff) != np.sign(diffs[i + 1]):
            alternating_extrema.append(extrema[i + 1])

    alternating_extrema.append(extrema[-1])
    alternating_extrema = np.array(alternating_extrema)

    # Compute retracements
    retracements = compute_retracements(price_data, alternating_extrema)

    # Identify waves and corrections
    waves = [(alternating_extrema[i], alternating_extrema[i + 1], alternating_extrema[i + 2]) for i, retracement in enumerate(retracements) if retracement >= 0.618]
    corrections = [(alternating_extrema[i], alternating_extrema[i + 1], alternating_extrema[i + 2]) for i, retracement in enumerate(retracements) if retracement < 0.618]

    return waves, corrections
