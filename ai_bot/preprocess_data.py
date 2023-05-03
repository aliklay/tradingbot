import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def simple_moving_average(data, window):
    return data.rolling(window=window).mean()

def rate_of_change(data, window):
    roc = np.zeros_like(data)
    for i in range(window, len(data)):
        roc[i] = (data[i] - data[i-window]) / data[i-window]
    return roc

def rate_of_change(data, window):
    return data.pct_change(periods=window)

def macd(price_data, short_window=12, long_window=26, signal_window=9):
    ewm_short = price_data.ewm(span=short_window).mean()
    ewm_long = price_data.ewm(span=long_window).mean()
    macd_line = ewm_short - ewm_long
    signal_line = macd_line.ewm(span=signal_window).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def preprocess_data(price_data, waves, corrections, sma_window=10, roc_window=10):
    price_data_df = pd.DataFrame(price_data, columns=['close'])
    price_data_df['wave'] = 0
    price_data_df['correction'] = 0

    for wave in waves:
        price_data_df.loc[wave[0]:wave[2], 'wave'] = 1

    for correction in corrections:
        price_data_df.loc[correction[0]:correction[2], 'correction'] = 1

    # Add the simple moving average as a feature
    price_data_df['sma'] = simple_moving_average(price_data_df['close'], sma_window)

    # Add the rate of change as a feature
    price_data_df['roc'] = rate_of_change(price_data_df['close'], roc_window)

    # Calculate MACD, signal line, and histogram
    macd_line, signal_line, histogram = macd(price_data_df['close'])

    # Add MACD features to the DataFrame
    price_data_df['macd_line'] = macd_line
    price_data_df['signal_line'] = signal_line
    price_data_df['histogram'] = histogram

    # Normalize the features
    features = price_data_df.values
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    return features