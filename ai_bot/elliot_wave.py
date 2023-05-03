# import ccxt
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from peakdetect import peakdetect

# def get_binance_data(symbol, timeframe='1d', limit=500):
#     exchange = ccxt.binance()
#     data = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
#     df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
#     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
#     return df

# def classify_correction(wave, threshold=0.618):
#     retracement = abs(wave[2][1] - wave[1][1]) / abs(wave[1][1] - wave[0][1])

#     if retracement < threshold:
#         return 'zigzag'
#     elif retracement < 1:
#         return 'flat'
#     else:
#         return 'triangle'

# def elliot_wave_advanced(price_data, delta=0.1):
#     max_peaks, min_peaks = peakdetect(price_data, lookahead=10, delta=delta)

#     max_peaks = sorted(max_peaks, key=lambda x: x[0])
#     min_peaks = sorted(min_peaks, key=lambda x: x[0])

#     waves = []
#     corrections = []
#     i = 0
#     j = 0

#     while i < len(max_peaks) - 1 and j < len(min_peaks) - 1:
#         if max_peaks[i][0] < min_peaks[j][0]:
#             waves.append(max_peaks[i])
#             i += 1
#         else:
#             waves.append(min_peaks[j])
#             j += 1

#         if len(waves) >= 3:
#             correction = classify_correction(waves[-3:])
#             corrections.append((waves[-2][0], waves[-2][1], correction))

#     return waves, corrections



# symbol = 'BTC/USDT'
# timeframe = '1h'
# price_data_df = get_binance_data(symbol, timeframe)
# price_data = price_data_df['close'].values

# waves, corrections = elliot_wave_advanced(price_data, delta=0.1)

# # Plot the price data and identified waves
# plt.plot(price_data)
# for wave in waves:
#     plt.scatter(*wave, c='r', marker='o')

# for correction in corrections:
#     plt.annotate(correction[2], xy=(correction[0], correction[1]), fontsize=12, bbox=dict(facecolor='yellow', edgecolor='black', boxstyle='round,pad=0.2'))

# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.title(f'Elliott Wave Pattern ({symbol}, {timeframe})')
# plt.show()

import math

print(math.floor(5.67))


