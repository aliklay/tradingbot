import numpy as np

def preprocess_data(price_data, waves, corrections):
    # Normalize the price data
    normalized_price_data = (price_data - np.min(price_data)) / (np.max(price_data) - np.min(price_data))

    # Calculate the Simple Moving Average (SMA)
    sma = np.convolve(normalized_price_data, np.ones(5) / 5, mode='valid')
    sma = np.pad(sma, (4, 0), 'constant', constant_values=0)

    # Ensure waves and corrections have the same length as price_data
    waves = np.pad(waves, ((0, len(price_data) - len(waves)), (0, 0)), 'constant', constant_values=0)
    corrections = np.pad(corrections, ((0, len(price_data) - len(corrections)), (0, 0)), 'constant', constant_values=0)

    # Print the shapes of the arrays
    print(f"price_data shape: {price_data.shape}")
    print(f"waves shape: {waves.shape}")
    print(f"corrections shape: {corrections.shape}")

    # Stack the features together
    elliott_wave_features = np.concatenate((waves, corrections), axis=-1)
    features = np.column_stack((normalized_price_data, sma, elliott_wave_features))

    return features



