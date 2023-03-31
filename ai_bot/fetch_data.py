import pandas as pd
from binance.client import Client

def get_historical_data(symbol, interval, start_str, end_str):
    client = Client("5wQ2J9PDbqA5oLuNLH8GOalQabXPEouYy3kfVKPcSQzXzI6tPz3ysAqGNkjePP6z", "5p9jN1vfYXV6NOLEP5wAdqeFddL78QVFA3gIedd2bqvKz3jUBopBP9gWO5s2f5YE")
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    
    data = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    
    return data