import pandas as pd
from binance.client import Client

def get_historical_data(symbol, interval, start_str, end_str):
    client = Client("public_key", "secret_key")
    klines = client.get_historical_klines(symbol, interval, start_str, end_str)
    
    data = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    
    return data
