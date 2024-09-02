import time
import logging
from binance.client import Client
import numpy as np
from keras.models import load_model
from trading_environment import TradingEnvironment
from dqn_model import DQNAgent
from elliott_wave_analysis import elliott_wave_analysis
from preprocess_data import preprocess_data
from fetch_data import get_historical_data
import math

logging.basicConfig(level=logging.INFO, filename='trading_bot.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

def get_latest_price(symbol):
    ticker = binance.get_symbol_ticker(symbol=symbol)
    return float(ticker['price'])

def get_trading_rules(symbol):
    exchange_info = binance.get_exchange_info()
    for s in exchange_info['symbols']:
        if s['symbol'] == symbol:
            filters = {f['filterType']: f for f in s['filters']}
            return {
                'baseAsset': s['baseAsset'],
                'quoteAsset': s['quoteAsset'],
                'limits': {
                    'cost': {'min': float(filters['MIN_NOTIONAL']['minNotional'])},
                    'amount': {'min': float(filters['LOT_SIZE']['minQty'])},
                    'price': {'min': float(filters['PRICE_FILTER']['tickSize'])},
                },
                'filters': filters
            }
    return None

def round_step(value, step_size):
    decimal_places = int(abs(math.log10(step_size)))
    return round(math.floor(value / step_size) * step_size, decimal_places)

def perform_trade(action, symbol, balance_data):
    percentage_of_balance_to_trade = 1
    trading_rules = get_trading_rules(symbol)
    
    base_asset = trading_rules['baseAsset']
    quote_asset = trading_rules['quoteAsset']
    
    balances = {item['asset']: float(item['free']) for item in balance_data['balances']}
    
    if action == 0:  # Buy
        amount_to_buy = balances[quote_asset] * percentage_of_balance_to_trade
        min_trade_size = float(trading_rules['limits']['cost']['min'])
        
        if amount_to_buy > min_trade_size:
            price_increment = float(trading_rules['limits']['price']['min'])
            adjusted_amount_to_buy = round_step(amount_to_buy, price_increment)

            logging.info(f"amount_to_buy: {amount_to_buy}, adjusted_amount_to_buy: {adjusted_amount_to_buy}")

            order = binance.create_order(symbol=symbol,
                                         side="BUY",
                                         type="MARKET",
                                         quoteOrderQty=adjusted_amount_to_buy)
            logging.info(f"Buy order placed: {order}")
    elif action == 1:  # Sell
        amount_to_sell = balances[base_asset] * percentage_of_balance_to_trade
        min_trade_size = float(trading_rules['limits']['amount']['min'])
        
        if amount_to_sell > min_trade_size:
            amount_increment = float(trading_rules['limits']['amount']['min'])
            adjusted_amount_to_sell = round_step(amount_to_sell, amount_increment)
            
            order = binance.create_order(symbol=symbol,
                                         side="SELL",
                                         type="MARKET",
                                         quantity=adjusted_amount_to_sell)
            logging.info(f"Sell order placed: {order}")

binance = Client('api_public_key', 'api_secret_key')

symbol = 'BTCUSDT'
timeframe = '1h'

historical_data = get_historical_data(symbol, timeframe, "1 year ago UTC", "now")
price_data = historical_data['close'].astype(float).to_numpy()
waves, corrections = elliott_wave_analysis(price_data)
features = preprocess_data(price_data, waves, corrections)

env = TradingEnvironment(features[:int(0.7 * len(features))])

model_weights_filepath = 'dqn_model_weights.h5'
model = load_model(model_weights_filepath)
input_shape = env.observation_space.shape
n_actions = 3  # Number of possible actions: Buy, Sell, Hold
agent = DQNAgent(model, n_actions)

trading_interval = 60 * 60

while True:
    try:
        balance = binance.get_account()
        latest_price = get_latest_price(symbol)
        price_data = np.append(price_data, latest_price)
        waves, corrections = elliott_wave_analysis(price_data[-1000:])
        new_features = preprocess_data(price_data[-1000:], waves, corrections)
        state = np.array([new_features[-1]])

        action = agent.act(state)
        logging.info(f"Action chosen: {action}")
        perform_trade(action, symbol, balance)

        time.sleep(trading_interval)
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
        time.sleep(trading_interval)
