import numpy as np
from keras.models import load_model
from trading_environment import TradingEnvironment
from dqn_model import DQNAgent
from preprocess_data import preprocess_data
from fetch_data import get_historical_data
from elliott_wave_analysis import elliott_wave_analysis
from binance.client import Client


def compute_portfolio_value(balance, position, current_price):
    return balance + position * current_price


symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1DAY
start_str = "3 year ago UTC"
end_str = "now"

price_data_df = get_historical_data(symbol, interval, start_str, end_str)
print(price_data_df.head())

price_data = price_data_df['close'].astype(float).to_numpy()

waves, corrections = elliott_wave_analysis(price_data)

features = preprocess_data(price_data, waves, corrections)

train_data = features[:int(0.7 * len(features))]
val_data = features[int(0.7 * len(features)):int(0.9 * len(features))]
test_data = features[int(0.9 * len(features)):]

env = TradingEnvironment(train_data)

input_shape = env.observation_space.shape
n_actions = env.action_space.n
model = DQNAgent.create_dqn_model(input_shape, n_actions)

agent = DQNAgent(model, n_actions)

n_episodes = 50
best_val_reward = -np.inf
model_weights_filepath = 'dqn_model_weights.h5'
patience = 5
no_improvement = 0
for episode in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, (1, *state.shape))
    done = False
    episode_reward = 0
    episode_loss = 0
    episode_loss_count = 0
    step_count = 0

    while not done:
        print(f"Episode {episode + 1}/{n_episodes}, Step {step_count + 1}")
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, (1, *next_state.shape))

        current_profit = compute_portfolio_value(env.balance, env.position, env.features[env.current_step, 0]) - env.initial_balance
        print(f"Step Profit: {current_profit}, Transaction: {info['transaction']}")

        losses = agent.train(state, action, reward, next_state, done)
        if losses:
            episode_loss += np.mean(losses)
            episode_loss_count += 1

        episode_reward += reward
        state = next_state

        step_count += 1

    portfolio_value = compute_portfolio_value(env.balance, env.position, env.features[env.current_step, 0])
    print(f"Episode {episode + 1}/{n_episodes} completed with reward {episode_reward}, average loss {episode_loss / episode_loss_count if episode_loss_count > 0 else 0}, and portfolio value {portfolio_value}")

    val_env = TradingEnvironment(val_data)
    state = val_env.reset()
    state = np.reshape(state, (1, *state.shape))
    done = False
    val_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = val_env.step(action)
        next_state = np.reshape(next_state, (1, *next_state.shape))

        val_reward += reward
        state = next_state

    print(f"Validation reward: {val_reward}")
    
    if val_reward > best_val_reward:
        best_val_reward = val_reward
        print(f"Saving model weights with validation reward {val_reward}")
        model.save(model_weights_filepath)

best_model = load_model(model_weights_filepath)

best_agent = DQNAgent(best_model, n_actions)

test_env = TradingEnvironment(test_data)
state = test_env.reset()
state = np.reshape(state, (1, *state.shape))
done = False
total_reward = 0

while not done:
    action = best_agent.act(state)
    next_state, reward, done, _ = test_env.step(action)
    next_state = np.reshape(next_state, (1, *next_state.shape))

    total_reward += reward
    state = next_state

print(f"Total reward on test data: {total_reward}")





