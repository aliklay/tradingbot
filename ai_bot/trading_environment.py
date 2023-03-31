import gym
import numpy as np

class TradingEnvironment(gym.Env):
    def __init__(self, features, initial_balance=10000, transaction_fee=0.001):
        super(TradingEnvironment, self).__init__()

        self.features = features
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.transaction_fee = transaction_fee
        self.current_step = 0
        self.position = 0
        

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # Buy, sell, or hold
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.features.shape[1],), dtype=np.float32)

    def step(self, action):
        self.current_step += 1

        # Get the current price of the asset
        current_price = self.features[self.current_step, 0]
        print(f"current_price: {current_price}")

        # Perform the action (buy, sell, or hold)
        if action == 0:  # Buy
            if self.position == 0:
                amount_to_buy = self.balance / (1 + self.transaction_fee)
                print(f"amount_to_buy: {amount_to_buy}")
                self.position += amount_to_buy / (current_price + 1e-8)
                self.balance -= amount_to_buy
        elif action == 1:  # Sell
            if self.position > 0:
                amount_to_sell = self.position * current_price
                print(f"amount_to_sell: {amount_to_sell}")
                self.balance += amount_to_sell * (1 - self.transaction_fee)
                self.position = 0

        # Calculate the reward
        portfolio_value = self.balance + self.position * current_price
        print(f"portfolio_value: {portfolio_value}")
        reward = portfolio_value - self.initial_balance
        print(f"reward: {reward}")

        # Check if the episode is done
        done = self.current_step == len(self.features) - 1

        # Get the next state
        next_state = self.features[self.current_step]

        return next_state, reward, done, {}



    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0
        return self.features[self.current_step]

    def render(self, mode='human'):
        if mode == 'human':
            print(f'Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}')
        else:
            raise NotImplementedError(f'Render mode {mode} not supported')

