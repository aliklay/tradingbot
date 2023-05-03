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
        self.last_trade_step = 0

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(3)  # Buy, sell, or hold
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.features.shape[1],), dtype=np.float32)

    def _get_reward(self, action):
        holding_duration_penalty = 0.0001
        profit_bonus = 2
        loss_penalty = 1.5

        current_price = self.features[self.current_step, 0]

        new_balance = self.balance
        new_position = self.position

        # Calculate new balance and position based on the action
        if action == 0:  # Buy
            if new_position == 0:
                amount_to_buy = new_balance / (1 + self.transaction_fee)
                new_position += amount_to_buy / (current_price + 1e-8)
                new_balance -= amount_to_buy
        elif action == 1:  # Sell
            if new_position > 0:
                amount_to_sell = new_position * current_price
                new_balance += amount_to_sell * (1 - self.transaction_fee)
                new_position = 0

        holding_duration = self.current_step - self.last_trade_step
        reward = (new_balance - self.initial_balance) + (new_position * current_price) - holding_duration * holding_duration_penalty

        # Apply bonuses and penalties
        reward *= profit_bonus if reward > 0 else loss_penalty

        return reward

    def step(self, action):
        self.current_step += 1
        current_price = self.features[self.current_step, 0]

        transaction_details = None
        if action == 0 and self.position == 0:  # Buy
            amount_to_buy = self.balance / (1 + self.transaction_fee)
            self.position += amount_to_buy / (current_price + 1e-8)
            self.balance -= amount_to_buy
            transaction_details = 'BUY'
        elif action == 1 and self.position > 0:  # Sell
            amount_to_sell = self.position * current_price
            self.balance += amount_to_sell * (1 - self.transaction_fee)
            self.position = 0
            transaction_details = 'SELL'

        reward = self._get_reward(action)
        done = self.current_step == len(self.features) - 1
        next_state = self.features[self.current_step]
        info = {'transaction': transaction_details}

        return next_state, reward, done, info

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0
        self.last_trade_step = 0
        return self.features[self.current_step]

    def render(self, mode='human'):
        if mode == 'human':
            print(f'Step: {self.current_step}, '
                  f'Balance: ${self.balance:.2f}, '
                  f'Position: {self.position:.8f}, '
                  f'Portfolio Value: ${(self.balance + self.position * self.features[self.current_step, 0]):.2f}')
        else:
            raise NotImplementedError(f'Render mode {mode} not supported')


