import numpy as np
import random
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import Callback

class DQNAgent:
    @staticmethod
    def create_dqn_model(input_shape, n_actions):
        model = Sequential()
        model.add(Dense(256, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.2))
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.2))
        model.add(Dense(n_actions, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def __init__(self, model, n_actions, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, gamma=0.95, batch_size=128, update_freq=2000, buffer_size=100000):
        self.model = model
        self.target_model = DQNAgent.create_dqn_model(model.input_shape[1:], n_actions)
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.steps = 0
        self.memory = ReplayBuffer(buffer_size)

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = self.memory.sample(self.batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in batch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                next_q_values = self.target_model.predict(next_state)
                target[0][action] = reward + self.gamma * np.max(next_q_values[0])

            states.append(state[0])
            targets.append(target[0])

        def step_decay(epoch):
            initial_lr = 0.001
            decay_rate = 0.5
            decay_step = 2000
            lr = initial_lr * decay_rate ** (epoch / decay_step)
            return lr

        lr_scheduler = LearningRateScheduler(step_decay)
        loss_history = LossHistory()

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0, callbacks=[lr_scheduler, loss_history])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss_history.losses


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        losses = self.replay()

        if self.steps % self.update_freq == 0:
            self.update_target_model()

        self.steps += 1

        return losses if losses is not None else 0

    
    def save(self, file_path):
        self.model.save(file_path)

    def load(self, file_path):
        self.model = load_model(file_path)

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)