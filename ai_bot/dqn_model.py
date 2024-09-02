import numpy as np
import random
from collections import deque
from keras.models import Sequential, load_model
from keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, Callback

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size, alpha=0.6):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.alpha = alpha

    def add(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        priorities = priorities ** self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()

        batch = list(zip(*samples))
        states, actions, rewards, next_states, dones = batch
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), weights, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


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

    def __init__(self, model, n_actions, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.99, gamma=0.95, batch_size=128, update_freq=2000, buffer_size=100000, alpha=0.6, beta=0.4):
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
        self.memory = PrioritizedReplayBuffer(buffer_size, alpha)
        self.beta = beta

    def step_decay(self, epoch):
        initial_lr = 0.001
        decay_rate = 0.5
        decay_step = 2000
        lr = initial_lr * decay_rate ** (epoch / decay_step)
        return lr

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

        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size, self.beta)
        
        # Ta bort onödig dimension
        states = np.squeeze(states, axis=1)
        next_states = np.squeeze(next_states, axis=1)

        targets = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)

        for i, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
            target = reward
            if not done:
                best_next_action = np.argmax(self.model.predict(np.expand_dims(next_state, axis=0))[0])
                target += self.gamma * next_q_values[i][best_next_action]
            targets[i][action] = target

        lr_scheduler = LearningRateScheduler(self.step_decay)
        self.model.fit(states, targets, sample_weight=weights, epochs=1, verbose=0, callbacks=[lr_scheduler])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return np.mean(targets)

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
