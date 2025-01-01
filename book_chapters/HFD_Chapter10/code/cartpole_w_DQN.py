# how DQN bridges the gap between traditional reinforcement learning and high-dimensional problems, 
# allowing agents to make informed decisions and improve performance through iterative learning.

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Initialize environment and parameters
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Define DQN model
model = Sequential([
    Dense(24, activation='relu', input_dim=state_size),
    Dense(24, activation='relu'),
    Dense(action_size, activation='linear')
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
