# Creation and training of a DQN agent: 

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy

# Environment setup
from gym.envs.classic_control import CartPoleEnv
env = CartPoleEnv()

# Model architecture
model = Sequential([
    Dense(24, activation='relu', input_shape=(1,) + env.observation_space.shape),
    Dense(24, activation='relu'),
    Dense(env.action_space.n, activation='linear')
])

# DQN Agent configuration
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=memory, policy=policy)
dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])

# Training
dqn.fit(env, nb_steps=5000, visualize=False, verbose=2)
