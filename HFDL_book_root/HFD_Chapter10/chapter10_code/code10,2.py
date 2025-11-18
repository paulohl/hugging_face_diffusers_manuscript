# Example: simple Q-learning

import gym
import numpy as np

# Initialize environment and Q-table
env = gym.make('FrozenLake-v1')
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Define parameters
learning_rate = 0.8
discount_factor = 0.95
num_episodes = 1000

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[state, :] + np.random.randn(1, env.action_space.n) * (1.0 / (episode + 1)))
        new_state, reward, done, _ = env.step(action)
        q_table[state, action] += learning_rate * (
            reward + discount_factor * np.max(q_table[new_state, :]) - q_table[state, action]
        )
        state = new_state
