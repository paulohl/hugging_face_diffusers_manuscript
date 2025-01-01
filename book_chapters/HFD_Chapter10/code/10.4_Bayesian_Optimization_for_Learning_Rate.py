# Bayesian optimization is used to find the best learning rate for a Rainbow DQN agent. 
# The train_model function evaluates agent performance for each candidate learning rate, 
# iteratively refining the search for best values.

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Define the parameter space
param_space = [Real(1e-5, 1e-1, name='learning_rate')]

# Define objective function
def train_model(learning_rate):
    agent = RainbowDQN(learning_rate=learning_rate)
    performance = agent.train(env, episodes=100)
    return -performance  # Negative because we aim to maximize performance

# Optimize using Bayesian Optimization
results = gp_minimize(train_model, param_space, n_calls=20)

print(f"Best learning rate: {results.x[0]}")
