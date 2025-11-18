# Safe explorationnexample:
# the environment penalizes the agent for entering unsafe states (encouraging safer learning behavior): 

class SafeEnvWrapper:
    def __init__(self, env):
        self.env = env

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # Penalize unsafe states
        reward -= 10 if self.is_unsafe(state) else 0
        return state, reward, done, info

    def is_unsafe(self, state):
        # Define unsafe conditions
        return state[0] < -1.0 or state[0] > 1.0
