# the clip_reward function restricts rewards to the range of -1 to 1. 
# During training, raw rewards are processed through this function before being used for Q-value updates. 
# This ensures that the agent focuses on relative improvements rather than being influenced by anomalous events

# Function to normalize rewards
def clip_reward(reward):
    return max(-1, min(1, reward))

# Usage in training loop
for step in range(training_steps):
    action = agent.select_action(state)
    next_state, reward, done, info = env.step(action)
    clipped_reward = clip_reward(reward)
    agent.update(state, action, clipped_reward, next_state)
    state = next_state
    if done:
        break
