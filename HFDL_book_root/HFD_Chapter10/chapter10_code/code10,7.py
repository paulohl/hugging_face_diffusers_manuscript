# Reward normalization implementation example: 

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
