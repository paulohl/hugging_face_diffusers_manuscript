# This example shows how to augment rewards with added context. 
# By penalizing actions that lead to dangerous states and rewarding progress toward the goal, 
# the agent can better prioritize strategies that align with long-term goals.

# Function for custom reward shaping

def shape_reward(state, reward):
    # Example: Encourage the agent to reach a goal
    if state == GOAL_STATE:
        reward += 10
    elif state in DANGER_ZONE:
        reward -= 5
    return reward

# Usage in training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, raw_reward, done, info = env.step(action)
        shaped_reward = shape_reward(next_state, raw_reward)
        agent.update(state, action, shaped_reward, next_state)
        state = next_state
