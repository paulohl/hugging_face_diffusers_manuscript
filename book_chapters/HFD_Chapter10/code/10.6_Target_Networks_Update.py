# the target network is updated every 5000 steps. 
# This periodic synchronization prevents the online network's frequent updates from destabilizing the training process.

update_frequency = 5000

# Training loop
for step in range(training_steps):
    action = agent.select_action(state)
    next_state, reward, done, info = env.step(action)
    agent.update(state, action, reward, next_state)

    # Update target network periodically
    if step % update_frequency == 0:
        agent.update_target_network()

    state = next_state
    if done:
        break
