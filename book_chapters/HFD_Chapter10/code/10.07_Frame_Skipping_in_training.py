# This code proves frame skipping by preprocessing states and pooling over multiple frames to capture essential transitions. 
# By focusing only on critical state changes, the agent learns efficiently without processing redundant information. 
# Grayscale conversion and resizing further reduce the input dimensions, streamlining neural network computations.

import cv2

# Function for preprocessing and skipping frames
def preprocess_and_skip_frames(env, skip=4):
    state_buffer = []
    state = env.reset()
    for _ in range(skip):
        next_state, reward, done, info = env.step(env.action_space.sample())
        gray_frame = cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        resized_frame = cv2.resize(gray_frame, (84, 84))           # Resize to 84x84
        state_buffer.append(resized_frame)
        if done:
            break
    return np.max(np.array(state_buffer), axis=0)  # Return max-pooled frame

# Usage in training loop
for episode in range(num_episodes):
    state = preprocess_and_skip_frames(env)
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        state = preprocess_and_skip_frames(env)
