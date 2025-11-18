# Data augmentation for image states: 

from imgaug import augmenters as iaa

# Define augmentation pipeline
augmenter = iaa.Sequential([
    iaa.Affine(rotate=(-15, 15)),  # Random rotations
    iaa.Fliplr(0.5),              # Random horizontal flips
    iaa.Multiply((0.8, 1.2))      # Brightness variations
])

# Function for augmenting state observations
def augment_state(state):
    augmented_state = augmenter.augment_image(state)
    return augmented_state

# Usage in training loop
for episode in range(num_episodes):
    state = env.reset()
    state = augment_state(state)  # Apply augmentation
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        next_state = augment_state(next_state)  # Apply augmentation
        agent.update(state, action, reward, next_state)
        state = next_state
