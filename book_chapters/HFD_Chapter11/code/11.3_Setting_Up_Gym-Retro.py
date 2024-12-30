# Libretro and Gym-Retro stand out as indispensable tools for researchers and practitioners in reinforcement learning. 
# These platforms enable the emulation of retro gaming environments, offering a reliable framework to benchmark algorithms 
# and refine strategies in dynamic and controlled settings.

import retro

# Setup for Gym-Retro
def setup_retro(game, state):
    env = retro.make(game=game, state=state)
    return env

# Example usage
env = setup_retro('Airstriker-Genesis', 'Level1')
state = env.reset()
print("Environment initialized for Airstriker.")
