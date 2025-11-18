# Example: managing noise in diffusion models:

import numpy as np

def linear_noise_schedule(t, max_noise=1.0, min_noise=0.01):
    return max_noise - (max_noise - min_noise) * t

# Simulating noise levels across 100 inference steps
steps = 100
noise_levels = [linear_noise_schedule(t/steps) for t in range(steps)]

print("Noise levels:", noise_levels)
