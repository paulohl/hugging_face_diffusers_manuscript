# AlphaZero for energy grid management


class EnergyGrid:
    def __init__(self):
        self.energy_supply = 100
        self.energy_demand = 50

def step(self, action):
        if action == "increase":
           self.energy_supply += 10
        elif action == "decrease" and self.energy_supply > 10:
           self.energy_supply -= 10
           self.energy_demand = np.random.randint(30, 70)  # Demand varies
        reward = -abs(self.energy_supply - self.energy_demand)
        return self.energy_supply, reward
  
# Simulate AlphaZero's decision-making process in the energy grid
grid = EnergyGrid()
for _ in range(20):  # Run a few steps
    action = np.random.choice(["increase",
"decrease"])
    supply, reward = grid.step(action)
    print(f"Action: {action}, Supply: {supply}, Reward: {reward}")
