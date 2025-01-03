# This code simulates a simplified real-time strategy (RTS) game where an AI agent must gather resources, build units, and evaluate performance. 

class RTSGame:
    def __init__(self):
        self.resources = 100
        self.units = 0

    def step(self, action):
        if action == "gather":
            self.resources += 10
        elif action == "build" and self.resources >= 20:
            self.units += 1
            self.resources -= 20

    def evaluate(self):
        return self.resources + 10 * self.units
