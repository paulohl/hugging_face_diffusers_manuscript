# Resource management simulation example: AlphaZero's methods also apply to real-world scenarios that require strategic resource management and 
# decision-making under constraints. 


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
