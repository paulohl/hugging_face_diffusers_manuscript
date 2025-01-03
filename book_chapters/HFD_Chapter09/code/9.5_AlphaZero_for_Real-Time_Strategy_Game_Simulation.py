# simplified version of a real-time strategy game where the AI must manage resources, build units, and defeat an opponent

class RTSGame:
    		def __init__(self):
        	self.resources = 100
        	self.units = 0
        	self.enemy_units = 5
        	self.time = 0

def simulate_action(self, action):
        	if action == "gather":
            	self.resources += 10	
        	elif action == "build":
            	if self.resources >= 20:
                		self.units += 1
                		self.resources -= 20
        	elif action == "attack":
            	if self.units > 0:
                		self.enemy_units -= 1
                self.units -= 1
        		self.time += 1

def is_game_over(self):
     	   return self.enemy_units <= 0 or self.time > 100

def evaluate_state(self):
        	if self.enemy_units <= 0:
            	return 1  # Win
        	elif self.time > 100:
            	return -1  # Lose
        	return 0  # Ongoing

# Example usage of RTSGame
game = RTSGame()
while not game.is_game_over():
action = np.random.choice(["gather", "build", "attack"])
    game.simulate_action(action)
    print(f"Resources: {game.resources}, Units: {game.units}, Enemy Units: {game.enemy_units}")
