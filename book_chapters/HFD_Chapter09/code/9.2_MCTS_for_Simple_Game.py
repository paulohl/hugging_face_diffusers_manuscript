# This example shows the practical use of MCTS to simulate, evaluate, and optimize moves in a simple game environment.
# By walking through this implementation, we can gain insight into how MCTS enables strategic foresight in game-playing AI.

class TicTacToe:
    	def __init__(self):
        self.board = [[None]*3 for _ in range(3)]
        self.player = 'X'

def move(self, x, y):
        if self.board[x][y] is None:
            self.board[x][y] = self.player
            self.player = 'O' if self.player == 'X' else 'X'
        return self

def is_winner(self, player):
        win_conditions = [
            [self.board[i][0] == player and self.board[i][1] == player and self.board[i][2] == player for i in range(3)],
            [self.board[0][i] == player and self.board[1][i] == player and self.board[2][i] == player for i in range(3)],
            [self.board[i][i] == player for i in range(3)],
            [self.board[i][2-i] == player for i in range(3)]
        ]
        return any(win_conditions)

def get_legal_moves(self):
        return [(x, y) for x in range(3) for y in range(3) if self.board[x][y] is None]

# This portion of the code defines the basic structure of the Tic-Tac-Toe game. 
# The TicTacToe class initializes the game board and manages the state of the game. 
# It includes methods to make a move (move), check for a winner (is_winner), and find available legal moves (get_legal_moves).

def mcts(root_state, iterations=1000): 
    root_node = MCTSNode(root_state)

    for _ in range(iterations):
        	node = root_node
           state = deepcopy(root_state)

# Selection
      while node.children:
           node = node.select_child()
           state = state.move(*node.move)

# Expansion
      if not state.is_winner('O') and not state.is_winner('X'):
      legal_moves = state.get_legal_moves()
           for move in legal_moves:
                new_state = deepcopy(state).move(*move)
                node.add_child(new_state, move)

# Simulation
while legal_moves:
            move = random.choice(legal_moves)
            state = state.move(*move)
            legal_moves = state.get_legal_moves()

# Backpropagation
      while node is not None:
            node.update(state.result(node.player))
            node = node.parent

return root_node.best_action()

# Example usage
game = TicTacToe()
mcts(game)

# This function integrates the four fundamental stages of MCTS.
