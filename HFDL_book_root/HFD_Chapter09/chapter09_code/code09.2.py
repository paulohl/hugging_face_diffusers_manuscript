# Using the classic game Tic-Tac-Toe to illustrate how MCTS can be applied in decision-making scenarios:


import random
from copy import deepcopy
import math


# -----------------------------
# Game Environment: TicTacToe
# -----------------------------
class TicTacToe:
    def __init__(self):
        self.board = [[None] * 3 for _ in range(3)]
        self.player = 'X'  # X always starts

    def move(self, x, y):
        """Place a mark at (x, y) if legal and switch players."""
        if self.board[x][y] is None:
            self.board[x][y] = self.player
            self.player = 'O' if self.player == 'X' else 'X'
        return self

    def is_winner(self, player):
        """Check whether the given player has a winning board."""
        rows = any(all(self.board[r][c] == player for c in range(3)) for r in range(3))
        cols = any(all(self.board[r][c] == player for r in range(3)) for c in range(3))
        diag = all(self.board[i][i] == player for i in range(3))
        anti = all(self.board[i][2 - i] == player for i in range(3))
        return rows or cols or diag or anti

    def get_legal_moves(self):
        """Return a list of all empty board positions."""
        return [(x, y) for x in range(3) for y in range(3) if self.board[x][y] is None]

    def is_terminal(self):
        """Determine whether the game is over."""
        return (
            self.is_winner('X') or
            self.is_winner('O') or
            len(self.get_legal_moves()) == 0
        )

    def result(self, player):
        """Return +1 if player wins, -1 if opponent wins, 0 otherwise."""
        opponent = 'O' if player == 'X' else 'X'
        if self.is_winner(player):
            return 1
        elif self.is_winner(opponent):
            return -1
        return 0


# -----------------------------
# MCTS Node
# -----------------------------
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0

    def ucb1(self, child):
        if child.visits == 0:
            return float('inf')
        return (child.wins / child.visits) + math.sqrt(2 * math.log(self.visits) / child.visits)

    def select_child(self):
        """Select child with highest UCB1 score."""
        return max(self.children, key=lambda c: self.ucb1(c))

    def expand(self):
        """Add all legal moves as child nodes."""
        for move in self.state.get_legal_moves():
            new_state = deepcopy(self.state).move(*move)
            self.children.append(MCTSNode(new_state, parent=self, move=move))

    def update(self, result):
        self.visits += 1
        self.wins += result


# -----------------------------
# Core MCTS Algorithm
# -----------------------------
def mcts(root_state, iterations=1000):
    root_node = MCTSNode(root_state)

    for _ in range(iterations):
        node = root_node
        state = deepcopy(root_state)

        # --- Selection ---
        while node.children and not state.is_terminal():
            node = node.select_child()
            state = state.move(*node.move)

        # --- Expansion ---
        if not state.is_terminal():
            node.expand()

        # If expansion created children, pick one at random to simulate
        if node.children:
            node = random.choice(node.children)
            state = deepcopy(node.state)
