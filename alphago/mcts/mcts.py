import random

from alphago.env.gotypes import Player


class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.num_rollouts = 0
        self.unvisited_moves = game_state.legal_moves()
        self.win_counts = {
            Player.black: 0,
            Player.white: 0,
        }

    def add_random_child(self):
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        next_state = self.game_state.apply_move(new_move)
        child = MCTSNode(next_state, self, new_move)
        self.children.append(child)
        return child

    def fully_expanded(self):
        return len(self.unvisited_moves) == 0

    def is_terminal(self):
        return self.game_state.is_over()

    def winning_pct(self, player):
        return float(self.win_counts[player]) / float(self.num_rollouts)
