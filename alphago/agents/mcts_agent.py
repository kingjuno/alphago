import math
import time

from alphago.env.gotypes import Player
from alphago.mcts import mcts

from . import FastRandomBot
from .base import Agent


class MCTSAgent(Agent):
    def __init__(self, num_rounds, temperature, timeout=None):
        Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature
        self.timeout = timeout

    def select_move(self, game_state):
        start_time = time.time()
        root = mcts.MCTSNode(game_state)
        for _ in range(self.num_rounds):
            node = root

            while node.fully_expanded() and not node.is_terminal():
                node = self.select_child(node)

            if not node.fully_expanded():
                """
                Expansion
                """
                node = node.add_random_child()

            winner = self.simulate(node.game_state)

            self.backpropagate(node, winner)
            if self.timeout is not None and time.time() - start_time > self.timeout:
                break
        print(time.time() - start_time)
        scored_moves = [
            (
                child.winning_pct(game_state.next_player),
                child.move,
                child.num_rollouts,
            )
            for child in root.children
        ]

        scored_moves.sort(key=lambda x: x[0], reverse=True)
        for s, m, n in scored_moves[:3]:
            print("%s - %s (%s)" % (m, s, n))

        best_move = scored_moves[0][1]
        return best_move

    def select_child(self, node):
        """
        Selection
        """
        total_rollouts = sum(child.num_rollouts for child in node.children)
        log_total_rollouts = math.log(total_rollouts)

        best_score = -1
        best_child = None
        for child in node.children:
            win_pct = child.winning_pct(node.game_state.next_player)
            exploration_factor = math.sqrt(log_total_rollouts / child.num_rollouts)
            uct_score = win_pct + self.temperature * exploration_factor
            if uct_score > best_score:
                best_score = uct_score
                best_child = child
        return best_child

    @staticmethod
    def simulate(game):
        """
        Simulation
        """
        bots = {
            Player.black: FastRandomBot(),
            Player.white: FastRandomBot(),
        }
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        return game.winner()

    def backpropagate(self, node, winner):
        """
        Backpropagate
        """
        while node is not None:
            node.num_rollouts += 1
            node.win_counts[winner] += 1
            node = node.parent
