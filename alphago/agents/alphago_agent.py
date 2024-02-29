import time

import numpy as np

from alphago.agents.add_ons import is_point_an_eye
from alphago.agents.base import Agent
from alphago.env.go_board import Move
from alphago.env.utils import print_board


class AlphaGoNode:
    def __init__(self, parent=None, probability=1.0):
        self.parent = parent
        self.children = {}

        self.visit_count = 0
        self.q_value = 0
        self.prior_value = probability
        self.u_value = probability

    def select_child(self):
        return max(
            self.children.items(), key=lambda child: child[1].q_value + child[1].u_value
        )

    def expand_children(self, moves, probabilities):
        for move, prob in zip(moves, probabilities):
            if move not in self.children:
                self.children[move] = AlphaGoNode(parent=self, probability=prob)

    def update_values(self, leaf_value):
        if self.parent is not None:
            self.parent.update_values(leaf_value)

        self.visit_count += 1
        self.q_value += leaf_value / self.visit_count
        if self.parent is not None:
            c_u = 5
            self.u_value = (
                c_u
                * np.sqrt(self.parent.visit_count)
                * self.prior_value
                / (1 + self.visit_count)
            )


class AlphaGoMCTS(Agent):
    def __init__(
        self,
        policy_agent,
        fast_policy_agent,
        value_agent,
        lambda_value=0.5,
        num_simulations=60,
        depth=20,
        rollout_limit=40,
    ):
        self.policy = policy_agent
        self.rollout_policy = fast_policy_agent
        self.value = value_agent

        self.lambda_value = lambda_value
        self.num_simulations = num_simulations
        self.depth = depth
        self.rollout_limit = rollout_limit
        self.root = AlphaGoNode()

    def select_move(self, game_state):
        start = time.time()
        for sim in range(self.num_simulations):
            current_state = game_state
            node = self.root
            for depth in range(self.depth):
                if not node.children:
                    if current_state.is_over():
                        break
                    moves, probabilities = self.policy_probabilities(current_state)
                    node.expand_children(moves, probabilities)
                move, node = node.select_child()
                current_state = current_state.apply_move(move)
            value = self.value.predict(current_state)
            rollout = self.policy_rollout(current_state)

            weighted_value = (
                1 - self.lambda_value
            ) * value + self.lambda_value * rollout

            node.update_values(weighted_value)

        moves = sorted(
            self.root.children,
            key=lambda move: self.root.children.get(move).visit_count,
            reverse=True,
        )

        for i in moves:
            if not is_point_an_eye(game_state.board, i.point, game_state.next_player):
                move = i
                break
        else:
            move = Move.pass_turn()
        self.root = AlphaGoNode()
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        # print(time.time() - start)
        return move

    def policy_probabilities(self, game_state):
        encoder = self.policy.encoder
        outputs = self.policy.predict(game_state)[0]
        legal_moves = game_state.legal_moves()
        if len(legal_moves) == 2:
            return legal_moves, [1, 0]
        encoded_points = [
            encoder.encode_point(move.point) for move in legal_moves if move.point
        ]
        legal_outputs = outputs[encoded_points]
        normalized_outputs = legal_outputs / np.sum(legal_outputs)
        return legal_moves, normalized_outputs

    def policy_rollout(self, game_state):
        for step in range(self.rollout_limit):
            if game_state.is_over():
                break
            move_probabilities = self.rollout_policy.predict(game_state)[0]
            encoder = self.rollout_policy.encoder
            for idx in np.argsort(move_probabilities)[::-1]:
                max_point = encoder.decode_point_index(idx)
                greedy_move = Move(max_point)
                if greedy_move in game_state.legal_moves():
                    game_state = game_state.apply_move(greedy_move)
                    break

        next_player = game_state.next_player
        winner = game_state.winner()

        if winner is not None:
            return 1 if winner == next_player else -1
        else:
            return 0
