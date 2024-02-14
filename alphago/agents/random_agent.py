import random

import numpy as np

from alphago.env.go_board import Move
from alphago.env.gotypes import Point

from .add_ons import is_point_an_eye
from .base import Agent

__all__ = ["RandomBot"]


class RandomBot(Agent):
    def select_move(self, game_state):
        """Choose a random valid move that preserves our own eyes."""
        candidates = []
        for r in range(1, game_state.board.num_rows + 1):
            for c in range(1, game_state.board.num_cols + 1):
                candidate = Point(row=r, col=c)
                if game_state.is_valid_move(
                    Move.play(candidate)
                ) and not is_point_an_eye(
                    game_state.board, candidate, game_state.next_player
                ):
                    candidates.append(candidate)
        if not candidates:
            return Move.pass_turn()
        return Move.play(random.choice(candidates))


class FastRandomBot(Agent):
    def __init__(self):
        self.candidate_cache = None

    def _update_cache(self, game_state):
        self.candidate_cache = []
        for r in range(1, game_state.board.num_rows + 1):
            for c in range(1, game_state.board.num_cols + 1):
                candidate = Point(row=r, col=c)
                self.candidate_cache.append(candidate)

    def select_move(self, game_state):
        if self.candidate_cache is None:
            self._update_cache(game_state)
        np.random.shuffle(self.candidate_cache)
        for candidate in self.candidate_cache:
            if game_state.is_valid_move(Move.play(candidate)) and not is_point_an_eye(
                game_state.board, candidate, game_state.next_player
            ):
                return Move.play(candidate)
        return Move.pass_turn()
