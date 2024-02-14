import random

from alphago.env.go_board import Move
from alphago.env.gotypes import Point
from alphago.env.utils import point_from_coords

from .base import Agent

__all__ = ["HumanAgent"]


class HumanAgent(Agent):
    def select_move(self, game_state):
        """Choose a random valid move that preserves our own eyes."""
        # candidates = []
        # for r in range(1, game_state.board.num_rows + 1):
        #     for c in range(1, game_state.board.num_cols + 1):
        #         candidate = Point(row=r, col=c)
        #         if game_state.is_valid_move(
        #             Move.play(candidate)
        #         ) and not is_point_an_eye(
        #             game_state.board, candidate, game_state.next_player
        #         ):
        #             candidates.append(candidate)
        # if not candidates:
        #     return Move.pass_turn()
        # return Move.play(random.choice(candidates))
        while True:
            movement = input("Enter your move(point/pass/resign)> ")
            if movement == "pass":
                return Move.pass_turn()
            elif movement == "resign":
                return Move.resign()
            else:
                point = point_from_coords(movement.strip())
                if game_state.is_valid_move(Move.play(point)):
                    return Move.play(point)
                else:
                    print("Invalid move")
