import random

from alphago.env.go_board import Move
from alphago.env.gotypes import Point
from alphago.env.utils import point_from_coords

from .base import Agent

__all__ = ["HumanAgent"]


class HumanAgent(Agent):
    def select_move(self, game_state):
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
