import numpy as np
import torch

from alphago import encoders
from alphago.env import go_board
from alphago.env.go_board import Move
from alphago.env.gotypes import Point

from . import Agent
from .add_ons import is_point_an_eye


class DLAgent(Agent):
    def __init__(self, model, encoder):
        Agent.__init__(self)
        self.model = model
        self.encoder = encoder

    def predict(self, game_state):
        encoded_state = self.encoder.encode(game_state)
        print(encoded_state.shape)
        board = torch.tensor([encoded_state], requires_grad=False).cuda().float()
        return self.model(board).detach().cpu().numpy()

    def select_move(self, game_state):
        num_moves = self.encoder.board_width * self.encoder.board_height
        move_probs = self.predict(game_state)[0]
        move_probs **= 3
        eps = 1e-6
        move_probs = np.clip(move_probs, eps, 1 - eps)
        move_probs /= sum(move_probs)
        moves = np.arange(num_moves)
        ranked_moves = np.random.choice(moves, num_moves, replace=False, p=move_probs)
        for point_idx in ranked_moves:
            point = self.encoder.decode_point_index(point_idx)
            candidate = Point(*point)
            if game_state.is_valid_move(Move.play(candidate)) and not is_point_an_eye(
                game_state.board, candidate, game_state.next_player
            ):
                return Move.play(candidate)
        return Move.pass_turn()
