import torch

from alphago.agents.add_ons import is_point_an_eye
from alphago.env import go_board

from . import Agent


def clip_probs(probs):
    min_p = 1e-5
    max_p = 1 - min_p
    clipped_probs = torch.clamp(probs, min_p, max_p)
    clipped_probs /= torch.sum(clipped_probs)
    return clipped_probs


class PGAgent(Agent):
    def __init__(self, model, encoder, model_path=None):
        self.model = model
        self.encoder = encoder
        self.buffer = None
        if model_path:
            self.model_path = model_path
            self.model.load_state_dict(torch.load(model_path))

    def set_buffer(self, buffer):
        self.buffer = buffer

    def select_move(self, game_state):
        board_tensor = self.encoder.encode(game_state)
        X = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0)
        move_probs  = self.model(X)
        move_probs = clip_probs(move_probs)
        num_moves = self.encoder.board_height * self.encoder.board_width
        candidates = torch.arange(num_moves)
        indices = torch.multinomial(move_probs, num_moves, replacement=False)
        ranked_moves = candidates[indices]
        import numpy
        print(move_probs)
        cd1 = numpy.arange(num_moves)
        ind = numpy.random.choice(
            cd1, num_moves,
            replace=False, p=move_probs.view(-1).detach()
        )
        rk = candidates[ind]
        print(ranked_moves, rk)
        for _point in ranked_moves:
            print(_point)
            point = self.encoder.decode_point_index(_point)
            print(point)
            raise
            move = go_board.Move.play(point)
            is_valid = game_state.is_valid_move(move)
            is_an_eye = is_point_an_eye(game_state.board, point, game_state.next_player)
            if is_valid and (not is_an_eye):
                if self.buffer:
                    self.buffer.store(state=board_tensor, action=_point)
                return go_board.Move.play(point)
        return go_board.Move.pass_turn()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
