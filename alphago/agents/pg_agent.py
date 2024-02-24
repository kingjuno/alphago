import numpy as np
import torch

from alphago.agents.add_ons import is_point_an_eye
from alphago.env import go_board

from . import Agent


def clip_prob(probs):
    min_p = 1e-5
    max_p = 1 - min_p
    clipped_probs = np.clip(probs, min_p, max_p)
    clipped_probs = clipped_probs / np.sum(clipped_probs)

    return clipped_probs


class PGAgent(Agent):
    def __init__(self, model, encoder, temperature=0.01, model_path=None):
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = encoder
        self.temperature = temperature
        self.buffer = None
        if model_path:
            self.model_path = model_path
            self.model.load_state_dict(torch.load(model_path))

    def set_buffer(self, buffer):
        self.buffer = buffer

    def select_move(self, game_state):
        self.model.eval()
        board_tensor = self.encoder.encode(game_state)
        X = torch.tensor(board_tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        move_probs, _ = self.model(X)
        num_moves = self.encoder.board_height * self.encoder.board_width
        if np.random.random() < self.temperature:
            move_probs = np.ones(num_moves) / num_moves
        else:
            move_probs = clip_prob(move_probs.cpu().detach().view(-1).numpy())
        candidates = np.arange(num_moves)
        indices = np.random.choice(
            candidates,
            num_moves,
            replace=False,
            p=move_probs,
        )
        ranked_moves = candidates[indices]
        for _point in ranked_moves:
            point = self.encoder.decode_point_index(_point)
            move = go_board.Move.play(point)
            is_valid = game_state.is_valid_move(move)
            is_an_eye = is_point_an_eye(game_state.board, point, game_state.next_player)
            if is_valid and (not is_an_eye):
                if self.buffer:
                    self.buffer.record_decision(state=board_tensor, action=_point)
                return go_board.Move.play(point)
        return go_board.Move.pass_turn()

    def train(self, cfg, exp):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr)
        criterion = torch.nn.CrossEntropyLoss()
        n = exp.states.shape[0]
        print(n)
        num_move = cfg.board_size**2
        y = np.zeros((n, num_move))
        for i in range(n):
            action = exp.actions[i]
            reward = exp.rewards[i]
            y[i][action] = reward

        states = torch.Tensor(exp.states).to(self.device)
        y = torch.Tensor(y).to(self.device)

        for epoch in range(cfg.epochs):
            # Shuffle the data indices
            print(f'epoch: {epoch+1}')
            indices = torch.randperm(states.size(0)).split(cfg.batch_size)

            for batch_indices in indices:
                # Get a batch of data
                states_batch = states[batch_indices]
                y_batch = y[batch_indices]

                optimizer.zero_grad()
                outputs, _ = self.model(states_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        return loss.item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
