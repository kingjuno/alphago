import numpy as np
import torch

from alphago.agents.add_ons import is_point_an_eye
from alphago.env.go_board import Move

from . import Agent


class ValueAgent(Agent):
    def __init__(
        self, model, encoder, temperature=0, policy="eps-greedy", model_path=None
    ):
        super().__init__()
        self.model = model
        self.encoder = encoder
        self.buffer = None
        self.temperature = temperature
        self.policy = policy
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_path:
            self.model_path = model_path
            self.model.load_state_dict(torch.load(model_path))

    def predict(self, game_state):
        encoded_state = np.array(self.encoder.encode(game_state))
        board = torch.tensor(encoded_state, requires_grad=False).cuda().float()
        return self.model(board)[1].detach().cpu().numpy()

    def set_buffer(self, buffer):
        self.buffer = buffer

    def select_move(self, game_state):
        self.model.eval()
        moves = []
        board_tensors = []
        for move in game_state.legal_moves():
            if not move.is_play:
                continue

            board_tensor = self.encoder.encode(game_state.apply_move(move))
            moves.append(move)
            board_tensors.append(board_tensor)
        if not moves:
            return Move.pass_turn()
        board_tensors = torch.tensor(np.array(board_tensors), dtype=torch.float32).to(self.device)

        # opponent value
        opp_val = self.model(board_tensors)[1].reshape(len(moves))
        val = 1 - opp_val

        if self.policy == "eps-greedy":
            if torch.rand(1) < self.temperature:
                val = torch.rand(val.shape)
            ranked_moves = torch.argsort(val)
            ranked_moves = torch.flip(ranked_moves, [0])
        elif self.policy == "weighted":
            p = val / torch.sum(val)
            p = torch.pow(p, 1.0 / self.temperature)
            p = p / torch.sum(p)
            ranked_moves = np.random.choice(
                np.arange(0, len(val)), size=len(val), p=p.tolist(), replace=False
            )
        else:
            raise NotImplementedError

        for move_idx in ranked_moves:
            move = moves[move_idx]
            if not is_point_an_eye(
                game_state.board, move.point, game_state.next_player
            ):
                if self.buffer:
                    self.buffer.record_decision(
                        state=board_tensor,
                        action=self.encoder.encode_point(move.point),
                    )
                return move
        return Move.pass_turn()

    def train(self, cfg, exp):
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr)
        criterion = torch.nn.MSELoss()
        n = exp.states.shape[0]

        y = torch.zeros((n)).to(self.device)
        for i in range(n):
            reward = exp.rewards[i]
            y[i] = 1 if reward > 0 else 0

        states = torch.Tensor(exp.states).to(self.device)
        for epoch in range(cfg.epochs):
            # Shuffle the data indices
            print(f"epoch: {epoch+1}")
            indices = torch.randperm(states.size(0)).split(cfg.batch_size)

            for batch_indices in indices:
                # Get a batch of data
                states_batch = states[batch_indices]
                y_batch = y[batch_indices]
                optimizer.zero_grad()
                _, outputs = self.model(states_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()

        return loss.item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
