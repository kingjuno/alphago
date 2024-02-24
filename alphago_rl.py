import torch

from alphago.agents import PGAgent
from alphago.encoders import OnePlaneEncoder
from alphago.networks import AlphaGoNet
from alphago.rl import experience_simulation

BOARD_SIZE = 9
model = AlphaGoNet((1, BOARD_SIZE, BOARD_SIZE))
if torch.cuda.is_available():
    model = model.cuda()


class cfg:
    lr = 0.01
    epochs = 1
    board_size = BOARD_SIZE
    batch_size = 64


encoder = OnePlaneEncoder((BOARD_SIZE, BOARD_SIZE))

agent1 = PGAgent(model, encoder, "experiment/weights/go2.pth")
agent2 = PGAgent(model, encoder, "experiment/weights/go2.pth")

num_games = 1000
exp = experience_simulation(num_games, agent1, agent2)

loss = agent1.train(cfg, exp)

exp.save_experience("experiment/experience/exp.npz")
agent1.save_model("experiment/weights/rlgo1.pth")
