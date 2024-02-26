import torch

from alphago.agents import ValueAgent
from alphago.encoders import OnePlaneEncoder
from alphago.networks import AlphaGoNet
from alphago.rl import ExperienceBuffer

BOARD_SIZE = 9
encoder = OnePlaneEncoder((BOARD_SIZE, BOARD_SIZE))
exp = ExperienceBuffer.load_experience("experiment/experience/exp.npz")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AlphaGoNet((1, BOARD_SIZE, BOARD_SIZE)).to(device)
agent = ValueAgent(model, encoder, 0.01, model_path="experiment/weights/rlgo1.pth")


class cfg:
    lr = 0.01
    epochs = 1
    board_size = BOARD_SIZE
    batch_size = 128


agent.train(cfg, exp)
agent.save_model("experiment/weights/valuego1.pth")
