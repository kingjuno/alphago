import torch

from alphago.agents import PGAgent
from alphago.encoders import OnePlaneEncoder
from alphago.networks import AlphaGoNet
from alphago.rl import ExperienceBuffer, experience_simulation

BOARD_SIZE = 9
model = AlphaGoNet((1, BOARD_SIZE, BOARD_SIZE))
if torch.cuda.is_available():
    model = model.cuda()
exp_path = "experiment/experience/exp.npz"


class cfg:
    lr = 0.00001
    epochs = 10
    board_size = BOARD_SIZE
    batch_size = 128


encoder = OnePlaneEncoder((BOARD_SIZE, BOARD_SIZE))

agent1 = PGAgent(model, encoder, model_path="experiment/weights/go2.pth")
agent2 = PGAgent(model, encoder, model_path="experiment/weights/go2.pth")

if exp_path == None:
    num_games = 10000
    exp = experience_simulation(num_games, agent1, agent2)
    exp.save_experience("experiment/experience/exp.npz")
else:
    exp = ExperienceBuffer.load_experience("experiment/experience/exp.npz")


loss = agent1.train(cfg, exp)
print(loss)
agent1.save_model("experiment/weights/rlgo1.pth")
