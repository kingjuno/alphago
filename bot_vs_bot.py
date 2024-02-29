import copy

import torch

from alphago import agents, encoders
from alphago.env import go_board, gotypes
from alphago.env.scoring import compute_game_result
from alphago.env.utils import print_board, print_move
from alphago.networks import AlphaGoNet, GoNet

device = "cuda"
board_size = 9
game = go_board.GameState.new_game(board_size=board_size)
encoder = encoders.OnePlaneEncoder((board_size, board_size))

model = AlphaGoNet((1, board_size, board_size))
model.load_state_dict(torch.load("experiment/weights/go2.pth"))
model = model.cuda()
model.eval()


model1 = AlphaGoNet((1, board_size, board_size))
model1.load_state_dict(torch.load("experiment/weights/rlgo1.pth"))
model1 = model1.cuda()
model1.eval()


model2 = AlphaGoNet((1, board_size, board_size))
model2.load_state_dict(torch.load("experiment/weights/valuego1.pth"))
model2 = model2.cuda()
model2.eval()


AGENTS = [
    agents.HumanAgent(), #0
    agents.DLAgent(model, encoder), #1
    agents.FastRandomBot(), #2
    agents.MCTSAgent(1000, 0.8), #3
    agents.PGAgent(model1, encoder, 0), #4
    agents.ValueAgent(model2, encoder, 0), #5
    agents.AlphaGoMCTS(
        agents.DLAgent(model, encoder),
        # agents.PGAgent(model1, encoder, 0),
        agents.PGAgent(model1, encoder, 0.1),
        agents.ValueAgent(model2, encoder, 0.1),
    )
]


def simulate_game(black_player, white_player):
    black_player = copy.deepcopy(
        black_player
    )  # ensure both players are different objects
    game = go_board.GameState.new_game(board_size=board_size)
    agents = {
        gotypes.Player.black: black_player,
        gotypes.Player.white: white_player,
    }
    i = 0
    while not game.is_over():
        i += 1
        # print_board(game.board)
        boards.append(encoder.encode(game))
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)
        game_result = compute_game_result(game)
        # print(game_result)
    print(i)
    return game_result.winner

boards = []
wins = {gotypes.Player.black: 0, gotypes.Player.white: 0}
for i in range(1):
    winner = simulate_game(AGENTS[2], copy.deepcopy(AGENTS[-1]))
    wins[winner] += 1
    print(wins)
print(wins)

import json
boards = torch.tensor(boards)
boards[::2] *= -1
boards = boards.tolist()
json_data = json.dumps(boards)
with open('AlphavsMCTS.json', 'w') as f:
    f.write(json_data)