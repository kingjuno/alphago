import copy
import torch

from alphago import agents
from alphago import encoders
from alphago.env import go_board, gotypes
from alphago.env.scoring import compute_game_result
from alphago.env.utils import print_board, print_move
from alphago.networks import GoNet

device = "cuda"
board_size = 5
game = go_board.GameState.new_game(board_size=board_size)
encoder = encoders.OnePlaneEncoder((board_size, board_size))
# encoder = encoders.SimpleEncoder((board_size, board_size))

model = GoNet(board_size, 11).to(device)
model.eval()


AGENTS = [
    agents.HumanAgent(),
    agents.DLAgent(model, encoder),
    agents.FastRandomBot(),
    agents.MCTSAgent(2000, 0.8),
    agents.PGAgent(model, encoder)
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
    while not game.is_over():
        print_board(game.board)
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)
        game_result = compute_game_result(game)
    return game_result.winner


wins = {gotypes.Player.black: 0, gotypes.Player.white: 0}
for i in range(3):
    winner = simulate_game(AGENTS[3], AGENTS[0])
    wins[winner] += 1
    print(winner)
print(wins)
