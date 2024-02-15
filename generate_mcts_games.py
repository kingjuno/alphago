import numpy as np

from alphago.agents import MCTSAgent
from alphago.encoders import OnePlaneEncoder
from alphago.env import go_board
from alphago.env.utils import print_board, print_move


def generate_game(board_size, rounds, temperature, max_moves=1000):
    boards, moves = [], []
    game = go_board.GameState.new_game(board_size)
    encoder = OnePlaneEncoder(board_size)
    bot = MCTSAgent(rounds, temperature)

    num_moves = 0
    while not game.is_over():
        print_board(game.board)
        move = bot.select_move(game)
        if move.is_play:
            boards.append(encoder.encode(game))
            move_one_hot = np.zeros(encoder.num_points())
            move_one_hot[encoder.encode_point(move.point)] = 1
            moves.append(move_one_hot)
        print_move(game.next_player, move)
        game = game.apply_move(move)
        num_moves += 1
        if num_moves > max_moves:
            break

    return np.array(boards), np.array(moves)


board_size = 9
rounds = 1000
temperature = 0.8
max_moves = 60
num_games = 10

_boards = []
_moves = []

dataset_loc = "datasets/mcts_games/"
for i in range(num_games):
    print("Generating game %d..." % i)
    boards, moves = generate_game(
        (board_size, board_size), rounds, temperature, max_moves
    )
    _boards.append(boards)
    _moves.append(moves)

np.savez(
    dataset_loc + "mcts_games_%dx%d.npz" % (board_size, num_games),
    boards=_boards,
    moves=_moves,
)
print("Saved %d games to %s" % (num_games, dataset_loc))
