import multiprocessing as mp

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


def generate_game_wrapper(args):
    board_size, rounds, temperature, max_moves = args
    return generate_game(board_size, rounds, temperature, max_moves)


board_size = 9
rounds = 1000
temperature = 0.8
max_moves = 60
num_games = 10
dataset_loc = "dataset/mcts_games_mt/"

# Define the arguments for each game
game_args = [
    ((board_size, board_size), rounds, temperature, max_moves) for _ in range(num_games)
]

# Use multiprocessing to generate games in parallel
pool = mp.Pool(mp.cpu_count())
results = pool.map(generate_game_wrapper, game_args)
pool.close()
pool.join()

_boards, _moves = zip(*results)


np.savez(
    dataset_loc + "mcts_games_mt_%dx%d.npz" % (board_size, num_games),
    targets=np.concatenate(_boards, axis=0),
    labels=np.concatenate(_moves, axis=0),
)
