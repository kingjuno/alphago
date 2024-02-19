import gzip
import os
import random
import shutil
import tarfile

import numpy as np
from tqdm import tqdm

from alphago.data.dataset import GoDataSet
from alphago.data.sgf import *
from alphago.encoders.one_plane import OnePlaneEncoder
from alphago.env.go_board import Board, GameState, Move
from alphago.env.gotypes import Player, Point

board_size = 19
encoder = OnePlaneEncoder((board_size, board_size))


a = GoDataSet(encoder, game="cwi-minigo-9x9", seed=10)
print(a.list_all_datasets())
# dataset_loc = "dataset/kgs/kgs-19-2015"
# # dataset_loc = 'dataset/cwi-minigo-9x9'
# files = os.listdir(f"{dataset_loc}")
# board_size = 19
# encoder = OnePlaneEncoder((board_size, board_size))


# def get_handicap(sgf):
#     go_board = Board(board_size, board_size)
#     first_move_done = False
#     move = None
#     game_state = GameState.new_game(board_size)
#     if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
#         for setup in sgf.get_root().get_setup_stones():
#             for move in setup:
#                 row, col = move
#                 go_board.place_stone(
#                     Player.black, Point(row + 1, col + 1)
#                 )  # black gets handicap
#         first_move_done = True
#         game_state = GameState(go_board, Player.white, None, move)
#     return game_state, first_move_done


# features, labels = [], []
# counter = 0
# for file in tqdm(files[:1000]):
#     with open(f"{dataset_loc}/{file}", "r") as f:
#         game_string = "".join(f.readlines())
#     sgf = Sgf_game.from_string(game_string)

#     game_state, first_move_done = get_handicap(sgf)

#     for item in sgf.main_sequence_iter():
#         color, move_tuple = item.get_move()
#         point = None
#         if color is not None:
#             if move_tuple is not None:
#                 row, col = move_tuple
#                 point = Point(row + 1, col + 1)
#                 move = Move.play(point)
#             else:
#                 move = Move.pass_turn()
#             if first_move_done and point is not None:
#                 features.append(encoder.encode(game_state))
#                 labels.append(encoder.encode_point(point))
#                 counter += 1
#             game_state = game_state.apply_move(move)
#             first_move_done = True

# features = np.array(features)
# labels = np.array(labels)
# print(features.shape)
