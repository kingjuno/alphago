import numpy as np


def _OnePlaneEncoder(board, move, color):
    board_array = np.zeros((1, board.side, board.side))
    for p in board.list_occupied_points():
        board_array[0, p[1][0], p[1][1]] = -1.0 + 2 * int(p[0] == color)
    return board_array, move[0] * board.side + move[1]

def _SevenPlaneEncoder(board, move, color):
    raise NotImplementedError