from alphago.env.go_board import Board, GameState
from alphago.env.gotypes import Player, Point


def get_handicap(sgf):
    board_size = sgf.size
    go_board = Board(board_size, board_size)
    first_move_done = False
    move = None
    game_state = GameState.new_game(board_size)
    if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
        for setup in sgf.get_root().get_setup_stones():
            for move in setup:
                row, col = move
                go_board.place_stone(
                    Player.black, Point(row + 1, col + 1)
                )  # black gets handicap
        first_move_done = True
        game_state = GameState(go_board, Player.white, None, move)
    return game_state, first_move_done
