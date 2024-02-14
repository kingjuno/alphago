import time

from alphago import agents
from alphago.env import go_board, gotypes
from alphago.env.utils import print_board, print_move

board_size = 9
game = go_board.GameState.new_game(board_size=board_size)
bots = {
    gotypes.Player.black: agents.RandomBot(),
    # gotypes.Player.white: agents.RandomBot(),
    gotypes.Player.white: agents.MCTSAgent(1000, 0.8),
}

while not game.is_over():
    # time.sleep(0.3)
    # print(chr(27)+"[2J")
    # print_board(game.board)
    bot_move = bots[game.next_player].select_move(game)
    print_move(game.next_player, bot_move)
    game = game.apply_move(bot_move)
print('------------')
print(game.winner())