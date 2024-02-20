import time

import torch
from torch import nn

from alphago import agents
from alphago.encoders.one_plane import OnePlaneEncoder
from alphago.env import go_board, gotypes
from alphago.env.utils import print_board, print_move

device = "cuda"
board_size = 19
game = go_board.GameState.new_game(board_size=board_size)


class GoNet(nn.Module):
    def __init__(self):
        super(GoNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 19 * 19, 128)
        self.fc2 = nn.Linear(128, 19 * 19)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 128 * 19 * 19)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


model = GoNet().to(device)
model.load_state_dict(torch.load("weights/go1.pth"))
model.eval()


bots = {
    gotypes.Player.black: agents.HumanAgent(),
    gotypes.Player.white: agents.HumanAgent(),
    # gotypes.Player.white: agents.RandomBot(),
    # gotypes.Player.white: agents.MCTSAgent(1000, 0.8),
}

for i in range(10):
    print_board(game.board)
    bot_move = bots[game.next_player].select_move(game)
    print_move(game.next_player, bot_move)
    game = game.apply_move(bot_move)

bots = {
    gotypes.Player.black: agents.HumanAgent(),
    gotypes.Player.white: agents.DLAgent(model, OnePlaneEncoder((19, 19))),
    # gotypes.Player.white: agents.RandomBot(),
    # gotypes.Player.white: agents.MCTSAgent(1000, 0.8),
}
while not game.is_over():
    time.sleep(0.3)
    # print(chr(27)+"[2J")
    print_board(game.board)
    bot_move = bots[game.next_player].select_move(game)
    print_move(game.next_player, bot_move)
    game = game.apply_move(bot_move)
print("------------")
print(game.winner())
