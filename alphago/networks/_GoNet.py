import torch.nn as nn
import torch.nn.functional as F


class GoNet(nn.Module):
    def __init__(self, board_size=19, in_channel=1):
        super(GoNet, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(
            in_channels=in_channel, out_channels=32, kernel_size=9, stride=1, padding=4
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.fc1 = nn.Linear(
            in_features=32 * self.board_size * self.board_size,
            out_features=self.board_size * self.board_size,
        )
        self.fc2 = nn.Linear(
            in_features=self.board_size * self.board_size,
            out_features=self.board_size * self.board_size,
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * self.board_size * self.board_size)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        x = F.log_softmax(x, dim=1)
        return x
