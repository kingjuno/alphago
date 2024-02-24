import torch
import torch.nn as nn
import torch.nn.functional as F


class AlphaGoNet(nn.Module):
    def __init__(
        self,
        input_shape,
        num_filters=192,
        dropout = 0.3,

    ):
        super(AlphaGoNet, self).__init__()
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.dropout = dropout

        self.conv1 = nn.Conv2d(input_shape[0], num_filters, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_filters, num_filters, 3, stride=1)
        self.conv4 = nn.Conv2d(num_filters, num_filters, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(num_filters)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.bn3 = nn.BatchNorm2d(num_filters)
        self.bn4 = nn.BatchNorm2d(num_filters)

        self.fc1 = nn.Linear(
            num_filters * (self.input_shape[1] - 4) * (self.input_shape[2] - 4), 1024
        )
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.input_shape[1]*self.input_shape[2])

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        s = s.view(-1, 1, self.input_shape[1], self.input_shape[2])
        s = F.relu(self.bn1(self.conv1(s)))
        s = F.relu(self.bn2(self.conv2(s)))
        s = F.relu(self.bn3(self.conv3(s)))
        s = F.relu(self.bn4(self.conv4(s)))
        s = s.view(
            -1, self.num_filters * (self.input_shape[1] - 4) * (self.input_shape[2] - 4)
        )

        s = F.dropout(
            F.relu(self.fc_bn1(self.fc1(s))),
            p=self.dropout,
            training=self.training,
        )
        s = F.dropout(
            F.relu(self.fc_bn2(self.fc2(s))),
            p=self.dropout,
            training=self.training,
        )

        pi = self.fc3(s)
        v = self.fc4(s)
        if self.training:
            return F.log_softmax(pi, dim=1), torch.tanh(v)
        else:
            return F.softmax(pi, dim=1), torch.tanh(v)