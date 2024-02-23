# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import matplotlib.pyplot as plt
import torch

from alphago.data.dataset import GoDataSet
from alphago.networks import ResNet

BOARD_SIZE = 19
training_dataset = GoDataSet(encoder="oneplane", no_of_games=1)
train_loader = torch.utils.data.DataLoader(
    training_dataset, batch_size=64, shuffle=True
)
model = ResNet(BOARD_SIZE)
x, y = next(iter(train_loader))
policy, value = model(x[0].unsqueeze(0))
print(policy.shape)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
print(policy.shape)
plt.bar(range(19*19+1), policy)
plt.show()
