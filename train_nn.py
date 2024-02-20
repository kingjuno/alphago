import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from alphago.data.dataset import GoDataSet

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

BOARD_SIZE = 19
training_dataset = GoDataSet(encoder="oneplane", no_of_games=3000)
test_dataset = GoDataSet(encoder="oneplane", no_of_games=100, avoid=training_dataset.games)
train_loader = torch.utils.data.DataLoader(
    training_dataset, batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=9, stride=1, padding=4
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=7, stride=1, padding=3
        )
        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.fc1 = nn.Linear(
            in_features=32 * BOARD_SIZE * BOARD_SIZE,
            out_features=BOARD_SIZE * BOARD_SIZE,
        )
        self.fc2 = nn.Linear(
            in_features=BOARD_SIZE * BOARD_SIZE, out_features=BOARD_SIZE * BOARD_SIZE
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 32 * BOARD_SIZE * BOARD_SIZE)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        x = F.log_softmax(x, dim=1)
        return x


def train(model, device, train_loader, optimizer, epoch):
    losses = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if batch_idx % 1000 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
    return losses


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()
            pred = output.argmax(
                dim=1, keepdim=True
            )
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return float(correct) / len(test_loader.dataset)


model = CNN()
if torch.cuda.is_available():
    model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
losses = []
accuracies = []
for epoch in range(0, 200):
    losses.extend(train(model, device, train_loader, optimizer, epoch))
    accuracies.append(test(model, device, train_loader))

torch.save(model.state_dict(), "weights/go1.pth")
