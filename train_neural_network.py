import torch
import torch.nn as nn

from alphago.data.dataset import GoDataSet
from alphago.encoders.one_plane import OnePlaneEncoder


def to_categorical(y, num_classes):
    """1-hot encodes a tensor"""
    return torch.eye(num_classes)[y]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = OnePlaneEncoder((19, 19))

train_dataset = GoDataSet(encoder, no_of_games=100)
test_dataset = GoDataSet(encoder, no_of_games=100, avoid=train_dataset.games)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

print(len(test_dataset))


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
        # x = self.softmax(x)
        return x


model = GoNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

for epoch in range(10):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.float().to(device)
        labels = to_categorical(labels, 19*19).float().to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.float().to(device)
            labels = to_categorical(labels, 19*19).float().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            _, _labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == _labels).sum().item()
            # print(predicted==_labels)
    print(f"Accuracy: {100 * correct / total}")

torch.save(model.state_dict(), "weights/go1.pth")
