import numpy as np
import torch
import torch.nn as nn

data = np.load("dataset/mcts_games/mcts_games.npz")
X, y = data["targets"], data["labels"]

random_indices = np.random.permutation(X.shape[0])
train_indices = random_indices[: int(0.8 * X.shape[0])]
test_indices = random_indices[int(0.8 * X.shape[0]) :]
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]
# convert to torch dataset
train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_train).float(), torch.tensor(y_train).long()
)
test_dataset = torch.utils.data.TensorDataset(
    torch.tensor(X_test).float(), torch.tensor(y_test).long()
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)


# 9x9 board
class GoNet(nn.Module):
    def __init__(self):
        super(GoNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 9 * 9, 128)
        self.fc2 = nn.Linear(128, 81)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 128 * 9 * 9)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


model = GoNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.float().to(device)
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
            labels = labels.float().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            print(predicted, labels)
            raise
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}")
