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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
print(len(train_dataset))

class GoNet(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(GoNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 48, kernel_size=7),
            nn.ReLU(),
            nn.ZeroPad2d(padding=2),
            nn.Conv2d(48, 32, kernel_size=5),
            nn.ReLU(),
            nn.ZeroPad2d(padding=2),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.ReLU(),
            nn.ZeroPad2d(padding=2),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.dense_layers = nn.Sequential(
            nn.Linear(32 * 19 * 19, 512), nn.ReLU(), nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.dense_layers(x)
        return x


model = GoNet(1, 19 * 19).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

from tqdm import tqdm

for epoch in range(10):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
    for i, (inputs, labels) in loop:
        optimizer.zero_grad()
        inputs = inputs.float().to(device)
        labels = to_categorical(labels, 19 * 19).float().to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            loop.set_description(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.float().to(device)
            labels = to_categorical(labels, 19 * 19).float().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            _, _labels = torch.max(labels, 1)
            total += labels.size(0)
            correct += (predicted == _labels).sum().item()
    accuracy = 100 * correct / total
    tqdm.write(f"Epoch {epoch+1}/{10}, Accuracy: {accuracy:.2f}%")


torch.save(model.state_dict(), "weights/go1.pth")
