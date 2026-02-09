import torch
import torch.nn as nn
import torch.nn.functional as F


class SetMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SetMLP, self).__init__()

        self.flatten = nn.Flatten()

        # First hidden layer
        self.fc1 = nn.Linear(input_size, 4000)
        self.bn1 = nn.BatchNorm1d(4000)
        self.srelu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        # Second hidden layer
        self.fc2 = nn.Linear(4000, 1000)
        self.srelu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        # Third hidden layer
        self.fc3 = nn.Linear(1000, 4000)
        self.srelu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)

        # Output layer
        self.fc4 = nn.Linear(4000, num_classes)

    def forward(self, x):
        x = self.flatten(x)

        # First layer operations
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.srelu1(x)
        x = self.dropout1(x)

        # Second layer operations
        x = self.fc2(x)
        x = self.srelu2(x)
        x = self.dropout2(x)

        # Third layer operations
        x = self.fc3(x)
        x = self.srelu3(x)
        x = self.dropout3(x)

        # Output layer
        x = self.fc4(x)
        x = F.softmax(x, dim=1)

        return x