import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.multiprocessing import freeze_support


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
input_size = 28 * 28
hidden_size = 3
output_size = 10 # Ther Number of MNIST Dataset Classes
learning_rate = 0.001
batch_size = 4
epochs = 5


def main():
    # Load MNIST dataset
    transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    # Initialize the model, loss function, and optimizer
    model = MLP(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.view(-1, input_size)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}')

    # Save the trained model if needed
    torch.save(model.state_dict(), 'mlp_model.pth')


if __name__ == '__main__':
    # Add the freeze_support() call for Windows
    freeze_support()
    
    # Call the main function
    main()
