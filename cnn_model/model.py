import torch
import torch.nn as nn
import torchvision.models as models


# Define the MLP model
class ModifiedCNN(nn.Module):
    def __init__(self, architecture="vgg16", input_channels=3, hidden_size=512, output_size=10, img_size=32):
        super(ModifiedCNN, self).__init__()

        # Load pre-trained AlexNet model
        if architecture == "vgg16":
            self.model = models.vgg16(pretrained=True)
        elif architecture == "alexnet":
            self.model = models.alexnet(pretrained=True)
        else:
            raise ValueError("Unsupported architecture. Choose 'vgg16' or 'alexnet'.")

        # Modify the first convolutional layer to accept 1 channel for grayscale images
        if input_channels == 1:
            self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, img_size, img_size)
            flatten_size = self._get_flattened_size(dummy_input)

        # Modify the classifier (fully connected layers)
        self.fc1 = nn.Linear(flatten_size, hidden_size)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(hidden_size, output_size)

    def _get_flattened_size(self, x):
        # This function calculates the size of the flattened layer after passing through the convolutional layers
        x = self.model.features(x)
        x = torch.flatten(x, 1)
        return x.size(1)

    def forward(self, x):
        # Pass through the convolutional layers
        x = self.model.features(x)

        # Flatten the output from convolution layers
        x = torch.flatten(x, 1)

        # Pass through the custom fully connected layers
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout2(x)

        x = self.fc3(x)  # Final output layer
        return x
