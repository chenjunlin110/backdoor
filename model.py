import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTModel(nn.Module):
    """Improved CNN model for MNIST classification with larger architecture"""

    def __init__(self, seed=42):
        super(MNISTModel, self).__init__()
        # Set random seed for reproducibility
        torch.manual_seed(seed)

        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate the size after convolutions and pooling
        # Input: 28x28 -> Conv1+Pool -> 14x14 -> Conv2+Pool -> 7x7 -> Conv3+Pool -> 3x3
        # So the flattened size will be 64 * 3 * 3 = 576

        # Fully connected layers
        self.fc1 = nn.Linear(576, 256)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(128, 10)

        # Save initial weights
        self.initial_weights = None
        self.save_initial_weights()

    def forward(self, x):
        # First block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Second block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Third block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flatten and feed to fully connected layers
        x = x.view(x.size(0), -1)  # Flatten

        # Debug print to check the shape
        # print(f"Shape after flattening: {x.shape}")

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

    def save_initial_weights(self):
        """Save model's initial weights"""
        self.initial_weights = {k: v.clone() for k, v in self.state_dict().items()}


    def get_weights(self):
        """Get model weights"""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        """Set model weights"""
        self.load_state_dict(weights)
