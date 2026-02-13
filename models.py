
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1=128, hidden_dim2=64):
        super(MLP, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        # Second hidden layer
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        # Output layer
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # Flatten the input if it's not already 1D (e.g., for image data)
        if x.dim() > 2: # Check if input has more than 2 dimensions (batch_size, features)
            x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self, input_channels, image_size, output_classes):
        super(CNN, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dynamically calculate the flattened size after convolutional and pooling layers
        # We'll pass a dummy tensor through the conv/pool layers to get the output shape
        dummy_input = torch.randn(1, input_channels, image_size, image_size)
        x = self.pool1(self.relu1(self.conv1(dummy_input)))
        x = self.pool2(self.relu2(self.conv2(x)))

        flattened_size = x.view(x.size(0), -1).size(1)

        # Fully connected layers (classification head)
        self.fc1 = nn.Linear(flattened_size, 128) # First hidden layer
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, output_classes) # Output layer

    def forward(self, x):
        # Pass through first conv block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Pass through second conv block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1) # Flatten starting from dimension 1 (batch is dim 0)

        # Pass through fully connected layers
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x
