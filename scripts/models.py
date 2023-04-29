import torch.nn as nn
import torch.nn.functional as F

# Convolutional Neural Network (CNN) that consists of 3 convolutional layers and ReLU activation function in each layer
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=1, padding=1)

    # Forward method that defines the computation performed by the model
    def forward(self, input):
        # First convolutional layer with ReLU activation
        input = F.relu(self.conv1(input))

        # Second convolutional layer with ReLU activation
        input = F.relu(self.conv2(input))

        # Third convolutional layer with ReLU activation
        input = F.relu(self.conv3(input))

        return input

# Class for a simple Autoencoder (AE) that consists of an encoder and a decoder
class SimpleAutoEncoder(nn.Module):
    def __init__(self):
        super(SimpleAutoEncoder, self).__init__()

        # Encoder consisting of 2 convolutional layers with ReLU activation
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.ReLU(True))

        # Decoder consisting of 2 transposed convolutional layers with ReLU activation
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(64, 32, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=5),
            nn.ReLU(True))

    # Forward method that defines the computation performed by the model
    def forward(self, input):
        # Encoding step
        input = self.encoder(input)

        # Decoding step
        input = self.decoder(input)
        return input