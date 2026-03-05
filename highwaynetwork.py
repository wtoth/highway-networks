import torch 
from torch import nn
import torch.nn.functional as F

class HighwayNetwork(nn.Module):
    def __init__(self, num_highway_layers=50, channels=64, classes=10):
        super().__init__()
        
        # creates a consistent number of channels to be used in the highway layers
        self.input = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.highway_layers = self._build_highway_layers(num_layers=num_highway_layers, channels=channels)
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, classes)
        ) 

    def forward(self, x):
        x = self.input(x)
        x = self.highway_layers(x)
        x = self.output(x)
        return x

    def _build_highway_layers(self, num_layers, channels, kernel_size=3):
        layers = [ConvolutionalHighwayBlock(channels=channels, kernel_size=kernel_size) for _ in range(num_layers)]
        return nn.Sequential(*layers)

class ConvolutionalHighwayBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.h = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.t = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        nn.init.normal_(self.t.bias, mean=-2, std=0.1) # per paper they set the bias to a negative value
    
    def forward(self, x):
        H = F.relu(self.h(x))
        T = torch.sigmoid(self.t(x))
        return H*T + x *(1-T) # !!! This is element wise multiplication not dot products 
        # so the [0,1] bounding principle holds

class LinearHighwayBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.H = nn.Linear(size, size)
        self.T = nn.Linear(size, size)
        nn.init.normal_(self.T.bias, mean=-2, std=0.1) # per paper they set the bias to a negative value
    
    def forward(self, x):
        H = F.relu(self.H(x))
        T = torch.sigmoid(self.T(x))
        return H*T + x *(1-T) # !!! This is element wise multiplication not dot products 
        # so the [0,1] bounding principle holds