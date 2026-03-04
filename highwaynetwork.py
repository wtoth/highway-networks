import torch 
from torch import nn
import torch.nn.functional as F

class HighwayNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        pass

class ConvolutionalHighwayBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.h = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.t = nn.Conv2d(channels, channels, kernel_size, padding=padding)
    
    def forward(self, x):
        H = F.relu(self.h(x))
        T = F.sigmoid(self.t(x))
        return H*T + x *(1-T)

class LinearHighwayBlock(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.H = nn.Linear(size, size)
        self.T = nn.Linear(size, size)
    
    def forward(self, x):
        H = F.relu(self.H(x))
        T = F.sigmoid(self.T(x))
        return H*T + x *(1-T)
