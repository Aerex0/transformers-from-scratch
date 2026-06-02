import torch
import torch.nn as nn
from config import D_MODEL

class FeedForward(nn.Module):
    def __init__(self, d_model=D_MODEL, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)  # First linear layer
        self.relu = nn.ReLU()                     # ReLU activation function
        self.linear2 = nn.Linear(d_ff, d_model)   # Second linear layer

    def forward(self, x):
        x = self.linear1(x)  # Apply first linear layer
        x = self.relu(x)     # Apply ReLU activation
        x = self.linear2(x)  # Apply second linear layer
        return x