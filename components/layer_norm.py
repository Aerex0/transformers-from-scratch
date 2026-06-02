import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))  # Scale parameter
        self.beta = nn.Parameter(torch.zeros(d_model))  # Shift parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # Compute mean across the last dimension
        std = x.std(dim=-1, keepdim=True)    # Compute standard deviation across the last dimension
        normalized_x = (x - mean) / (std + self.eps)  # Normalize the input
        return self.gamma * normalized_x + self.beta  # Scale and shift the normalized input