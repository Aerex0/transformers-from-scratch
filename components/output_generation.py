import torch
import torch.nn as nn
from config import D_MODEL, DEVICE
from .embeddings import vocab_size

class OutputGeneration(nn.Module):
    def __init__(self, d_model=D_MODEL, vocab_size=vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size).to(DEVICE)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, decoder_output):
        # Pass through linear layer
        linear_output = self.linear(decoder_output)
        # print(f'----> Output after Linear Layer:\n {linear_output}')
        # print(f'----> Output after Linear Layer shape: {linear_output.shape}')
        # Apply softmax to get probabilities
        output_probs = self.softmax(linear_output)
        # print(f'----> Output probabilities after Softmax:\n {output_probs}')
        # print(f'----> Output probabilities shape: {output_probs.shape}')
        return output_probs