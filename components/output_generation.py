import torch
import torch.nn as nn
from config import D_MODEL, DEVICE
from .embeddings import vocab_size



# Linear Layer
layer = nn.Linear(D_MODEL, vocab_size).to(DEVICE)

# Softmax
softmax = nn.Softmax(dim=-1)

def generate_output(decoder_output):
    # Pass through linear layer
    linear_output = layer(decoder_output)
    print(f'----> Output after Linear Layer:\n {linear_output}')
    print(f'----> Output after Linear Layer shape: {linear_output.shape}')
    # Apply softmax to get probabilities
    output_probs = softmax(linear_output)
    print(f'----> Output probabilities after Softmax:\n {output_probs}')
    print(f'----> Output probabilities shape: {output_probs.shape}')
    return output_probs