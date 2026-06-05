import torch
import torch.nn as nn
from config import D_MODEL, DEVICE
from components.dataset import token_ids

class OutputGeneration(nn.Module):
    def __init__(self, d_model=D_MODEL, vocab_size=len(token_ids)):
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

    def print_parameters(self, Layer_values=False, print_values=False):
        """
        Prints the layer names, shapes, and optionally the actual tensor values.
        """
        print(f"\n{'='*40}\n OUTPUT GENERATION PARAMETERS OVERVIEW")
        total_params = 0
        
        for name, param in self.named_parameters():
            if Layer_values:
                print(f"Layer: {name}")
                print(f" -> Shape: {list(param.shape)}")
                print(f" -> Requires Grad: {param.requires_grad}")
            
            if print_values:
                print(f" -> Values:\n{param.data}\n")
            
            # Calculate total elements in this specific parameter tensor
            total_params += param.numel()
        print(f'Output Generation Parameters: {total_params}')
        print("=" * 40)
            
        # print(f"\nTotal Trainable Parameters: {total_params:,}\n{'='*40}")
        return total_params