import torch
import torch.nn as nn
from config import D_MODEL, DEVICE
from components.dataset import token_ids
# from components.embeddings import embedding_matrix

debug = False

class OutputGeneration(nn.Module):
    def __init__(self, d_model=D_MODEL, vocab_size=len(token_ids), shared_weight=None):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=False).to(DEVICE)
        if shared_weight is not None:
            # Tie the weights directly to the embedding weights
            self.linear.weight = shared_weight

    def forward(self, decoder_output):
        # Pass through linear layer
        logits = self.linear(decoder_output)
        if debug:
            print(f'----> Output after Linear Layer:\n {logits}')
            print(f'----> Output after Linear Layer shape: {logits.shape}')
        return logits

    def print_parameters(self, Layer_values=False, print_values=False):
        """
        Prints the layer names, shapes, and optionally the actual tensor values.
        """
        print(f"\n{'='*40}\n OUTPUT GENERATION PARAMETERS OVERVIEW")
        # print(f'Address of linear layer weights: {id(self.linear.weight)}, Address of embedding matrix weights: {id(embedding_matrix.weight)}')
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