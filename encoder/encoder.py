import torch
import torch.nn as nn
from config import D_MODEL, N_HEADS, DEVICE, ENOCDER_LAYERS
from components.positional_embeddings import get_positional_embeddings
from components.multi_head_attn import MultiHeadAttention
from components.layer_norm import LayerNorm
from components.feed_forward import FeedForward

class EncoderBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(d_model, n_heads).to(DEVICE)
        self.layer_norm1 = LayerNorm(d_model).to(DEVICE)
        self.feed_forward = FeedForward(d_model).to(DEVICE)
        self.layer_norm2 = LayerNorm(d_model).to(DEVICE)

    def forward(self, x):
        """
        Steps in the forward pass of the Encoder Block:
        1. Multi-Head Attention
        2. Residual Connection + Layer Norm
        3. Feed Forward Network
        4. Residual Connection + Layer Norm
        """
        # Multi-Head Attention
        attn_output = self.multi_head_attn(x)
        # print(f'----> Multi Head Attention output:\n {attn_output}')
        # print(f'----> Multi Head Attention output shape: {attn_output.shape}')
        # Residual Connection + Layer Norm
        x = self.layer_norm1(attn_output + x)
        # print(f'----> Output after first residual connection and layer norm:\n {x}')
        # print(f'----> Output after first residual connection and layer norm shape: {x.shape}')
        # Feed Forward Network
        ff_output = self.feed_forward(x)
        # print(f'----> Feed Forward output:\n {ff_output}')
        # print(f'----> Feed Forward output shape: {ff_output.shape}')
        # Residual Connection + Layer Norm
        output = self.layer_norm2(ff_output + x)
        # print(f'----> Final output after second residual connection and layer norm:\n {output}')
        # print(f'----> Final output after second residual connection and layer norm shape: {output.shape}')
        return output

    def print_parameters(self, print_values=False):
            """
            Prints the layer names, shapes, and optionally the actual tensor values.
            """
            print(f"\n{'='*40}\n MODEL PARAMETERS OVERVIEW \n{'='*40}")
            total_params = 0
            
            for name, param in self.named_parameters():
                print(f"Layer: {name}")
                print(f" -> Shape: {list(param.shape)}")
                print(f" -> Requires Grad: {param.requires_grad}")
                
                if print_values:
                    print(f" -> Values:\n{param.data}\n")
                
                # Calculate total elements in this specific parameter tensor
                total_params += param.numel()
                print("-" * 40)
                
            print(f"\nTotal Trainable Parameters: {total_params:,}\n{'='*40}")

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=ENOCDER_LAYERS, d_model=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads).to(DEVICE) for _ in range(num_layers)
            ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def print_parameters(self, Layer_values=False, print_values=False):
        """
        Prints the layer names, shapes, and optionally the actual tensor values.
        """
        print(f"\n{'='*40}\n ENCODER PARAMETERS OVERVIEW ")
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
        print(f'Encoder Parameters: {total_params}')
        print("=" * 40)
            
        # print(f"\nTotal Trainable Parameters: {total_params:,}\n{'='*40}")
        return total_params