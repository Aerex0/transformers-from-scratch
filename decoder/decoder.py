import torch
import torch.nn as nn
from config import D_MODEL, DEVICE, N_HEADS, DECODER_LAYERS
# from components.embeddings import output_embeddings
from components.positional_embeddings import get_positional_embeddings
from components.multi_head_attn import MultiHeadAttention
from components.layer_norm import LayerNorm
from components.feed_forward import FeedForward
from components.masked_multi_head_attention import MaskedMultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.masked_multi_head_attn = MaskedMultiHeadAttention(d_model, n_heads).to(DEVICE)
        self.layer_norm1 = LayerNorm(d_model).to(DEVICE)
        self.multi_head_attn = MultiHeadAttention(d_model, n_heads, cross_attention=True, encoder_output=None).to(DEVICE)
        self.layer_norm2 = LayerNorm(d_model).to(DEVICE)
        self.feed_forward = FeedForward(d_model).to(DEVICE)
        self.layer_norm3 = LayerNorm(d_model).to(DEVICE)

        # Dropout layers for regularization
        self.masked_multi_head_attn_dropout = nn.Dropout(0.1).to(DEVICE)
        self.cross_attn_dropout = nn.Dropout(0.1).to(DEVICE)
        self.ff_dropout = nn.Dropout(0.1).to(DEVICE)


    def forward(self, x, encoder_output):
        """
        Steps in the forward pass of the Decoder Block:
        1. Masked Multi-Head Attention (self-attention)
        2. Residual Connection + Layer Norm
        3. Multi-Head Attention with encoder output as context (cross-attention)
        4. Residual Connection + Layer Norm
        5. Feed Forward Network
        6. Residual Connection + Layer Norm
        """
        # Masked Multi-Head Attention
        masked_attn_output = self.masked_multi_head_attn(x)
        # print(f'----> Masked Multi Head Attention output:\n {masked_attn_output}')
        # print(f'----> Masked Multi Head Attention output shape: {masked_attn_output.shape}')
        x = self.layer_norm1(self.masked_multi_head_attn_dropout(masked_attn_output) + x)  # Residual Connection + Layer Norm
        # print(f'----> Output after first residual connection and layer norm:\n {x}')
        # print(f'----> Output after first residual connection and layer norm shape: {x.shape}')
        
        # Multi-Head Attention with encoder output as context
        attn_output = self.multi_head_attn(x, encoder_output)
        # print(f'----> Multi Head Attention output:\n {attn_output}')
        # print(f'----> Multi Head Attention output shape: {attn_output.shape}')
        x = self.layer_norm2(self.cross_attn_dropout(attn_output) + x)  # Residual Connection + Layer Norm
        # print(f'----> Output after second residual connection and layer norm:\n {x}')
        # print(f'----> Output after second residual connection and layer norm shape: {x.shape}')
        
        # Feed Forward Network
        ff_output = self.feed_forward(x)
        # print(f'----> Feed Forward output:\n {ff_output}')
        # print(f'----> Feed Forward output shape: {ff_output.shape}')
        output = self.layer_norm3(self.ff_dropout(ff_output) + x)  # Residual Connection + Layer Norm
        # print(f'----> Output after third residual connection and layer norm:\n {output}')
        # print(f'----> Output after third residual connection and layer norm shape: {output.shape}')
        
        return output

    def print_parameters(self, print_values=False):
            """
            Prints the layer names, shapes, and optionally the actual tensor values.
            """
            print(f"\n{'='*40}\n DECODER PARAMETERS OVERVIEW \n{'='*40}")
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

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=DECODER_LAYERS, d_model=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads).to(DEVICE) for _ in range(num_layers)
            ])

    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x

    def print_parameters(self, Layer_values=False, print_values=False):
        """
        Prints the layer names, shapes, and optionally the actual tensor values.
        """
        print(f"\n{'='*40}\n DECODER PARAMETERS OVERVIEW")
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
        print(f'Decoder Parameters: {total_params}')
        print("-" * 40)
            
        # print(f"\nTotal Trainable Parameters: {total_params:,}\n{'='*40}")
        return total_params