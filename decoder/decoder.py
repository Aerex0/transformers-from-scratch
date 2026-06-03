import torch
import torch.nn as nn
from config import D_MODEL, DEVICE, N_HEADS
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
        x = self.layer_norm1(masked_attn_output + x)  # Residual Connection + Layer Norm
        # print(f'----> Output after first residual connection and layer norm:\n {x}')
        # print(f'----> Output after first residual connection and layer norm shape: {x.shape}')
        
        # Multi-Head Attention with encoder output as context
        attn_output = self.multi_head_attn(x, encoder_output)
        # print(f'----> Multi Head Attention output:\n {attn_output}')
        # print(f'----> Multi Head Attention output shape: {attn_output.shape}')
        x = self.layer_norm2(attn_output + x)  # Residual Connection + Layer Norm
        # print(f'----> Output after second residual connection and layer norm:\n {x}')
        # print(f'----> Output after second residual connection and layer norm shape: {x.shape}')
        
        # Feed Forward Network
        ff_output = self.feed_forward(x)
        # print(f'----> Feed Forward output:\n {ff_output}')
        # print(f'----> Feed Forward output shape: {ff_output.shape}')
        output = self.layer_norm3(ff_output + x)  # Residual Connection + Layer Norm
        # print(f'----> Output after third residual connection and layer norm:\n {output}')
        # print(f'----> Output after third residual connection and layer norm shape: {output.shape}')
        
        return output