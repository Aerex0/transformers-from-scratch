import math
import torch
import torch.nn as nn

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # 1. Define learnable weight projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # 2. Output projection layer to mix head contexts back together
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)  # Adding dropout for regularization

    def self_attention(self, Q, K, V):
        # Q, K, V shapes: (batch_size, n_heads, seq_len, d_head)
        # We grab d_k from the last dimension dynamically
        d_k = Q.size(-1)
        
        # Matrix multiply Q and K^T across the last two dimensions
        # (seq_len, d_head) @ (d_head, seq_len) -> (seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Create a mask to prevent attention to future tokens (for autoregressive decoding)
        mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1), device=scores.device))
        # Mask out future tokens by setting their scores to a very large negative value
        scores = scores.masked_fill(mask == 0, -1e9)
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Multiply weights with V: (seq_len, seq_len) @ (seq_len, d_head)
        return torch.matmul(attention_weights, V)

    def forward(self, embeddings):
        batch_size, seq_len, d_model = embeddings.shape
        
        # Linear Projections
        q_all = self.q_proj(embeddings)
        k_all = self.k_proj(embeddings)
        v_all = self.v_proj(embeddings)
        
        # Reshape and Transpose to separate heads
        # (batch_size, n_heads, seq_len, d_head)
        Q = q_all.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = k_all.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = v_all.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Compute Scaled Dot-Product Attention for all heads simultaneously
        attn_out = self.self_attention(Q, K, V) # (batch_size, n_heads, seq_len, d_head)
        
        # SConcatenate heads back together
        # We must call .contiguous() before using .view() after a transpose operation
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch_size, seq_len, d_model)
        
        # Final linear projection
        return self.out_proj(attn_out)