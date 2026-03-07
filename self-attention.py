import numpy as np

def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x/np.sum(exp_x, axis=axis, keepdims=True)

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    d_k = len(K[0])
    attention_scores = np.dot(Q,K.T)/np.sqrt(d_k)
    attention_weights = softmax(attention_scores)
    return np.dot(attention_weights,V)
