import numpy as np
from typing import Tuple

def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x/np.sum(exp_x, axis=axis, keepdims=True)

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return (Q,K,V)

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    d_k = len(K[0])
    attention_scores = np.dot(Q,K.T)/np.sqrt(d_k)
    attention_weights = softmax(attention_scores)
    return np.dot(attention_weights,V)
    

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    seq_len = len(Q)
    d_model = len(Q[0])
    d_head = d_model // n_heads

    Q = Q.reshape(seq_len, n_heads, d_head)
    K = K.reshape(seq_len, n_heads, d_head)
    V = V.reshape(seq_len, n_heads, d_head)

    Q = Q.transpose(1,0,2)
    K = K.transpose(1,0,2)
    V = V.transpose(1,0,2)

    scores = np.matmul(Q, K.transpose(0,2,1)) / np.sqrt(d_head)
    weights = softmax(scores, axis=-1)
    heads = np.matmul(weights, V)

    heads = heads.transpose(1,0,2)

    output = heads.reshape(seq_len, d_model)

    return output