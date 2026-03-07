import numpy as np

def softmax(x, axis=-1):
    x_shifted = x - np.max(x, axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x/np.sum(exp_x, axis=axis, keepdims=True)

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	d_k = len(K[0])
	attention_scores = np.dot(Q,K.T)/np.sqrt(d_k) + mask
	attention_weights = softmax(attention_scores)
	return np.dot(attention_weights,V)
	