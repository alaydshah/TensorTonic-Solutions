import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V):
    """
    Returns: torch.Tensor of shape (batch, seq_q, d_v)
    """
    # All tensors are of shape (batch, seq_q, d_k)
    Q = torch.tensor(Q, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32)
    V = torch.tensor(V, dtype=torch.float32)

    d_k = Q.shape[-1]

    scores = Q @ K.transpose(-2, -1)

    scaled_scores = scores / d_k ** 0.5

    max_values = torch.max(scaled_scores, dim=-1, keepdim=True).values

    shifted_scores = scaled_scores - max_values

    exp_scores = torch.exp(shifted_scores)

    attention_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)

    output = attention_weights @ V

    return output
    