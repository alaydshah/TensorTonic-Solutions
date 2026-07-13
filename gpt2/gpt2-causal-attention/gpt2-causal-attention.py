import torch
import torch.nn.functional as F
import math

def causal_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, seq_len, d_v)
    """
    # Your code here
    d_k = Q.shape[-1]

    # scaled scores
    scores = Q @ K.transpose(-2, -1)  / d_k ** 0.5

    # Sequence length
    seq_len = Q.shape[-2]

    # Causal mask. True where j > i meaning "future token"
    mask = torch.triu(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool), 
                     diagonal=1)

    scores = scores.masked_fill(mask, float('-inf'))

    max_val = torch.max(scores, dim=-1, keepdim=True).values

    exp_scores = torch.exp(scores - max_val)

    attention_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)

    return attention_weights @ V