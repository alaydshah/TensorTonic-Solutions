import torch
import torch.nn.functional as F
import math

def multi_head_attention(x: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor, W_o: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Returns: torch.Tensor of shape (batch, seq_len, d_model)
    """
    
    batch_size, seq_len, d_model = x.shape

    d_head = d_model // n_heads
    
    # shape (B, S, D)
    q = x @ W_q
    k = x @ W_k
    v = x @ W_v

    # Reshape into separate heads 
    q = q.reshape(batch_size, seq_len, n_heads, d_head)
    k = k.reshape(batch_size, seq_len, n_heads, d_head)
    v = v.reshape(batch_size, seq_len, n_heads, d_head)

    # Move the heads before the sequence dimenions
    # shape (batch_size, n_heads, seq_len, d_head)
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    scores = (q @ k.transpose(-2, -1)) / d_head ** 0.5

    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

    masked_scores = scores.masked_fill(mask, float('-inf'))

    max_scores = torch.max(masked_scores, dim=-1, keepdim=True).values

    shifted_scores = masked_scores - max_scores

    exp_scores = torch.exp(shifted_scores)

    attention_weights = exp_scores / torch.sum(exp_scores, dim=-1, keepdim=True)

    output = attention_weights @ v

    output = output.permute(0,2,1,3).reshape(batch_size, seq_len, -1)

    return output @ W_o