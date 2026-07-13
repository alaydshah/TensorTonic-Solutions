import torch

def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """
    Returns: torch.Tensor with LayerNorm applied across the last dimension
    """

    mean = x.mean(dim=-1, keepdim=True)

    var = x.var(dim=-1, keepdim=True, unbiased=False)

    x_norm = (x - mean) / torch.sqrt(var + eps)
    
    return gamma * x_norm + beta

    

