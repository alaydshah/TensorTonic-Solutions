import torch

def gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Returns: torch.Tensor with GELU applied element-wise
    """

    sqrt_two = torch.sqrt(torch.tensor(2.0))
    scaled_x = x / sqrt_two
    
    erf_term = torch.erf(scaled_x)

    gate = 0.5 * (1.0 + erf_term)

    return x * gate    