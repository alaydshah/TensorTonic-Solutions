import torch
import torch.nn as nn

def gpt2_embedding(token_ids, token_embed_weight, position_embed_weight):
    """
    Returns: torch.Tensor of shape (seq_len, d_model)
    """

    token_ids = torch.tensor(token_ids, dtype=torch.long)

    token_embed_weight = torch.tensor(token_embed_weight, dtype=torch.float32)

    position_embed_weight = torch.tensor(position_embed_weight, dtype=torch.float32)

    # Get the sequence length from the number of token ids
    seq_len = token_ids.shape[0]

    # Build the position indices [0, 1, 2, ..., seq_len - 1]
    position_ids = torch.arange(seq_len)

    # Look up token embeddings for every token id
    token_embeddings = token_embed_weight[token_ids]

    # Look up position embeddings for every position in sequence
    position_embeddings = position_embed_weight[position_ids]

    # Add token embeddings and position embeddings elementwise
    output = token_embeddings + position_embeddings

    # Return the final embeddings
    return output