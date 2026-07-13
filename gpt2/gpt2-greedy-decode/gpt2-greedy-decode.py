import torch

def greedy_decode(input_ids, logits_map, num_steps):
    """
    Returns: list of int (full generated sequence including input_ids)
    """
    generated = []

    for token in input_ids:

        generated.append(token)
    
    for _ in range(num_steps):

        key = tuple(generated)

        if not key in logits_map:

            return generated

        logits = logits_map[key]

        sampled_token = torch.argmax(torch.tensor(logits)).item()

        generated.append(sampled_token)
    
    return generated