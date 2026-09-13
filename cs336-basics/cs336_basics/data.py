import numpy as np
import torch
from jaxtyping import Int
from torch import Tensor


def get_batch(
    x: Int[np.ndarray, " token_ids"],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[
        Int[Tensor, "batch_size context_length"], 
        Int[Tensor, "batch_size context_length"]
    ]:
    """Sample a batch of (input, target) sequences from a flat token array.

    Any start index in [0, n - context_length) gives a valid training example, so
    sampling is a single uniform draw with no padding and no document alignment.
    Targets are the inputs shifted one position left -- position i predicts i+1.

    Gathers context_length + 1 tokens per row and slices twice, so each token is
    read once.

    Args:
        x: 1-D array of token IDs, typically an np.memmap over a .bin file.
        batch_size: Number of sequences per batch.
        context_length: Tokens per sequence.
        device: Torch device string, e.g. "cpu", "cuda:0", "mps".

    Returns:
        (inputs, targets), each of shape (batch_size, context_length) and dtype
        torch.long, on the requested device.
    """
    
    n = x.shape[0]
    max_start = n - context_length
    sample_start_idx = np.random.randint(0, max_start, size=(batch_size,1))
    offsets = np.arange(0, context_length + 1).reshape(1, -1)

    indices = sample_start_idx + offsets
    dataset = x[indices]

    inputs = torch.from_numpy(dataset[:, :-1]).to(device=device, dtype=torch.long)
    targets = torch.from_numpy(dataset[:, 1:]).to(device=device, dtype=torch.long)

    return inputs, targets