import math
import os
from collections.abc import Iterable
from typing import IO, BinaryIO

import torch
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    logits: Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Average cross-entropy loss between logits and integer targets.

        l = -log softmax(o)[t] = log sum_a exp(o[a]) - o[t]

    Softmax is never materialized. Cancelling the log against the exp leaves a
    logsumexp, which torch evaluates with the row max already factored out.

    Args:
        logits: Unnormalized scores of shape (batch_size, vocab_size).
        targets: Index of the correct class for each row, shape (batch_size,).

    Returns:
        Scalar tensor: the loss averaged over the batch.
    """
    # log of the softmax denominator, one per row
    log_exp_sum = torch.logsumexp(logits, dim=-1)

    # pick out o[t], the logit of the correct class in each row. gather needs
    # the index to match the source rank, hence the unsqueeze/squeeze pair
    correct_logit = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    loss = (log_exp_sum - correct_logit).mean()

    return loss

def cosine_learning_rate_schedule(
    t: int,
    a_max: float,
    a_min: float,
    T_w: int,
    T_c: int
) -> float:
    """Learning rate at iteration `t` under warmup + cosine annealing.

    Three regimes:

        t < T_w             (linear warmup):
            a_t = (t / T_w) * a_max
        T_w <= t <= T_c     (cosine annealing):
            p = (t - T_w) / (T_c - T_w)
            a_t = a_min + (1 + cos(pi * p)) / 2 * (a_max - a_min)
        t > T_c             (constant tail)
            a_t = a_min

    Warmup exists because the moment estimates in Adam are not meaningful on
    the first few steps.

    Args:
        t (int): Current iteration, counted from 0.
        a_max (float): Peak learning rate, reached at t = T_w.
        a_min (float): Final learning rate, held for all t > T_c.
        T_w (int): Number of warmup iterations.
        T_c (int): Iteration at which annealing finishes.

    Returns:
        The scalar learning rate to use at iteration `t`.
    """
    if t < T_w:
        return a_max * t / T_w
    elif t <= T_c:
        cosine_factor = math.cos(math.pi * (t - T_w) / (T_c - T_w)) + 1
        lr_range = a_max - a_min
        return a_min + cosine_factor * lr_range / 2
    else:
        return a_min

@torch.no_grad()
def gradient_clipping(
    params: Iterable[Tensor],
    max_l2: float,
    eps: float = 1e-6
) -> None:
    """Rescale gradients in place so their combined L2 norm is at most `max_l2`.

    The norm is global: every gradient is treated as one slice of a single
    flat vector, so the rescale is a uniform shrink. Direction is preserved
    exactly and only magnitude changes.

    Args:
        params: Parameters whose `.grad` should be clipped. Gradients that are
            None (frozen parameters) are skipped.
        max_l2 (float): Maximum allowed L2 norm of the combined gradient.
        eps (float): Added to the denominator so the division stays safe and the
            result lands just inside the ball rather than exactly on it.

    Returns:
        None. `param.grad` is modified in place.
    """
    total_norm = 0
    gradients = []
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        # accumulate in fp32: squaring an fp16 gradient overflows to inf
        # above |g| ~ 256, which would zero every gradient in the model
        total_norm += grad.float().square().sum()
        gradients.append(grad)

    total_norm = math.sqrt(total_norm)

    if total_norm < max_l2:
        return

    scale = max_l2 / (total_norm + eps)

    for grad in gradients:
        grad.mul_(scale)

    return

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
):
    """Serialize model weights, optimizer state, and iteration count to `out`.

    Args:
        model: Model whose state_dict to save.
        optimizer: Optimizer whose state_dict to save.
        iteration: Training step reached, so the LR schedule can resume in phase.
        out: Destination path or writable binary file object.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """Restore model and optimizer state from `src` in place.

    Args:
        src: Path or readable binary file object written by save_checkpoint.
        model: Model to load weights into. Must match the saved architecture.
        optimizer: Optimizer to load state into. Must be the same class, and its
            param_groups must line up with the saved ones.

    Returns:
        The iteration count that was saved, for resuming the LR schedule.
    """
    checkpoint = torch.load(src, map_location="cpu")

    iteration = checkpoint["iteration"]
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return iteration