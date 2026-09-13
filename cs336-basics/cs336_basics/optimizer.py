import math
from collections.abc import Callable, Iterable

import torch
from torch import Tensor, optim


class SGD(optim.Optimizer):
    """Reference SGD with a decaying step size.

        weights <- weights - (lr / sqrt(t + 1)) * grad

    Included as a worked example of the torch.optim.Optimizer contract rather
    than as a serious optimizer: the 1/sqrt(t) decay is hard-wired instead of
    being driven by a schedule, and there is no momentum.

    Args:
        params: Parameters to optimize, or an iterable of param group dicts.
        lr (float): Base learning rate, before the 1/sqrt(t + 1) decay.
    """
    def __init__(self, params: Iterable[Tensor], lr = 1e-3):
        # every key here becomes a per-group default, so param_groups can later
        # override lr per group (e.g. no decay on norms) without touching step()
        defaults = {"lr": lr}
        super().__init__(params=params, defaults=defaults)


    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> float | None: # type: ignore[override]
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                # frozen parameters (requires_grad=False) never get a .grad
                if param.grad is None:
                    continue

                state = self.state[param]
                t = state.get("t", 0)
                param.add_(param.grad, alpha=-lr / math.sqrt(t + 1))
                state["t"] = t + 1

        return loss

class AdamW(optim.Optimizer):
    """AdamW: Adam with decoupled weight decay.

    Per parameter, per step t:

        weights <- weights - alpha * gamma * weights    (decoupled weight decay)
        m     <- b1 * m + (1 - b1) * g                  (first moment, EMA of g)
        v     <- b2 * v + (1 - b2) * g^2                (second moment, EMA of g^2)
        a_t   <- alpha * sqrt(1 - b2^t) / (1 - b1^t)    (bias-corrected step size)
        weights <- weights - a_t * m / (sqrt(v) + eps)

    Adam rescales each coordinate by its own recent gradient magnitude, so
    parameters with small but consistent gradients still move. The moments are
    EMAs initialized at zero.

    Stateful: two full-size buffers (m and v) per parameter, so the optimizer
    costs 2x the model's parameter memory on top of the parameters and gradients.

    Args:
        params: Parameters to optimize, or an iterable of param group dicts.
        lr (float): Learning rate alpha.
        betas: (b1, b2), the decay rates for the first and second moments.
        eps (float): Added to sqrt(v) to keep the division finite when a
            coordinate's gradient history is all zeros.
        weight_decay (float): gamma, the decoupled decay rate.
    """
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        betas: tuple[float, float],
        eps: float = 1e-8,
        weight_decay: float = 0,
    ):
        defaults = {
            "lr": lr,
            "beta_1": betas[0],
            "beta_2": betas[1],
            "weight_decay": weight_decay,
            "eps": eps,
        }
        super().__init__(params=params, defaults=defaults)

    # every tensor op below mutates parameters in place; without no_grad, autograd
    # raises on in-place writes to leaf variables that require grad
    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> float | None: # type: ignore[override]
        loss = None
        if closure is not None:
            # needs a backward graph -- so grad mode is switched back on
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue

                state = self.state[param]
                t = state.get("t", 1)
                if t == 1:
                    state["first_moment"] = torch.zeros_like(param)
                    state["second_moment"] = torch.zeros_like(param)

                first_moment = state["first_moment"]
                second_moment = state["second_moment"]

                # decoupled decay: plain lr, not adjusted_lr, and applied
                # directly to weights so it never enters m or v
                param.mul_(1 - lr * weight_decay)

                # EMAs updated in place -- mul_ then add_(alpha=)
                first_moment.mul_(beta_1).add_(grad, alpha=(1 - beta_1))
                # addcmul_ fuses g*g and the accumulate into one kernel
                second_moment.mul_(beta_2).addcmul_(grad, grad, value=(1 - beta_2))

                adjusted_lr = lr * math.sqrt(1 - math.pow(beta_2, t)) / (1 - math.pow(beta_1, t))
                param.addcdiv_(first_moment, second_moment.sqrt().add_(eps), value=-adjusted_lr)
                    
                state["t"] = t + 1

        return loss