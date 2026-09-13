import argparse
import json
import time

import numpy as np
import torch

from cs336_basics.configs import RUNS, RunConfig
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import (
    cosine_learning_rate_schedule,
    cross_entropy,
    gradient_clipping,
    save_checkpoint,
)
from cs336_basics.optimizer import AdamW
from cs336_basics.plotting import load_metrics, plot_curve


def build_model(config: RunConfig) -> TransformerLM:
    return TransformerLM(
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        theta=config.model.theta,
        vocab_size=config.dataset.vocab_size,
        context_length=config.train.context_length,
        num_layers=config.model.num_layers,
        device=config.train.torch_device,
        dtype=config.model.dtype,
        eps=config.model.eps,
    )

def build_optimizer(config: RunConfig, model: torch.nn.Module) -> AdamW:
    return AdamW(
        params=model.parameters(),
        lr=config.optimizer.lr,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )


def describe(config: RunConfig, train_tokens: int, validation_tokens: int) -> None:
    """Print what produced this log -- a sweep's numbers are useless without it."""
    arch, optimizer, run = config.model, config.optimizer, config.train
    print(f"run        {config.run_name}  ({config.dataset.name}, {arch.name})")
    print(f"model      {config.num_parameters:,} params  d_model {arch.d_model} "
          f"x {arch.num_layers} layers x {arch.num_heads} heads")
    print(f"data       {train_tokens:,} train / {validation_tokens:,} validation tokens")
    print(f"budget     {run.steps:,} steps x {run.batch_size} batch x {run.context_length} ctx "
          f"= {run.tokens_processed:,} tokens, {config.flops_per_step * run.steps:.2e} FLOPs")
    print(f"schedule   lr {optimizer.lr:.2e} -> {run.min_lr:.2e}, warmup {run.warmup_iters:,}, "
          f"decay to step {run.cosine_cycle_iters:,}")
    print(f"device     {run.device}  seed {run.random_seed}  -> {config.checkpoint_path}")


def format_progress(row: dict, config: RunConfig, flops_per_step: int) -> str:
    """One progress line: losses, throughput, and MFU where the peak is known."""
    run, step, elapsed = config.train, row["step"], row["elapsed"]
    flops = flops_per_step * step / elapsed
    mfu = f" ({flops / run.peak_flops:5.1%} MFU)" if run.peak_flops else ""
    return (
        f"  step {step:>7,}/{run.steps:,}  lr {row['lr']:.2e}  "
        f"train {row['train_loss']:.4f}  val {row['validation_loss']:.4f}  "
        f"{step * run.batch_size * run.context_length / elapsed:>9,.0f} tok/s  "
        f"{flops / 1e12:6.2f} TFLOP/s{mfu}  "
        f"eta {elapsed * (run.steps - step) / step / 60:5.1f}m"
    )


@torch.no_grad()
def evaluate(model: TransformerLM, data, config: RunConfig) -> float:
    """Mean cross-entropy over a few batches of held-out data."""
    run = config.train
    model.eval()
    total = 0.0
    for _ in range(run.eval_batches):
        inputs, targets = get_batch(data, run.batch_size, run.context_length, device=run.device)
        logits = model(inputs)
        total += cross_entropy(
            logits.reshape(-1, config.dataset.vocab_size), targets.reshape(-1)
        ).item()
    model.train()

    return total / run.eval_batches


def train(config: RunConfig) -> list[dict]:
    run = config.train
    # get_batch draws from numpy, parameter init from torch -- seed both
    np.random.seed(run.random_seed)
    torch.manual_seed(run.random_seed)
    # TF32 on cuda; must stay None on mps, where torch 2.9 picks broken kernels
    if run.matmul_precision is not None:
        torch.set_float32_matmul_precision(run.matmul_precision)

    config.checkpoint_path.mkdir(parents=True, exist_ok=True)
    metrics_path = config.checkpoint_path / "metrics.jsonl"
    curve_path = config.checkpoint_path / "curve.png"
    title = f"{config.run_name}  ·  {config.num_parameters:,} params"

    train_data, validation_data = config.dataset.load()
    model = build_model(config)
    optimizer = build_optimizer(config)
    describe(config, train_data.size, validation_data.size)

    model.train()
    flops_per_step = config.flops_per_step
    start = time.perf_counter()

    for step in range(1, run.steps + 1):
        lr = cosine_learning_rate_schedule(
            step, config.optimizer.lr, run.min_lr, run.warmup_iters, run.cosine_cycle_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch(
            train_data, run.batch_size, run.context_length, device=run.device
        )
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits.reshape(-1, config.dataset.vocab_size), targets.reshape(-1))
        loss.backward()
        gradient_clipping(
            model.parameters(), config.optimizer.grad_clip, config.optimizer.grad_clip_eps
        )
        optimizer.step()

        # loss.item() forces a device sync, so only read it on the eval interval
        if step % run.eval_interval == 0 or step == run.steps:
            # evaluate() flips the model to eval() and back, so keep that call
            # visible rather than buried inside the dict literal below
            train_loss = loss.item()
            validation_loss = evaluate(model, validation_data, config)
            row = {
                "step": step,
                "lr": lr,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "elapsed": time.perf_counter() - start,
            }
            # one JSON object per line: a killed run keeps every row before the kill
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(format_progress(row, config, flops_per_step))

        if step % run.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step, config.checkpoint_path / f"step_{step:07d}.pt")
            plot_curve(metrics_path, curve_path, title)

    save_checkpoint(model, optimizer, run.steps, config.checkpoint_path / "final.pt")
    plot_curve(metrics_path, curve_path, title)

    total = time.perf_counter() - start
    print(f"done       {total / 60:.1f} min, {run.tokens_processed / total:,.0f} tok/s, "
          f"{flops_per_step * run.steps / total / 1e12:.2f} TFLOP/s")
    print(f"artifacts  {config.checkpoint_path}/  (curve.png, metrics.jsonl, "
          f"{len(list(config.checkpoint_path.glob('*.pt')))} checkpoints)")

    return load_metrics(metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer LM.")
    parser.add_argument("run", nargs="?", default="smoke", choices=sorted(RUNS))
    args = parser.parse_args()

    train(config=RUNS[args.run])
