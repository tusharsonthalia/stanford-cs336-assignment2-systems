from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


@dataclass
class DataConfig:
    """Base Config for all the datasets."""

    name: str
    vocab_size: int
    vocab_path: str
    merges_path: str
    train_path: str
    validation_path: str
    token_dtype: str = "uint16"

    def load(self) -> tuple[np.memmap, np.memmap]:
        """Memory-map the train and validation token arrays.

        Opens files, so call it once and hold the result. The dtype must match
        what encode_to_uint16_bin wrote -- a mismatch reads silent garbage.
        """
        for path in (self.train_path, self.validation_path):
            if not Path(path).exists():
                raise FileNotFoundError(f"{path} -- run scripts/tokenize_and_encode.py {self.name}")
        return (
            np.memmap(self.train_path, dtype=self.token_dtype, mode="r"),
            np.memmap(self.validation_path, dtype=self.token_dtype, mode="r"),
        )


@dataclass
class ModelConfig:
    """Base Class for all model configurations."""

    name: str
    d_model: int
    num_heads: int
    d_ff: int
    num_layers: int
    theta: float = 10000.0
    eps: float = 1e-5
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError(f"d_model {self.d_model} not divisible by num_heads {self.num_heads}")
        if (self.d_model // self.num_heads) % 2:
            raise ValueError(f"head_dim {self.d_model // self.num_heads} is odd; RoPE needs pairs")
        if self.d_ff % 64:
            raise ValueError(f"d_ff {self.d_ff} is not a multiple of 64")


@dataclass
class OptimizerConfig:
    """AdamW plus the gradient clipping applied before each step."""

    name: str = "AdamW"
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    eps: float = 1e-8
    grad_clip: float = 1.0
    grad_clip_eps: float = 1e-6


@dataclass
class TrainConfig:
    """Base Class for all the training experiments."""
    name: str = "smoke"
    steps: int = 20
    batch_size: int = 4
    context_length: int = 256
    device: str = "mps"
    random_seed: int = 42

    # cosine schedule; cycle defaults to `steps` so decay ends exactly at the last step
    warmup_iters: int = 2
    min_lr: float = 0.0
    cosine_cycle_iters: int = 0

    eval_interval: int = 10
    eval_batches: int = 10
    checkpoint_interval: int = 10
    checkpoint_dir: str = "checkpoints"

    # "high" enables TF32 on cuda; leave None on mps
    matmul_precision: str | None = None

    # device peak FLOP/s, for reporting MFU. 0 means "unknown, do not report"
    peak_flops: float = 0.0

    def __post_init__(self) -> None:
        if not self.cosine_cycle_iters:
            self.cosine_cycle_iters = self.steps
        if self.warmup_iters >= self.steps:
            raise ValueError(f"warmup_iters {self.warmup_iters} >= steps {self.steps}")

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def tokens_processed(self) -> int:
        return self.steps * self.batch_size * self.context_length


# datasets

def tinystories() -> DataConfig:
    return DataConfig(
        name="tinystories",
        vocab_size=10_000,
        vocab_path="tokenizer/tinystories_bpe_vocab_10000.pkl",
        merges_path="tokenizer/tinystories_bpe_merges_10000.pkl",
        train_path="tokenizer/tinystories_train.uint16.bin",
        validation_path="tokenizer/tinystories_valid.uint16.bin",
    )


def owt() -> DataConfig:
    return DataConfig(
        name="owt",
        vocab_size=32_000,
        vocab_path="tokenizer/owt_bpe_vocab_32000.pkl",
        merges_path="tokenizer/owt_bpe_merges_32000.pkl",
        train_path="tokenizer/owt_train.uint16.bin",
        validation_path="tokenizer/owt_valid.uint16.bin",
    )


# architectures

def small_model() -> ModelConfig:
    """TinyStories architecture: ~17M non-embedding parameters."""
    return ModelConfig(name="small", d_model=512, num_heads=16, d_ff=1344, num_layers=4)


def medium_model() -> ModelConfig:
    """GPT-2-small scale: ~85M non-embedding parameters."""
    return ModelConfig(name="medium", d_model=768, num_heads=12, d_ff=2048, num_layers=12)


# runs


@dataclass
class RunConfig:
    run_name: str = "smoke"
    dataset: DataConfig = field(default_factory=tinystories)
    model: ModelConfig = field(default_factory=small_model)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.train.checkpoint_dir) / self.run_name

    @property
    def flops_per_step(self) -> int:
        """Forward+backward matmul FLOPs for one optimizer step."""
        m, run = self.model, self.train
        context, d_model = run.context_length, m.d_model
        forward = (
            m.num_layers * (24 * context * d_model**2 + 4 * context**2 * d_model)
            + 2 * context * d_model * self.dataset.vocab_size
        )
        return 3 * forward * run.batch_size

    @property
    def num_parameters(self) -> int:
        m, v = self.model, self.dataset.vocab_size
        per_layer = 4 * m.d_model**2 + 3 * m.d_model * m.d_ff + 2 * m.d_model
        return 2 * v * m.d_model + m.num_layers * per_layer + m.d_model


def smoke_run() -> RunConfig:
    """20 steps on mps -- proves the loop runs end to end. Same as RunConfig()."""
    return RunConfig()


def mps_run() -> RunConfig:
    """TinyStories on Apple Silicon: 32 x 5000 x 256 = 41M tokens."""
    return RunConfig(
        run_name="tinystories-mps",
        dataset=tinystories(),
        model=small_model(),
        train=TrainConfig(
            name="mps", steps=5_000, batch_size=32, device="mps",
            warmup_iters=250, min_lr=1e-4, eval_interval=100, checkpoint_interval=1_000,
        ),
    )


def tinystories_small_run() -> RunConfig:
    """TinyStories at full budget: 128 x 10000 x 256 = 328M tokens."""
    return RunConfig(
        run_name="tinystories-small",
        dataset=tinystories(),
        model=small_model(),
        train=TrainConfig(
            name="tinystories-small", steps=10_000, batch_size=128, device="cuda",
            warmup_iters=500, min_lr=1e-4, eval_interval=200, checkpoint_interval=2_000,
            matmul_precision="high", peak_flops=495e12,  # H100 TF32
        ),
    )


def owt_small_run() -> RunConfig:
    """OWT with the SAME architecture and iteration count as the TinyStories
    run, so the two learning curves are directly comparable."""
    config = tinystories_small_run()
    config.run_name = "owt-small"
    config.dataset = owt()
    config.train.name = "owt-small"
    return config


def owt_medium_run() -> RunConfig:
    """OWT with a larger model and a longer context."""
    return RunConfig(
        run_name="owt-medium",
        dataset=owt(),
        model=medium_model(),
        optimizer=OptimizerConfig(lr=6e-4, weight_decay=0.1),
        train=TrainConfig(
            name="medium", steps=20_000, batch_size=64, context_length=512,
            device="cuda", warmup_iters=1_000, min_lr=6e-5,
            eval_interval=250, checkpoint_interval=2_000, matmul_precision="high",
            peak_flops=495e12,  # H100 TF32
        ),
    )


RUNS = {
    "smoke": smoke_run(),
    "mps": mps_run(),
    "h100": tinystories_small_run(),
    "owt": owt_small_run(),
    "medium": owt_medium_run(),
}
