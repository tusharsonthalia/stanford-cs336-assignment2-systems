import argparse
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Literal, cast, get_args

import pandas as pd
import torch

import cs336_basics.model
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

# monkeypatching the annotated sdpa
cs336_basics.model.scaled_dot_product_attention = cs336_basics.model.annotated_scaled_dot_product_attention

Mode = Literal[
        "forward",
        "forward+backward",
        "training-loop"
    ]

MODES: tuple[str, ...] = get_args(Mode)
MATMUL_PRECISIONS = ("highest", "high", "medium")


@dataclass
class Config:
    name: str
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    theta: float = 10000.0
    vocab_size: int = 10000
    batch_size: int = 4
    context_length: int = 512

benchmark_configs = {
    "small": Config(name="small",d_model= 768,d_ff= 3072,num_layers= 12,num_heads= 12),
    "medium": Config(name="medium",d_model=1024,d_ff=4096,num_layers=24,num_heads=16),
    "large": Config(name="large",d_model=1280,d_ff=5120,num_layers=36,num_heads=20),
    # "xl": Config(name="xl",d_model=2560,d_ff=10240,num_layers=32,num_heads=32),
    # "10B": Config(name="10B",d_model=4608,d_ff=12288,num_layers=50,num_heads=36),
}

def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")

def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
        
@dataclass
class Timer:
    name: str
    device: torch.device
    start_time: float = 0.0
    elapsed: float = 0.0
    samples: list[float] = field(default_factory=list)

    def __enter__(self):
        self.elapsed = 0.0
        synchronize(self.device)
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc, tb):
        synchronize(self.device)
        self.elapsed = time.perf_counter() - self.start_time
        self.samples.append(self.elapsed)

@dataclass
class Benchmark:
    name: str
    samples: list[float]

    @property
    def n(self): return len(self.samples)

    @property
    def min(self): return min(self.samples)

    @property
    def max(self): return max(self.samples)
    
    @property
    def mean(self): return sum(self.samples) / self.n

    @property
    def std(self):
        mean = self.mean
        var = sum(map(lambda x: (x - mean) ** 2, self.samples)) / self.n
        return var ** 0.5
    
    @property
    def stats(self):
        return {
            "step_name": self.name,
            "samples": self.samples,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
        }

def make_batch(vocab_size: int, batch: int, context: int, device: torch.device):
    data = torch.randint(0, vocab_size, (batch, context + 1), device=device)
    
    return data[:, :-1], data[:, 1:]

def make_model(config: Config, device: torch.device):
    return TransformerLM(
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        theta=config.theta,
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        device=device
    )

def make_optimizer(model: torch.nn.Module):
    return AdamW(
        params=model.parameters(),
        lr=1e-3,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
def make_step(model, optimizer, train, targets, vocab_size, mode: Mode):
    def forward():
        logits = model(train)
        return cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

    if mode == "forward":
        def step():
            forward()

    elif mode == "forward+backward":
        def step():
            optimizer.zero_grad()
            forward().backward()

    elif mode == "training-loop":
        def step():
            optimizer.zero_grad()
            forward().backward()
            optimizer.step()

    else:
        raise ValueError(f"unknown mode: {mode!r} (expected one of {MODES})")

    return step

def benchmark(
    step,
    warmup: int,
    measurements: int,
    name: str,
    device: torch.device,
):
    for _ in range(warmup):
        step()
    synchronize(device)

    timer = Timer(name=name, device=device)
    for _ in range(measurements):
        with timer:
            step()

    return Benchmark(name=name, samples=timer.samples.copy())

        
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark the Transformer forward / backward / optimizer step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", choices=list(benchmark_configs), default=None, help="model size; omit for all")
    p.add_argument("--mode", choices=list(MODES), default=None, help="phase to time; omit for all")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("-w", "--warmup", type=int, default=5)
    p.add_argument("-n", "--measurements", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="force a device, e.g. cpu / cuda:1")
    p.add_argument("--matmul-precision", choices=MATMUL_PRECISIONS, default="highest",
                   help="'high' enables TF32 for fp32 matmuls")
    p.add_argument("-o", "--out", default=None, help="write the results table to CSV")
    return p


def run(config: Config, device: torch.device, modes: list[str], warmup: int, measurements: int) -> list[dict]:
    train, targets = make_batch(config.vocab_size, config.batch_size, config.context_length, device)

    model = make_model(config, device)
    optimizer = make_optimizer(model)

    results = []
    for mode in modes:
        step = make_step(model, optimizer, train, targets, config.vocab_size, cast(Mode, mode))
        stats = benchmark(step, warmup, measurements, mode, device).stats
        results.append({**asdict(config), **stats, "warmup": warmup})

    return results


if __name__ == "__main__":
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    device = torch.device(args.device) if args.device else resolve_device()

    configs = [args.config] if args.config else list(benchmark_configs)
    modes = [args.mode] if args.mode else list(MODES)

    print(f"device={device} tf32={torch.backends.cuda.matmul.allow_tf32} "
          f"warmup={args.warmup} n={args.measurements}\n")

    results = []
    for name in configs:
        config = replace(benchmark_configs[name],
                         context_length=args.context_length, batch_size=args.batch_size)
        results.extend(run(config, device, modes, args.warmup, args.measurements))
        print(f"{name} done...")

    df = pd.DataFrame(results).drop(columns=["samples", "theta", "vocab_size"])
    print()
    print(df.to_string(index=False))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    ### Commands
    
    # 1.1 Default run wall clock benchmarking with warmup = 5
    # uv run cs336_systems/benchmarking_script.py --context-length 512 --batch-size 4 -w 5 -n 10 -o results/base_benchmark_w5_m10.csv
    
    # 1.2. Default run wall clock benchmarking with warmup = 1
    # uv run cs336_systems/benchmarking_script.py --context-length 512 --batch-size 4 -w 1 -n 10 -o results/base_benchmark_w1_m10.csv
    
    # 1.3. Default run wall clock benchmarking with warmup = 0
    # uv run cs336_systems/benchmarking_script.py --context-length 512 --batch-size 4 -w 0 -n 10 -o results/base_benchmark_w0_m10.csv

    # 2.1 Intro run with Nsys
    # uv run nsys profile --stats=true -o profiles/first python cs336_systems/benchmarking_script.py --config small --context-length 256 --mode forward -w 1 -n 3
    
    # 2.2 Intro run with Nsys with more config options
    # uv run nsys profile --trace=cuda,cublas,nvtx --pytorch=functions-trace --force-overwrite true -o profiles/second python cs336_systems/benchmarking_script.py --config small --context-length 256 --mode forward -w 1 -n 3
    
    # 2.3 Intro run with Nsys with more config options and annotated sdpa
    # uv run nsys profile --trace=cuda,cublas,nvtx --pytorch=functions-trace --force-overwrite true -o profiles/third python cs336_systems/benchmarking_script.py --config small --context-length 256 --mode forward -w 1 -n 3
    
    # 2.4 Intro run with Nsys with more config options and annotated sdpa and full training loop
    # uv run nsys profile --trace=cuda,cublas,nvtx --pytorch=functions-trace,autograd-nvtx --force-overwrite true -o profiles/fourth python cs336_systems/benchmarking_script.py --config small --context-length 256 --mode training-loop -w 1 -n 3
