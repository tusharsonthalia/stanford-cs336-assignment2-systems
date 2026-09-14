# Benchmarking results: warm-up study

End-to-end timings for `benchmark_forward` / `forward+backward` / `training-loop`
across three model sizes, at three warm-up settings.

## Setup

- **Script:** `cs336_systems/benchmarking_script.py`
- **Measurement steps:** 10 per cell; timings are wall-clock (`time.perf_counter`)
  with `torch.cuda.synchronize()` on both sides of the timed region.
- **Precision:** fp32, TF32 **disabled** (`allow_tf32 = False`,
  `get_float32_matmul_precision() == "highest"`), so matmuls run on FP32 CUDA
  cores rather than tensor cores.
- **Fixed:** vocab 10,000; batch 4; context length 512; RoPE theta 10,000.
- **Model and optimizer are reused across the three modes** within a config.
  See the caveat at the bottom -- this matters for interpreting the w=0 run.
- xl and 10B were commented out for these runs.

| config | d_model | d_ff | layers | heads | params |
|---|---:|---:|---:|---:|---:|
| small | 768 | 3072 | 12 | 12 | 128,625,408 |
| medium | 1024 | 4096 | 24 | 16 | 423,183,360 |
| large | 1280 | 5120 | 36 | 20 | 969,411,840 |

> Note: the w=5 run is labelled from the `run(..., warmup=5)` default in the
> script at the time it was produced; only the w=0 and w=1 runs were explicitly
> labelled. Worth confirming before it goes in the writeup.

## Raw results

### warm-up = 0

| config | mode | mean (ms) | std (ms) | min (ms) | max (ms) | rel. std |
|---|---|---:|---:|---:|---:|---:|
| small | forward | 56.748 | 108.857 | 20.365 | 383.318 | 191.83% |
| small | forward+backward | 68.496 | 15.236 | 63.179 | 114.193 | 22.24% |
| small | training-loop | 68.400 | 1.537 | 67.460 | 72.784 | 2.25% |
| medium | forward | 64.557 | 27.660 | 55.213 | 147.537 | 42.85% |
| medium | forward+backward | 173.593 | 0.231 | 173.123 | 174.000 | 0.13% |
| medium | training-loop | 185.829 | 0.632 | 185.390 | 187.669 | 0.34% |
| large | forward | 127.026 | 10.452 | 123.420 | 158.380 | 8.23% |
| large | forward+backward | 393.840 | 0.270 | 393.217 | 394.195 | 0.07% |
| large | training-loop | 421.308 | 0.938 | 420.776 | 424.056 | 0.22% |

### warm-up = 1

| config | mode | mean (ms) | std (ms) | min (ms) | max (ms) | rel. std |
|---|---|---:|---:|---:|---:|---:|
| small | forward | 20.702 | 0.090 | 20.562 | 20.856 | 0.43% |
| small | forward+backward | 63.706 | 0.110 | 63.574 | 63.916 | 0.17% |
| small | training-loop | 68.205 | 0.112 | 68.036 | 68.435 | 0.16% |
| medium | forward | 55.609 | 0.046 | 55.554 | 55.697 | 0.08% |
| medium | forward+backward | 174.031 | 0.128 | 173.882 | 174.344 | 0.07% |
| medium | training-loop | 186.659 | 0.129 | 186.500 | 186.970 | 0.07% |
| large | forward | 123.925 | 0.097 | 123.799 | 124.143 | 0.08% |
| large | forward+backward | 394.928 | 0.195 | 394.602 | 395.270 | 0.05% |
| large | training-loop | 422.077 | 0.142 | 421.895 | 422.356 | 0.03% |

### warm-up = 5

| config | mode | mean (ms) | std (ms) | min (ms) | max (ms) | rel. std |
|---|---|---:|---:|---:|---:|---:|
| small | forward | 20.603 | 0.043 | 20.522 | 20.675 | 0.21% |
| small | forward+backward | 63.640 | 0.050 | 63.550 | 63.707 | 0.08% |
| small | training-loop | 68.399 | 0.117 | 68.318 | 68.738 | 0.17% |
| medium | forward | 55.841 | 0.034 | 55.791 | 55.888 | 0.06% |
| medium | forward+backward | 174.790 | 0.102 | 174.677 | 175.005 | 0.06% |
| medium | training-loop | 187.374 | 0.093 | 187.259 | 187.564 | 0.05% |
| large | forward | 124.389 | 0.089 | 124.258 | 124.547 | 0.07% |
| large | forward+backward | 396.608 | 0.170 | 396.418 | 397.020 | 0.04% |
| large | training-loop | 423.784 | 0.145 | 423.555 | 424.054 | 0.03% |

## Phase decomposition (warm-up = 5)

Backward and optimizer are obtained by subtraction, which is only valid because
the three modes are strictly nested -- the forward is identical in all three.

| config | forward | backward (derived) | optimizer (derived) | bwd/fwd |
|---|---:|---:|---:|---:|
| small | 20.60 | 43.04 | 4.76 | 2.09x |
| medium | 55.84 | 118.95 | 12.58 | 2.13x |
| large | 124.39 | 272.22 | 27.18 | 2.19x |

The measured bwd/fwd ratio of 2.09-2.19x matches the theoretical expectation of
~2x (backward computes gradients w.r.t. both inputs and weights, each roughly a
forward's worth of work). Optimizer cost tracks parameter count rather than
FLOPs, as expected for a bandwidth-bound elementwise update: 37.0, 29.7 and
28.0 ms per billion parameters respectively.

## Warm-up effect

### First measurement vs steady state (warm-up = 0)

| cell | sample[0] (ms) | steady min (ms) | ratio |
|---|---:|---:|---:|
| small / forward | 383.3 | 20.4 | **18.82x** |
| small / forward+backward | 114.2 | 63.2 | **1.81x** |
| small / training-loop | 72.8 | 67.5 | **1.08x** |
| medium / forward | 147.5 | 55.2 | **2.67x** |
| medium / forward+backward | 173.1 | 173.1 | **1.00x** |
| medium / training-loop | 187.7 | 185.4 | **1.01x** |
| large / forward | 158.4 | 123.4 | **1.28x** |
| large / forward+backward | 393.2 | 393.2 | **1.00x** |
| large / training-loop | 424.1 | 420.8 | **1.01x** |

Four of nine cells show essentially no warm-up effect. The one-time costs are
per-*kind-of-work*, not per-measurement, and are paid by whichever cell hits
them first:

- **383 ms, small/forward** -- first CUDA work in the process: context creation
  and cuBLAS handle initialisation. Happens once, ever.
- **114 ms, small/forward+backward** -- first backward: autograd engine setup
  and backward kernels.
- **72.8 ms, small/training-loop** -- first `optimizer.step()`: allocating
  AdamW's two moment buffers (fresh `cudaMalloc`, synchronous) and first launch
  of those kernels.
- **148 / 158 ms, medium and large forward** -- no context cost left, but new
  matmul shapes require fresh cuBLAS algorithm selection and allocator growth.
- **~1.00x everywhere else** -- nothing new left to initialise.

### Variability

Relative standard deviation, worst cell per run:

| warm-up | worst rel. std | cell |
|---|---:|---|
| 0 | 191.83% | small / forward |
| 1 | 0.44% | small / forward |
| 5 | 0.21% | small / forward |

One warm-up step retires effectively all of the one-time cost.

### warm-up = 1 vs warm-up = 5

| config | mode | w=1 mean (ms) | w=5 mean (ms) | delta |
|---|---|---:|---:|---:|
| small | forward | 20.702 | 20.603 | +0.48% |
| small | forward+backward | 63.706 | 63.640 | +0.10% |
| small | training-loop | 68.205 | 68.399 | -0.28% |
| medium | forward | 55.609 | 55.841 | -0.42% |
| medium | forward+backward | 174.031 | 174.790 | -0.43% |
| medium | training-loop | 186.659 | 187.374 | -0.38% |
| large | forward | 123.925 | 124.389 | -0.37% |
| large | forward+backward | 394.928 | 396.608 | -0.42% |
| large | training-loop | 422.077 | 423.784 | -0.40% |

Seven of nine cells are marginally *faster* with fewer warm-up steps. The
effect is ~0.4% and consistent in direction, which makes pure run-to-run noise
less likely; GPU clocks settling from boost to sustained frequency under
continued load is the usual explanation. It is small enough that it should not
be asserted without interleaving the two conditions or repeating the sweep.

## Caveats

1. **The w=0 run is not fully cold.** Model and optimizer are reused across
   modes, and modes run in the order forward -> forward+backward ->
   training-loop. So the forward kernels and memory pool are already warm by the
   time mode 2 starts. This is why the ratio column collapses to ~1.00x after
   the first cell of each config. A genuinely cold per-cell measurement needs a
   fresh process (or at minimum `--fresh-model`).
2. **Per-step samples are partial.** The pasted output truncated the `samples`
   column, so only the first one or two values per cell are recorded here.
   Summary statistics are complete.
3. **xl and 10B are missing** and are required for part (b). From the parameter
   accounting, 10B needs ~191 GB for the full training step and cannot run on a
   single 80 GB card; expect to report OOM for some cells.
4. **TF32 is off.** Achieved throughput is 26-33 TFLOP/s, i.e. 39-49% of the
   H100's 67 TFLOP/s non-tensor FP32 peak. Any mixed-precision comparison
   against this baseline will conflate "started using tensor cores" with
   "reduced precision helped".
