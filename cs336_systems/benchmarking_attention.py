import torch
import pandas as pd
from cs336_systems.benchmarking_script import resolve_device
from cs336_basics.model import scaled_dot_product_attention
import argparse

torch._dynamo.config.cache_size_limit = 40

D_MODEL = [16,32,64,128]
SEQ_LEN = [256, 1024,  4096, 8192, 16384]
BATCH_SIZE = 8
STEP_COUNT = 100
WARMUP_COUNT = 5

def cuda_sync():
    torch.cuda.synchronize()

def bench_fwd(fn, Q, K, V, steps):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(steps):
        _ = fn(Q, K, V)
    end.record()
    cuda_sync()

    return start.elapsed_time(end) / steps
    
def bench_bwd(fn, Q, K, V, steps):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    total_ms = 0

    for _ in range(steps):
        cuda_sync()
        results = fn(Q,K,V)
        loss = results.sum()
        cuda_sync()
        
        start.record()
        loss.backward()
        end.record()

        Q.grad = K.grad = V.grad = None
        
        cuda_sync()
        
        total_ms += start.elapsed_time(end)

    return total_ms / steps

def bench_mem(fn, Q, K, V):
    Q.grad = K.grad = V.grad = None
    cuda_sync()
    torch.cuda.reset_peak_memory_stats()

    _ = fn(Q, K, V)
    cuda_sync()
    live = torch.cuda.memory_allocated() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return live, peak


def bench_atten(device: torch.device, compile=False):
    results = []
        
    for d_model in D_MODEL:
        for seq_len in SEQ_LEN:
            Q = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
            K = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)
            V = torch.randn(BATCH_SIZE, seq_len, d_model, device=device, requires_grad=True)

            for compile in [0, 1]:
                fn = scaled_dot_product_attention
                if compile:
                    fn = torch.compile(fn, dynamic=False)
                
                # warmup
                bench_fwd(fn, Q, K, V, WARMUP_COUNT)
                bench_bwd(fn, Q, K, V, WARMUP_COUNT)

                torch.cuda.empty_cache()

                mem_usage, peak_mb = bench_mem(fn, Q, K, V)

                fwd_ms = bench_fwd(fn, Q, K, V, STEP_COUNT)
                bwd_ms = bench_bwd(fn, Q, K, V, STEP_COUNT)

                result = {
                    "d_model": d_model,
                    "seq_len": seq_len,
                    "fwd_ms": fwd_ms,
                    "bwd_ms": bwd_ms,
                    "mem_usage_mb": mem_usage,
                    "peak_mb": peak_mb,
                    "compile": compile,
                    "warmup": WARMUP_COUNT,
                    "steps": STEP_COUNT,
                }
                
                results.append(result)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="profiles/pytorch_attention_bench.csv")

    args = parser.parse_args()

    device = resolve_device()
    results = bench_atten(device)

    results = pd.DataFrame(results)
    
    print(results)
    
    results.to_csv(args.out_dir, index=False)
