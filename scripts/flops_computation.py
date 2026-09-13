from typing import NamedTuple

from rich.console import Console
from rich.table import Table


class Op(NamedTuple):
    """One operation in a forward pass."""
    path: str                      # "/"-separated, gives the report its hierarchy
    params: int                    # trainable parameters (0 for buffers/activations)
    flops: int                     # per occurrence
    frequency: int = 1                 # occurrences (num_layers for per-block ops)
    kind: str = "matmul"           # "matmul" | "elementwise"


def transformer_lm(vocab_size, context_length, num_layers, d_model, num_heads, d_ff):
    """Enumerate every operation in one forward pass over `context_length` tokens."""
    D, C, L, F, V, H = d_model, context_length, num_layers, d_ff, vocab_size, num_heads
    E = "elementwise"
    return [
        # token embedding is a row gather, not a matmul
        Op("embedding",          D * V,         0,                 1),

        # --- attention sublayer ------------------------------------------------
        Op("block/ln1",          D,             3 * C * D,         L, E),
        Op("block/attn/proj",    4 * D * D,     8 * C * D * D,     L),
        Op("block/attn/rope",    0,             6 * C * D,         L, E),
        Op("block/attn/QK^T",    0,             2 * C * C * D,     L),
        Op("block/attn/scale",   0,             C * C * H,         L, E),
        Op("block/attn/softmax", 0,             4 * C * C * H,     L, E),
        Op("block/attn/A@V",     0,             2 * C * C * D,     L),
        Op("block/residual1",    0,             C * D,             L, E),

        # --- feed-forward sublayer ---------------------------------------------
        Op("block/ln2",          D,             3 * C * D,         L, E),
        Op("block/ffn/w1",       D * F,         2 * C * D * F,     L),
        Op("block/ffn/w3",       D * F,         2 * C * D * F,     L),
        Op("block/ffn/silu",     0,             2 * C * F,         L, E),
        Op("block/ffn/gate",     0,             C * F,             L, E),
        Op("block/ffn/w2",       F * D,         2 * C * F * D,     L),
        Op("block/residual2",    0,             C * D,             L, E),

        # --- output head --------------------------------------------------------
        Op("ln_final",           D,             3 * C * D,         1, E),
        Op("lm_head",            D * V,         2 * C * D * V,     1),
    ]


def aggregate(ops):
    """Roll ops up into every prefix of their path. Insertion order is already
    hierarchical, since an ancestor is always inserted before its first child."""
    agg = {}
    for op in ops:
        parts = op.path.split("/")
        for i in range(1, len(parts) + 1):
            row = agg.setdefault(tuple(parts[:i]), [0, 0])
            row[0] += op.params * op.frequency
            row[1] += op.flops * op.frequency
    return agg


def report(ops, title=""):
    """Print the full hierarchical breakdown for one model."""
    agg = aggregate(ops)
    tp, tf = (sum(v[i] for k, v in agg.items() if len(k) == 1) for i in (0, 1))
    mm = sum(o.flops * o.frequency for o in ops if o.kind == "matmul")

    table = Table(title=title, title_style="bold cyan", header_style="bold")
    table.add_column("component")
    for col in ("params", "% p", "flops", "% f"):
        table.add_column(col, justify="right")

    for key, (p, f) in agg.items():
        table.add_row("  " * (len(key) - 1) + key[-1],
                      f"{p:,}", f"{100 * p / tp:.1f}%",
                      f"{f:,}", f"{100 * f / tf:.1f}%",
                      style="dim" if len(key) > 1 else None)

    table.add_section()
    table.add_row("TOTAL", f"{tp:,}", "100.0%", f"{tf:,}", "100.0%", style="bold")
    table.add_row("  of which matmuls", "", "", f"{mm:,}", f"{100 * mm / tf:.1f}%", style="dim")
    Console().print(table)
    Console().print(f"  [dim]{tp / 1e9:.3f}B params · {tp * 4 / 1e9:.2f} GB fp32 · "
                    f"{mm / 1e12:.3f} TFLOPs/forward (matmuls)[/]\n")


if __name__ == "__main__":
    VOCAB, CONTEXT = 50_257, 1_024

    # (num_layers, d_model, num_heads, d_ff)
    # d_ff = 8/3 * d_model rounded to a multiple of 64.
    #    768 -> 2048    1024 -> 2752    1280 -> 3392    1600 -> 4288
    CONFIGS = {
        "GPT-2 small":  (12, 768, 12, 2048),
        "GPT-2 medium": (24, 1024, 16, 2752),
        "GPT-2 large":  (36, 1280, 20, 3392),
        "GPT-2 XL":     (48, 1600, 25, 4288),
    }

    models = {
        name: transformer_lm(VOCAB, CONTEXT, L, D, H, F)
        for name, (L, D, H, F) in CONFIGS.items()
    }

    # (a)-(c): full breakdown of GPT-2 XL
    report(models["GPT-2 XL"], "GPT-2 XL  (context 1,024)")