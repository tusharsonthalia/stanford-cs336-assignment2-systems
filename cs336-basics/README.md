# cs336-basics

Your own Assignment 1 implementation, vendored here so Assignment 2 can build on
it. The staff reference implementation that originally shipped in this directory
has been removed.

Module layout matches what the Assignment 2 test suite imports
(`from cs336_basics.model import Embedding, Linear, RMSNorm`):

| file | contents |
|---|---|
| `model.py` | `TransformerLM`, `TransformerBlock`, attention, `RoPE`, `SwiGLU`, `RMSNorm`, `Linear`, `Embedding` |
| `optimizer.py` | `AdamW`, `SGD` |
| `nn_utils.py` | `cross_entropy`, `cosine_learning_rate_schedule`, `gradient_clipping`, `save_checkpoint`, `load_checkpoint` |
| `data.py` | `get_batch` |
| `configs.py` | `DataConfig` / `ModelConfig` / `RunConfig` and the named runs |
| `train.py` | training loop |
| `plotting.py` | loss-curve plotting |

Upstream of this copy: `../../stanford-cs336-assignment1-basics`.
