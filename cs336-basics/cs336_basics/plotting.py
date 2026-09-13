import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(path: Path) -> list[dict]:
    """Read a metrics.jsonl written by the training loop."""
    if not Path(path).exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def plot_curve(metrics_path: Path, image_path: Path, title: str = "") -> None:
    """Write a train/validation loss curve from the metrics log."""
    rows = load_metrics(metrics_path)
    if not rows:
        return

    steps = [r["step"] for r in rows]
    figure, axes = plt.subplots(figsize=(8, 4.5))
    axes.plot(steps, [r["train_loss"] for r in rows], "#2a78d6", linewidth=2, label="train")
    axes.plot(steps, [r["validation_loss"] for r in rows], "#eb6834", linewidth=2, label="validation")
    axes.set_xlabel("step")
    axes.set_ylabel("cross-entropy")
    axes.set_title(title, loc="left")
    axes.legend(frameon=False)
    axes.grid(alpha=0.3)
    axes.spines[["top", "right"]].set_visible(False)

    figure.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    for directory in sys.argv[1:] or ["."]:
        run = Path(directory)
        plot_curve(run / "metrics.jsonl", run / "curve.png", run.name)
        print(f"  {run / 'curve.png'}")
