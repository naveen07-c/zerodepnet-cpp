"""Train a small MLP from scratch on MNIST with an explicit train/val loop.

Outputs:
  artifacts/model.bin   - custom binary format consumed by the C++ engine
  artifacts/history.json - per-epoch train/val loss + accuracy
  artifacts/loss_curves.png
"""

import argparse
import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

import mnist

ARTIFACTS = Path("artifacts")


class MLP(nn.Module):
    """784 -> 128 -> 64 -> 10 with ReLU activations."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_loaders(batch_size: int, val_split: float, seed: int):
    train_full = TensorDataset(*mnist.load_split(Path("data"), "train"))
    test_set = TensorDataset(*mnist.load_split(Path("data"), "test"))

    n_val = int(len(train_full) * val_split)
    n_train = len(train_full) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(train_full, [n_train, n_val], generator=generator)

    # num_workers=0 keeps this portable (no fork issues in containers).
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(val_set, batch_size=batch_size, shuffle=False),
        DataLoader(test_set, batch_size=batch_size, shuffle=False),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x = x.view(x.size(0), -1)
        logits = model(x)
        total_loss += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    train_loader, val_loader, test_loader = build_loaders(
        args.batch_size, args.val_split, args.seed
    )

    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, n = 0.0, 0
        for x, y in train_loader:
            x = x.view(x.size(0), -1)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            n += y.size(0)

        train_loss = running_loss / n
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_acc {val_acc:.4f}"
        )

        # Early stopping on validation loss, with best-weight restore.
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"early stopping after epoch {epoch} (no val improvement)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val
    history["epochs_run"] = len(history["train_loss"])
    history["test_loss"], history["test_acc"] = evaluate(model, test_loader, criterion)
    print(f"test | loss {history['test_loss']:.4f} | acc {history['test_acc']:.4f}")
    return {"model": model, "history": history}


def write_weights(model: MLP, history: dict, out: Path) -> None:
    """Serialize linear layers to a compact binary: name-free, ordered by depth.

    Layout (little-endian):
      magic 'MLP1', u32 n_layers
      per layer: u32 in_dim, u32 out_dim, f32 bias[out_dim], f32 weight[out_dim][in_dim]
    Only Linear layers are stored; ReLU is implied between consecutive layers.
    """
    linear = [m for m in model.net if isinstance(m, nn.Linear)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(b"MLP1")
        f.write(struct.pack("<I", len(linear)))
        for layer in linear:
            w = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            out_dim, in_dim = w.shape
            f.write(struct.pack("<II", in_dim, out_dim))
            f.write(b.astype("<f4").tobytes())
            f.write(w.astype("<f4").tobytes())
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(linear)} layers)")

    meta = {
        "architecture": [layer.in_features for layer in linear] + [linear[-1].out_features],
        "activation": "relu",
        "test_acc": history["test_acc"],
        "test_loss": history["test_loss"],
        "history": {k: v for k, v in history.items() if isinstance(v, list)},
    }
    (out.parent / "model.json").write_text(json.dumps(meta, indent=2))


def plot_curves(history: dict, out: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_loss.plot(epochs, history["train_loss"], label="train loss")
    ax_loss.plot(epochs, history["val_loss"], label="val loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss curves")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["val_acc"], color="tab:green", label="val accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Validation accuracy")
    ax_acc.set_ylim(0.8, 1.0)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=ARTIFACTS)
    args = p.parse_args()

    result = train(args)
    write_weights(result["model"], result["history"], args.out_dir / "model.bin")
    plot_curves(result["history"], args.out_dir / "loss_curves.png")


if __name__ == "__main__":
    main()
