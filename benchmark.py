"""Benchmark PyTorch (Python) vs standalone C++ inference on 1,000 MNIST samples.

Steps:
  1. Export N test samples to artifacts/samples.bin (raw float32) for the C++ engine.
  2. Compile cpp/src/*.cpp with -O3.
  3. Time PyTorch forward passes and the C++ forward pass, verify accuracy parity.
  4. Write artifacts/benchmark.json and artifacts/benchmark.png.
"""

import json
import os
import platform
import shutil
import struct
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import mnist

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
BIN_DIR = ROOT / "build"
N_SAMPLES = 1000
WARMUP = 5
REPS = 20


def build_cpp() -> Path:
    BIN_DIR.mkdir(exist_ok=True)
    exe = BIN_DIR / "mlp_bench"
    sources = sorted(str(p) for p in (ROOT / "cpp" / "src").glob("*.cpp"))
    cmd = [
        "g++",
        "-std=c++17",
        "-O3",
        "-march=native",
        "-I",
        str(ROOT / "cpp" / "include"),
        *sources,
        "-o",
        str(exe),
    ]
    subprocess.run(cmd, check=True)
    return exe


def export_samples():
    xs, ys = mnist.load_split(ROOT / "data", "test")
    xs, ys = xs[:N_SAMPLES].contiguous(), ys[:N_SAMPLES].contiguous()

    path = ARTIFACTS / "samples.bin"
    with path.open("wb") as f:
        f.write(struct.pack("<II", N_SAMPLES, xs.shape[1]))
        f.write(xs.numpy().astype("<f4").tobytes())
        f.write(ys.numpy().astype("<u4").tobytes())
    return xs, ys, path


def load_model() -> nn.Module:
    """Rebuild the MLP architecture and load weights from the exported .bin."""
    with (ARTIFACTS / "model.bin").open("rb") as f:
        assert f.read(4) == b"MLP1"
        (n_layers,) = struct.unpack("<I", f.read(4))
        modules, dims = [], []
        for _ in range(n_layers):
            in_dim, out_dim = struct.unpack("<II", f.read(8))
            bias = np.frombuffer(f.read(out_dim * 4), dtype="<f4").copy()
            weight = np.frombuffer(f.read(out_dim * in_dim * 4), dtype="<f4").copy()
            linear = nn.Linear(in_dim, out_dim)
            with torch.no_grad():
                linear.weight.copy_(torch.from_numpy(weight.reshape(out_dim, in_dim)))
                linear.bias.copy_(torch.from_numpy(bias))
            modules.append(linear)
            modules.append(nn.ReLU())
            dims.append(in_dim)
        modules.pop()  # last activation is the identity (logits)
        model = nn.Sequential(*modules).eval()
    return model


def time_python(model: nn.Module, xs: torch.Tensor) -> dict:
    x = xs.clone()
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        t0 = time.perf_counter()
        for _ in range(REPS):
            logits = model(x)
        total_ms = (time.perf_counter() - t0) * 1000 / REPS
    preds = logits.argmax(dim=1)
    return {"total_ms": total_ms, "preds": preds}


def time_python_single(model: nn.Module, xs: torch.Tensor) -> float:
    """Per-sample loop: realistic latency when inputs arrive one at a time."""
    with torch.no_grad():
        for _ in range(3):
            for i in range(50):
                model(xs[i : i + 1])
        t0 = time.perf_counter()
        for i in range(N_SAMPLES):
            model(xs[i : i + 1])
        return (time.perf_counter() - t0) * 1000


def run_cpp(exe: Path, samples: Path) -> tuple[dict, np.ndarray]:
    dump = ARTIFACTS / "cpp_preds.bin"
    out = subprocess.run(
        [
            str(exe),
            str(ARTIFACTS / "model.bin"),
            str(samples),
            str(WARMUP),
            str(REPS),
            str(dump),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    preds = np.fromfile(dump, dtype="<u4")
    return json.loads(out.stdout), preds


def plot(results: dict, out: Path) -> None:
    labels = ["PyTorch\n1-by-1", "C++\n1-by-1", "PyTorch\nbatched", "C++\nbatched"]
    totals = [
        results["python_single"]["total_ms"],
        results["cpp"]["single_total_ms"],
        results["python_batched"]["total_ms"],
        results["cpp"]["total_ms"],
    ]
    throughput = [N_SAMPLES * 1000 / t for t in totals]

    fig, (ax_ms, ax_tp) = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = ["tab:orange", "#8c4a12", "tab:blue", "tab:red"]

    bars = ax_ms.bar(labels, totals, color=colors)
    ax_ms.set_ylabel("total ms for 1000 samples")
    ax_ms.set_title("Inference time (lower is better)")
    ax_ms.set_yscale("log")
    ax_ms.bar_label(bars, fmt="%.1f ms", padding=2)
    ax_ms.grid(alpha=0.3, axis="y")

    bars = ax_tp.bar(labels, throughput, color=colors)
    ax_tp.set_ylabel("samples / second")
    ax_tp.set_title("Throughput (higher is better)")
    ax_tp.bar_label(bars, fmt="%.0f", padding=2)
    ax_tp.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Python vs C++ inference — {results['device']}, {results['cpu']}")
    fig.tight_layout()
    fig.savefig(out, dpi=130)


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    if not (ARTIFACTS / "model.bin").exists():
        raise SystemExit("artifacts/model.bin missing — run train.py first")

    torch.set_num_threads(1)  # fair single-threaded comparison against the C++ build
    model = load_model()
    xs, ys, samples_path = export_samples()
    exe = build_cpp()
    print(f"built {exe}")

    py_batched = time_python(model, xs)
    py_single_ms = time_python_single(model, xs)
    cpp, cpp_preds = run_cpp(exe, samples_path)

    python_acc = (py_batched["preds"] == ys).float().mean().item()
    python_preds = py_batched["preds"].numpy()
    n_mismatch = int((python_preds != cpp_preds).sum())
    print(
        f"prediction parity: {N_SAMPLES - n_mismatch}/{N_SAMPLES} agree "
        f"({n_mismatch} mismatches)"
    )

    results = {
        "device": "cpu",
        "cpu": platform.processor() or platform.machine(),
        "n_samples": N_SAMPLES,
        "reps": REPS,
        "python_batched": {
            "total_ms": py_batched["total_ms"],
            "per_sample_ms": py_batched["total_ms"] / N_SAMPLES,
            "accuracy": python_acc,
        },
        "python_single": {
            "total_ms": py_single_ms,
            "per_sample_ms": py_single_ms / N_SAMPLES,
            "accuracy": python_acc,
        },
        "cpp": cpp,
        "prediction_mismatches": n_mismatch,
        "speedup_batched": py_batched["total_ms"] / cpp["total_ms"],
        "speedup_single": py_single_ms / cpp["single_total_ms"],
    }
    (ARTIFACTS / "benchmark.json").write_text(json.dumps(results, indent=2))
    plot(results, ARTIFACTS / "benchmark.png")

    print(f"\n{'engine':<20}{'total ms':>10}{'per-sample us':>16}{'acc':>8}")
    rows = [
        ("PyTorch 1-by-1", results["python_single"]["total_ms"], results["python_batched"]["accuracy"]),
        ("C++ 1-by-1", cpp["single_total_ms"], cpp["accuracy"]),
        ("PyTorch batched", results["python_batched"]["total_ms"], results["python_batched"]["accuracy"]),
        ("C++ batched", cpp["total_ms"], cpp["accuracy"]),
    ]
    for name, total_ms, acc in rows:
        print(f"{name:<20}{total_ms:>10.2f}{total_ms / N_SAMPLES * 1000:>16.2f}{acc:>8.4f}")
    print(
        f"\nsingle-sample speedup (C++ vs PyTorch): {results['speedup_single']:.2f}x\n"
        f"batched speedup     (C++ vs PyTorch): {results['speedup_batched']:.2f}x"
    )
    print(f"wrote {ARTIFACTS / 'benchmark.json'} and {ARTIFACTS / 'benchmark.png'}")


if __name__ == "__main__":
    main()
