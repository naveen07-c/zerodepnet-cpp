# zerodepnet-cpp

A small neural network trained in PyTorch, exported to a flat binary file, and run by a
standalone C++ inference engine with **no dependencies** (no BLAS, no Eigen, no OpenMP —
just the standard library). Includes a benchmark that compares inference latency against
PyTorch on the same 1,000 MNIST samples.

The point isn't to beat MKL. It's to show the whole path end to end: explicit train/val
loops, regularization, early stopping, a documented weight format, a hand-written forward
pass, and a measured, honest comparison.

## Results

MNIST MLP, 784 → 128 → 64 → 10, ReLU, trained 15 epochs (early stopped):

| metric | value |
| --- | --- |
| test accuracy | **0.9759** (9,759 / 10,000) |
| test loss | 0.0820 |
| best val loss | 0.0919 (epoch 11) |
| parameters | 109,386 |

Inference on 1,000 test samples, single-threaded, 20 timed reps after 5 warmup reps
(Intel Core 5 210H, `g++ -O3 -march=native`, torch 2.15.0 CPU build):

| engine | total (ms) | per sample (µs) | accuracy |
| --- | --- | --- | --- |
| PyTorch, 1 sample at a time | 21.94 | 21.94 | 0.9720 |
| C++, 1 sample at a time | **7.44** | **7.44** | 0.9720 |
| PyTorch, batched (1 × 1000) | **2.18** | 2.18 | 0.9720 |
| C++, batched (1 × 1000) | 3.00 | 3.00 | 0.9720 |

- Single-sample latency: C++ **2.95× faster** (PyTorch pays Python + dispatcher overhead per call).
- Batched throughput: PyTorch **1.38× faster** (MKL SGEMM is better tuned than the hand-written kernel).
- **Prediction parity: 1000/1000** samples produce the same argmax in both engines.

Both engines run with a single thread (`torch.set_num_threads(1)`), so the comparison is
core-for-core.

## So why not just use PyTorch?

Batched throughput is the one row where PyTorch wins, and it is also the row that matters
least for serving: a real request arrives one at a time, so per-call latency and process
cost dominate. Measured on the same machine:

| | PyTorch | this engine |
| --- | --- | --- |
| time to first prediction (fresh process: start, load model, run) | 1.4 s — and that is just `import torch` | **10–30 ms**, including loading the model and scoring 1,000 samples |
| peak RSS | ~300 MB | **~8 MB** |
| shipped artifact | ~2 GB wheel tree plus a Python runtime | **46 KB binary**, links only libstdc++ / libc / libm |
| single-sample latency | 21.9 µs | **7.4 µs** |
| batched throughput (1,000 samples) | **2.2 ms** | 3.0 ms |

```bash
/usr/bin/time -f "%e s  %M KB" ./build/mlp_bench artifacts/model.bin artifacts/samples.bin 0 1
/usr/bin/time -f "%e s  %M KB" .venv/bin/python -c "import torch, numpy, matplotlib"
```

The batched loss is expected rather than a defect: MKL SGEMM is decades of hand-tuned
assembly, and the kernel here is a few hundred lines of C++ that has to stay correct for
arbitrary layer sizes. Getting within 1.4× of it while shipping an 8 MB process is the
result worth reporting — not a claim that this beats a BLAS library at its own job.

## Training

`train.py` defines a plain `nn.Module` and an explicit loop — no `Lightning`, no trainer
abstraction:

- AdamW, `lr=1e-3`, `weight_decay=1e-4`, batch size 128 (all defaults are CLI flags)
- early stopping on validation loss with patience 4, restoring the best epoch's weights
- per-epoch train/val loss and val accuracy recorded, plotted to `artifacts/loss_curves.png`

`artifacts/model.json` holds the architecture, hyperparameters, final metrics, and the full
history so the run is reproducible from the artifacts alone.

MNIST is downloaded and parsed by `mnist.py`, a dependency-free IDX reader, so the project
does not need `torchvision`.

## Weight file format (`MLP1`)

Little-endian, no padding, no metadata beyond what is needed to rebuild the network:

```
magic       4 bytes   "MLP1"
n_layers    u32
per layer:
  in_dim    u32
  out_dim   u32
  bias      f32[out_dim]
  weight    f32[out_dim][in_dim]    row-major: weight[o][i]
```

The C++ loader validates the magic, rejects zero-sized layers, and checks that each layer's
`in_dim` matches the previous layer's `out_dim`. `model.bin` for this network is 437,576 bytes.

Activations between layers are ReLU; the output layer is linear.

## C++ engine

```
cpp/include/mlp.hpp   Layer, Network (load / forward / forward_batch), argmax
cpp/src/mlp.cpp       loader + kernels
cpp/src/main.cpp      CLI benchmark harness, prints one JSON line
```

Two kernels, because the two access patterns want different things:

- **Single sample** — weights are stored transposed (`[in_dim][out_dim]`) at load time, so
  `out[o] += x[i] * w_t[i][o]` walks contiguous memory and vectorizes. One sample is
  bandwidth-bound on the 401 KB weight matrix, which is why it lands at ~7 µs.
- **Batch** — an 8-sample × 8-output tile is accumulated in registers, so each weight vector
  load feeds 64 FMAs. This is what took the batched path from 28.8 ms to 3.0 ms. Samples and
  outputs that don't fill a tile fall back to a scalar dot-product path, which keeps arbitrary
  dimensions correct (the 10-wide output layer uses it for its last two columns).

`-ffast-math` was deliberately not used: it would let the compiler reassociate the dot
products and change the numerics that the parity check depends on. Without it GCC also won't
contract `a += b * c` into an FMA inside a reduction, which is part of the remaining gap to
MKL.

## Build and run

```bash
python -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

# train + export weights and loss curves
.venv/bin/python train.py --epochs 30 --patience 4

# compile the engine (no external libs)
mkdir -p build
g++ -std=c++17 -O3 -march=native -Wall -Wextra -I cpp/include cpp/src/*.cpp -o build/mlp_bench

# run it directly: model, samples, warmup, reps, optional prediction dump
./build/mlp_bench artifacts/model.bin artifacts/samples.bin 5 20 artifacts/cpp_preds.bin

# full comparison: builds the engine, exports samples, times both, checks parity
.venv/bin/python benchmark.py
```

`benchmark.py` writes `artifacts/benchmark.json` and `artifacts/benchmark.png`.

## Limitations

- Fixed MLP topology with ReLU hidden layers; no convolutions, batches of one layer at a time.
- The batch kernel is tuned for wide, contiguous layers. It is ~15% off the peaks I measured
  in isolation (~85 GFLOP/s in-engine), and MKL is still faster at this size.
- Timing is single-run wall clock with warmup, not a statistical benchmark; expect a few
  percent of run-to-run noise.
- No threading in the C++ engine, so it doesn't use the other cores the CPU has.
