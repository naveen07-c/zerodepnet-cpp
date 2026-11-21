// Standalone inference benchmark for the exported MLP weights.
//
// Usage: mlp_bench <model.bin> <samples.bin> [warmup_reps] [timed_reps] [pred_dump.bin]
//
// samples.bin layout (little-endian): u32 n, u32 dim, f32 data[n*dim], u32 labels[n]
// Prints a single JSON object with aggregate timing and accuracy. If a fifth
// argument is given, the predicted class per sample is written there as u32[n].

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlp.hpp"

namespace {

struct Samples {
  std::uint32_t n = 0;
  std::uint32_t dim = 0;
  std::vector<float> data;
  std::vector<std::uint32_t> labels;
};

Samples load_samples(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open samples file: " + path);

  Samples s;
  in.read(reinterpret_cast<char*>(&s.n), 4);
  in.read(reinterpret_cast<char*>(&s.dim), 4);
  if (!in) throw std::runtime_error("truncated samples header");

  s.data.resize(static_cast<std::size_t>(s.n) * s.dim);
  in.read(reinterpret_cast<char*>(s.data.data()), static_cast<std::streamsize>(s.data.size() * 4));
  s.labels.resize(s.n);
  in.read(reinterpret_cast<char*>(s.labels.data()), static_cast<std::streamsize>(s.n * 4));
  if (!in) throw std::runtime_error("truncated samples body");
  return s;
}

double now_ms() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model.bin> <samples.bin> [warmup] [reps]\n", argv[0]);
    return 2;
  }
  const int warmup = argc > 3 ? std::stoi(argv[3]) : 3;
  const int reps = argc > 4 ? std::stoi(argv[4]) : 20;
  const char* pred_dump_path = argc > 5 ? argv[5] : nullptr;

  try {
    const mlp::Network net = mlp::Network::load(argv[1]);
    const Samples s = load_samples(argv[2]);
    if (s.dim != net.input_dim()) throw std::runtime_error("sample dim != model input dim");

    // Warmup (also validates the forward pass runs over every sample).
    for (int r = 0; r < warmup; ++r) {
      volatile float sink = net.forward_batch(s.data.data(), s.n).front();
      (void)sink;
    }

    // Per-sample latency: one forward() call per input, matching how a service
    // would score a single request. Averaged over `reps` passes of the batch.
    std::vector<float> single_in(net.input_dim());
    double single_total_ms = 0.0;
    for (int r = 0; r < reps; ++r) {
      const double s0 = now_ms();
      for (std::uint32_t i = 0; i < s.n; ++i) {
        std::copy(s.data.begin() + static_cast<std::size_t>(i) * s.dim,
                  s.data.begin() + static_cast<std::size_t>(i + 1) * s.dim,
                  single_in.begin());
        volatile float sink = net.forward(single_in).front();
        (void)sink;
      }
      single_total_ms += now_ms() - s0;
    }
    single_total_ms /= reps;

    const double start = now_ms();
    std::vector<float> logits;
    for (int r = 0; r < reps; ++r) {
      logits = net.forward_batch(s.data.data(), s.n);
    }
    const double end = now_ms();

    const double total_ms = (end - start) / reps;
    std::size_t correct = 0;
    std::vector<std::uint32_t> preds(s.n);
    for (std::uint32_t i = 0; i < s.n; ++i) {
      const int pred = mlp::argmax(
          logits.data() + static_cast<std::size_t>(i) * net.output_dim(), net.output_dim());
      preds[i] = static_cast<std::uint32_t>(pred);
      if (pred == static_cast<int>(s.labels[i])) ++correct;
    }

    if (pred_dump_path != nullptr) {
      std::ofstream dump(pred_dump_path, std::ios::binary);
      dump.write(reinterpret_cast<const char*>(preds.data()),
                 static_cast<std::streamsize>(preds.size() * sizeof(std::uint32_t)));
    }

    std::printf(
        "{\"engine\":\"cpp\",\"samples\":%u,\"reps\":%d,\"total_ms\":%.4f,"
        "\"per_sample_ms\":%.6f,\"samples_per_sec\":%.1f,\"accuracy\":%.4f,"
        "\"single_total_ms\":%.4f,\"single_per_sample_ms\":%.6f,"
        "\"params\":%zu,\"layers\":%zu,\"preds\":%zu}\n",
        s.n, reps, total_ms, total_ms / s.n, s.n * 1000.0 / total_ms,
        static_cast<double>(correct) / s.n,
        single_total_ms, single_total_ms / s.n,
        net.num_params(), net.num_layers(),
        pred_dump_path != nullptr ? preds.size() : std::size_t{0});
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
