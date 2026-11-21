#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlp {

struct Layer {
  std::size_t in_dim = 0;
  std::size_t out_dim = 0;
  std::vector<float> weight;    // out_dim x in_dim, row-major (as stored in the file)
  std::vector<float> weight_t;  // in_dim x out_dim, transposed for the fast kernel
  std::vector<float> bias;      // out_dim
};

// Multi-layer perceptron with ReLU between hidden layers and a linear output.
// Weights are loaded from the binary format written by train.py.
class Network {
 public:
  // Throws std::runtime_error on bad magic, truncated file, or shape mismatch.
  static Network load(const std::string& path);

  // Forward pass for one flat input vector of size input_dim().
  // Returns logits (size output_dim()).
  std::vector<float> forward(const std::vector<float>& input) const;

  // Batched forward: input is row-major [n, input_dim()], output row-major
  // [n, output_dim()]. Reuses internal buffers across calls.
  std::vector<float> forward_batch(const float* input, std::size_t n) const;

  std::size_t input_dim() const { return layers_.front().in_dim; }
  std::size_t output_dim() const { return layers_.back().out_dim; }
  std::size_t num_layers() const { return layers_.size(); }
  std::size_t num_params() const;

 private:
  std::vector<Layer> layers_;
};

int argmax(const float* data, std::size_t n);

}  // namespace mlp
