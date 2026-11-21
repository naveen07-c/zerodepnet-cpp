#include "mlp.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace mlp {

namespace {

std::uint32_t read_u32(std::ifstream& in) {
  std::uint32_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) throw std::runtime_error("unexpected end of model file");
  return value;
}

template <typename T>
void read_vec(std::ifstream& in, std::vector<T>& out, std::size_t n) {
  out.resize(n);
  in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n * sizeof(T)));
  if (!in) throw std::runtime_error("unexpected end of model file");
}

inline float activate(float v, bool relu) { return relu ? std::max(v, 0.0f) : v; }

void write_value(float* dst, std::size_t stride, std::size_t s, std::size_t o, float v,
                 bool relu) {
  dst[s * stride + o] = activate(v, relu);
}

// Single row: walk the transposed weight rows so the accumulator loop is
// contiguous and vectorizes (this path is memory-bound on the weight matrix).
void compute_row(const Layer& layer, const float* src_row, float* dst_row, bool relu) {
  const std::size_t in_dim = layer.in_dim;
  const std::size_t out_dim = layer.out_dim;
  const float* wt = layer.weight_t.data();

  std::copy(layer.bias.begin(), layer.bias.end(), dst_row);
  for (std::size_t i = 0; i < in_dim; ++i) {
    const float xi = src_row[i];
    const float* row = wt + i * out_dim;
    for (std::size_t o = 0; o < out_dim; ++o) dst_row[o] += xi * row[o];
  }
  if (relu) {
    for (std::size_t o = 0; o < out_dim; ++o) dst_row[o] = std::max(dst_row[o], 0.0f);
  }
}

// Fast path: accumulate an 8-sample x 8-output tile in registers.
// Weights are stored transposed ([in_dim][out_dim]) so the inner loop walks
// contiguous memory and each weight vector load feeds eight FMAs.
void compute_tile(const Layer& layer, const float* src, std::size_t src_stride, float* dst,
                  std::size_t dst_stride, std::size_t s0, std::size_t o0, bool relu) {
  const std::size_t in_dim = layer.in_dim;
  const std::size_t out_dim = layer.out_dim;
  const float* wt = layer.weight_t.data();

  float acc[8][8];
  for (std::size_t k = 0; k < 8; ++k) {
    for (std::size_t j = 0; j < 8; ++j) acc[j][k] = layer.bias[o0 + k];
  }

  for (std::size_t i = 0; i < in_dim; ++i) {
    const float* row = wt + i * out_dim + o0;
    for (std::size_t k = 0; k < 8; ++k) {
      const float rk = row[k];
      for (std::size_t j = 0; j < 8; ++j) {
        acc[j][k] += src[(s0 + j) * src_stride + i] * rk;
      }
    }
  }

  for (std::size_t j = 0; j < 8; ++j) {
    for (std::size_t k = 0; k < 8; ++k) {
      write_value(dst, dst_stride, s0 + j, o0 + k, acc[j][k], relu);
    }
  }
}

// One layer: n rows in, n rows out.
void compute_layer(const Layer& layer, const float* src, std::size_t n, std::size_t src_stride,
                   float* dst, std::size_t dst_stride, bool relu) {
  constexpr std::size_t kBlock = 8;
  const std::size_t n_main = n - (n % kBlock);
  const std::size_t o_main = layer.out_dim - (layer.out_dim % kBlock);

  for (std::size_t s0 = 0; s0 < n_main; s0 += kBlock) {
    for (std::size_t o0 = 0; o0 < o_main; o0 += kBlock) {
      compute_tile(layer, src, src_stride, dst, dst_stride, s0, o0, relu);
    }
    // Output columns that do not fill a full tile.
    for (std::size_t o = o_main; o < layer.out_dim; ++o) {
      const float* w = layer.weight.data() + o * layer.in_dim;
      for (std::size_t j = 0; j < kBlock; ++j) {
        const float* x = src + (s0 + j) * src_stride;
        float acc = layer.bias[o];
        for (std::size_t i = 0; i < layer.in_dim; ++i) acc += w[i] * x[i];
        dst[(s0 + j) * dst_stride + o] = activate(acc, relu);
      }
    }
  }

  // Leftover samples (fewer than eight).
  for (std::size_t s = n_main; s < n; ++s) {
    compute_row(layer, src + s * src_stride, dst + s * dst_stride, relu);
  }
}

}  // namespace

Network Network::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open model file: " + path);

  char magic[4] = {};
  in.read(magic, 4);
  if (!in || std::memcmp(magic, "MLP1", 4) != 0) {
    throw std::runtime_error("bad magic in model file: " + path);
  }

  Network net;
  const std::uint32_t n_layers = read_u32(in);
  if (n_layers == 0) throw std::runtime_error("model has no layers");

  net.layers_.resize(n_layers);
  for (std::uint32_t l = 0; l < n_layers; ++l) {
    Layer& layer = net.layers_[l];
    layer.in_dim = read_u32(in);
    layer.out_dim = read_u32(in);
    if (layer.in_dim == 0 || layer.out_dim == 0) {
      throw std::runtime_error("zero-sized layer in model file");
    }
    read_vec(in, layer.bias, layer.out_dim);
    read_vec(in, layer.weight, layer.out_dim * layer.in_dim);

    layer.weight_t.resize(layer.weight.size());
    for (std::size_t o = 0; o < layer.out_dim; ++o) {
      for (std::size_t i = 0; i < layer.in_dim; ++i) {
        layer.weight_t[i * layer.out_dim + o] = layer.weight[o * layer.in_dim + i];
      }
    }
  }

  for (std::size_t i = 1; i < net.layers_.size(); ++i) {
    if (net.layers_[i].in_dim != net.layers_[i - 1].out_dim) {
      throw std::runtime_error("layer dimension mismatch in model file");
    }
  }
  return net;
}

std::vector<float> Network::forward(const std::vector<float>& input) const {
  if (input.size() != input_dim()) {
    throw std::runtime_error("input size does not match model input_dim");
  }
  std::vector<float> output(output_dim());
  if (layers_.size() == 1) {
    compute_row(layers_[0], input.data(), output.data(), false);
    return output;
  }

  std::size_t max_dim = 0;
  for (const Layer& layer : layers_) max_dim = std::max(max_dim, layer.out_dim);
  std::vector<float> buf_a(max_dim);
  std::vector<float> buf_b(max_dim);
  const float* src = input.data();
  float* dst = buf_a.data();

  for (std::size_t li = 0; li < layers_.size(); ++li) {
    const Layer& layer = layers_[li];
    const bool is_last = li + 1 == layers_.size();
    compute_layer(layer, src, 1, layer.in_dim, is_last ? output.data() : dst, layer.out_dim,
                  !is_last);
    if (is_last) break;
    src = dst;
    dst = (dst == buf_a.data()) ? buf_b.data() : buf_a.data();
  }
  return output;
}

std::vector<float> Network::forward_batch(const float* input, std::size_t n) const {
  if (n == 0) return {};
  std::vector<float> output(n * output_dim());
  if (layers_.size() == 1) {
    compute_layer(layers_[0], input, n, layers_[0].in_dim, output.data(), layers_[0].out_dim,
                  false);
    return output;
  }

  std::size_t max_dim = 0;
  for (const Layer& layer : layers_) max_dim = std::max(max_dim, layer.out_dim);
  std::vector<float> buf_a(n * max_dim);
  std::vector<float> buf_b(n * max_dim);
  const float* src = input;
  float* dst = buf_a.data();

  for (std::size_t li = 0; li < layers_.size(); ++li) {
    const Layer& layer = layers_[li];
    const bool is_last = li + 1 == layers_.size();
    compute_layer(layer, src, n, layer.in_dim, is_last ? output.data() : dst, layer.out_dim,
                  !is_last);
    if (is_last) break;
    src = dst;
    dst = (dst == buf_a.data()) ? buf_b.data() : buf_a.data();
  }
  return output;
}

std::size_t Network::num_params() const {
  std::size_t total = 0;
  for (const Layer& layer : layers_) total += layer.weight.size() + layer.bias.size();
  return total;
}

int argmax(const float* data, std::size_t n) {
  return static_cast<int>(std::max_element(data, data + n) - data);
}

}  // namespace mlp
