#include <metal_stdlib>
using namespace metal;

struct MatrixParams {
  uint m;
  uint n;
  uint k;
  uint _pad;
};
struct NormParams {
  uint rows;
  uint width;
  float epsilon;
  uint _pad;
};
struct RopeParams {
  uint tokens;
  uint heads;
  uint head_dim;
  uint offset;
  float theta;
  uint _pad0;
  uint _pad1;
  uint _pad2;
};
struct AttentionParams {
  uint tokens;
  uint q_heads;
  uint kv_heads;
  uint head_dim;
  uint causal;
  uint query_offset;
  uint kv_length;
  uint _pad;
};

kernel void add_f16(device const half *a [[buffer(0)]],
                    device const half *b [[buffer(1)]],
                    device half *out [[buffer(2)]],
                    constant uint &count [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
  if (id < count)
    out[id] = a[id] + b[id];
}

kernel void matmul_f16(device const half *x [[buffer(0)]],
                       device const half *weight [[buffer(1)]],
                       device half *out [[buffer(2)]],
                       constant MatrixParams &p [[buffer(3)]],
                       uint2 id [[thread_position_in_grid]],
                       uint2 local [[thread_position_in_threadgroup]]) {
  threadgroup half x_tile[16][16];
  threadgroup half weight_tile[16][16];
  float sum = 0.0f;
  for (uint base = 0; base < p.k; base += 16) {
    uint inner = base + local.x;
    x_tile[local.y][local.x] =
        id.y < p.m && inner < p.k ? x[id.y * p.k + inner] : half(0.0f);
    uint tile_column = uint(id.x - local.x) + local.y;
    weight_tile[local.y][local.x] = tile_column < p.n && inner < p.k
                                        ? weight[tile_column * p.k + inner]
                                        : half(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 16; ++i)
      sum += float(x_tile[local.y][i]) * float(weight_tile[local.x][i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (id.x < p.n && id.y < p.m)
    out[id.y * p.n + id.x] = half(sum);
}

kernel void matvec_f16(device const half *x [[buffer(0)]],
                       device const half *weight [[buffer(1)]],
                       device half *out [[buffer(2)]],
                       constant MatrixParams &p [[buffer(3)]],
                       uint column [[thread_position_in_grid]]) {
  if (column >= p.n)
    return;
  float sum = 0.0f;
  for (uint i = 0; i < p.k; ++i)
    sum += float(x[i]) * float(weight[column * p.k + i]);
  out[column] = half(sum);
}

kernel void rms_norm_f16(device const half *x [[buffer(0)]],
                         device const half *weight [[buffer(1)]],
                         device half *out [[buffer(2)]],
                         constant NormParams &p [[buffer(3)]],
                         uint row [[thread_position_in_grid]]) {
  if (row >= p.rows)
    return;
  float sum = 0.0f;
  uint base = row * p.width;
  for (uint i = 0; i < p.width; ++i) {
    float v = float(x[base + i]);
    sum += v * v;
  }
  float scale = rsqrt(sum / float(p.width) + p.epsilon);
  for (uint i = 0; i < p.width; ++i)
    out[base + i] = half(float(x[base + i]) * scale * float(weight[i]));
}

kernel void swiglu_f16(device const half *gate [[buffer(0)]],
                       device const half *up [[buffer(1)]],
                       device half *out [[buffer(2)]],
                       constant uint &count [[buffer(3)]],
                       uint id [[thread_position_in_grid]]) {
  if (id >= count)
    return;
  float x = float(gate[id]);
  out[id] = half((x / (1.0f + exp(-x))) * float(up[id]));
}

kernel void embedding_f16(device const uint *tokens [[buffer(0)]],
                          device const half *table [[buffer(1)]],
                          device half *out [[buffer(2)]],
                          constant uint2 &p [[buffer(3)]],
                          uint2 id [[thread_position_in_grid]]) {
  if (id.x >= p.y || id.y >= p.x)
    return;
  out[id.y * p.y + id.x] = table[tokens[id.y] * p.y + id.x];
}

kernel void rope_f16(device const half *input [[buffer(0)]],
                     device half *out [[buffer(1)]],
                     constant RopeParams &p [[buffer(2)]],
                     uint3 id [[thread_position_in_grid]]) {
  uint pair = id.x;
  uint head = id.y;
  uint token = id.z;
  if (pair >= p.head_dim / 2 || head >= p.heads || token >= p.tokens)
    return;
  uint base = (token * p.heads + head) * p.head_dim;
  uint second = pair + p.head_dim / 2;
  float frequency = pow(p.theta, -float(pair * 2) / float(p.head_dim));
  float angle = float(token + p.offset) * frequency;
  float x = float(input[base + pair]);
  float y = float(input[base + second]);
  out[base + pair] = half(x * cos(angle) - y * sin(angle));
  out[base + second] = half(x * sin(angle) + y * cos(angle));
}

kernel void attention_reference_f16(device const half *q [[buffer(0)]],
                                    device const half *k [[buffer(1)]],
                                    device const half *v [[buffer(2)]],
                                    device half *out [[buffer(3)]],
                                    constant AttentionParams &p [[buffer(4)]],
                                    uint3 id [[thread_position_in_grid]]) {
  uint d = id.x, h = id.y, qi = id.z;
  if (d >= p.head_dim || h >= p.q_heads || qi >= p.tokens)
    return;
  uint kvh = h / (p.q_heads / p.kv_heads);
  uint available =
      p.causal != 0 ? min(p.kv_length, p.query_offset + qi + 1) : p.kv_length;
  float maximum = -INFINITY;
  float scale = rsqrt(float(p.head_dim));
  for (uint j = 0; j < available; ++j) {
    float score = 0.0f;
    for (uint x = 0; x < p.head_dim; ++x)
      score += float(q[(qi * p.q_heads + h) * p.head_dim + x]) *
               float(k[(j * p.kv_heads + kvh) * p.head_dim + x]);
    maximum = max(maximum, score * scale);
  }
  float denominator = 0.0f, numerator = 0.0f;
  for (uint j = 0; j < available; ++j) {
    float score = 0.0f;
    for (uint x = 0; x < p.head_dim; ++x)
      score += float(q[(qi * p.q_heads + h) * p.head_dim + x]) *
               float(k[(j * p.kv_heads + kvh) * p.head_dim + x]);
    float probability = exp(score * scale - maximum);
    denominator += probability;
    numerator +=
        probability * float(v[(j * p.kv_heads + kvh) * p.head_dim + d]);
  }
  out[(qi * p.q_heads + h) * p.head_dim + d] = half(numerator / denominator);
}

kernel void attention_tiled_f16(device const half *q [[buffer(0)]],
                                device const half *k [[buffer(1)]],
                                device const half *v [[buffer(2)]],
                                device half *out [[buffer(3)]],
                                constant AttentionParams &p [[buffer(4)]],
                                uint3 id [[thread_position_in_grid]],
                                uint lane [[thread_index_in_threadgroup]],
                                uint3 group_size [[threads_per_threadgroup]]) {
  uint d = id.x, h = id.y, qi = id.z;
  if (d >= p.head_dim || h >= p.q_heads || qi >= p.tokens)
    return;
  uint kvh = h / (p.q_heads / p.kv_heads);
  uint available =
      p.causal != 0 ? min(p.kv_length, p.query_offset + qi + 1) : p.kv_length;
  threadgroup float partial[256];
  threadgroup float coefficients[3];
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float accumulator = 0.0f;
  float scale = rsqrt(float(p.head_dim));
  for (uint j = 0; j < available; ++j) {
    partial[lane] = float(q[(qi * p.q_heads + h) * p.head_dim + d]) *
                    float(k[(j * p.kv_heads + kvh) * p.head_dim + d]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size.x / 2; stride > 0; stride >>= 1) {
      if (lane < stride)
        partial[lane] += partial[lane + stride];
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0) {
      float score = partial[0] * scale;
      float next_max = max(running_max, score);
      coefficients[0] = exp(running_max - next_max);
      coefficients[1] = exp(score - next_max);
      coefficients[2] = next_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    accumulator =
        accumulator * coefficients[0] +
        coefficients[1] * float(v[(j * p.kv_heads + kvh) * p.head_dim + d]);
    running_sum = running_sum * coefficients[0] + coefficients[1];
    running_max = coefficients[2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  out[(qi * p.q_heads + h) * p.head_dim + d] = half(accumulator / running_sum);
}

kernel void copy_kv_f16(device const half *source [[buffer(0)]],
                        device half *cache [[buffer(1)]],
                        constant uint3 &p [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
  uint count = p.x * p.z;
  if (id < count)
    cache[p.y * p.z + id] = source[id];
}
