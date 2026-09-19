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
struct MultiMatrixParams {
  uint n0;
  uint n1;
  uint n2;
  uint k;
};
struct QkTransformParams {
  uint tokens;
  uint q_heads;
  uint kv_heads;
  uint head_dim;
  uint offset;
  float theta;
  uint cache_capacity;
  float epsilon;
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
                       uint group [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simdgroup_index
                       [[simdgroup_index_in_threadgroup]]) {
  constexpr uint rows_per_simdgroup = 4;
  constexpr uint simdgroups_per_threadgroup = 8;
  uint first_row = (group * simdgroups_per_threadgroup + simdgroup_index) *
                   rows_per_simdgroup;
  float sums[rows_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};
  if ((p.k & 3) == 0) {
    device const half4 *x4 = reinterpret_cast<device const half4 *>(x);
    uint vectors = p.k / 4;
    for (uint i = lane; i < vectors; i += 32) {
      float4 input = float4(x4[i]);
      for (uint output = 0; output < rows_per_simdgroup; ++output) {
        uint row = first_row + output;
        if (row < p.n) {
          device const half4 *weight4 =
              reinterpret_cast<device const half4 *>(weight + row * p.k);
          sums[output] += dot(input, float4(weight4[i]));
        }
      }
    }
  } else {
    for (uint i = lane; i < p.k; i += 32) {
      float input = float(x[i]);
      for (uint output = 0; output < rows_per_simdgroup; ++output) {
        uint row = first_row + output;
        if (row < p.n)
          sums[output] += input * float(weight[row * p.k + i]);
      }
    }
  }
  for (uint output = 0; output < rows_per_simdgroup; ++output) {
    float total = simd_sum(sums[output]);
    uint row = first_row + output;
    if (lane == 0 && row < p.n)
      out[row] = half(total);
  }
}

kernel void matvec2_f16(device const half *x [[buffer(0)]],
                        device const half *weight0 [[buffer(1)]],
                        device const half *weight1 [[buffer(2)]],
                        device half *out0 [[buffer(3)]],
                        device half *out1 [[buffer(4)]],
                        constant MultiMatrixParams &p [[buffer(5)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simdgroup_index
                        [[simdgroup_index_in_threadgroup]]) {
  constexpr uint rows_per_simdgroup = 4;
  constexpr uint simdgroups_per_threadgroup = 8;
  uint first_row = (group * simdgroups_per_threadgroup + simdgroup_index) *
                   rows_per_simdgroup;
  float sums0[rows_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};
  float sums1[rows_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};
  if ((p.k & 3) == 0) {
    device const half4 *x4 = reinterpret_cast<device const half4 *>(x);
    uint vectors = p.k / 4;
    for (uint i = lane; i < vectors; i += 32) {
      float4 input = float4(x4[i]);
      for (uint output = 0; output < rows_per_simdgroup; ++output) {
        uint row = first_row + output;
        if (row < p.n0) {
          device const half4 *weight4 =
              reinterpret_cast<device const half4 *>(weight0 + row * p.k);
          sums0[output] += dot(input, float4(weight4[i]));
        }
        if (row < p.n1) {
          device const half4 *weight4 =
              reinterpret_cast<device const half4 *>(weight1 + row * p.k);
          sums1[output] += dot(input, float4(weight4[i]));
        }
      }
    }
  } else {
    for (uint i = lane; i < p.k; i += 32) {
      float input = float(x[i]);
      for (uint output = 0; output < rows_per_simdgroup; ++output) {
        uint row = first_row + output;
        if (row < p.n0)
          sums0[output] += input * float(weight0[row * p.k + i]);
        if (row < p.n1)
          sums1[output] += input * float(weight1[row * p.k + i]);
      }
    }
  }
  for (uint output = 0; output < rows_per_simdgroup; ++output) {
    float total0 = simd_sum(sums0[output]);
    float total1 = simd_sum(sums1[output]);
    uint row = first_row + output;
    if (lane == 0) {
      if (row < p.n0)
        out0[row] = half(total0);
      if (row < p.n1)
        out1[row] = half(total1);
    }
  }
}

kernel void matvec3_f16(device const half *x [[buffer(0)]],
                        device const half *weight0 [[buffer(1)]],
                        device const half *weight1 [[buffer(2)]],
                        device const half *weight2 [[buffer(3)]],
                        device half *out0 [[buffer(4)]],
                        device half *out1 [[buffer(5)]],
                        device half *out2 [[buffer(6)]],
                        constant MultiMatrixParams &p [[buffer(7)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simdgroup_index
                        [[simdgroup_index_in_threadgroup]]) {
  constexpr uint rows_per_simdgroup = 4;
  constexpr uint simdgroups_per_threadgroup = 8;
  uint first_row = (group * simdgroups_per_threadgroup + simdgroup_index) *
                   rows_per_simdgroup;
  float sums0[rows_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};
  float sums1[rows_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};
  float sums2[rows_per_simdgroup] = {0.0f, 0.0f, 0.0f, 0.0f};
  if ((p.k & 3) == 0) {
    device const half4 *x4 = reinterpret_cast<device const half4 *>(x);
    uint vectors = p.k / 4;
    for (uint i = lane; i < vectors; i += 32) {
      float4 input = float4(x4[i]);
      for (uint output = 0; output < rows_per_simdgroup; ++output) {
        uint row = first_row + output;
        if (row < p.n0) {
          device const half4 *weight4 =
              reinterpret_cast<device const half4 *>(weight0 + row * p.k);
          sums0[output] += dot(input, float4(weight4[i]));
        }
        if (row < p.n1) {
          device const half4 *weight4 =
              reinterpret_cast<device const half4 *>(weight1 + row * p.k);
          sums1[output] += dot(input, float4(weight4[i]));
        }
        if (row < p.n2) {
          device const half4 *weight4 =
              reinterpret_cast<device const half4 *>(weight2 + row * p.k);
          sums2[output] += dot(input, float4(weight4[i]));
        }
      }
    }
  } else {
    for (uint i = lane; i < p.k; i += 32) {
      float input = float(x[i]);
      for (uint output = 0; output < rows_per_simdgroup; ++output) {
        uint row = first_row + output;
        if (row < p.n0)
          sums0[output] += input * float(weight0[row * p.k + i]);
        if (row < p.n1)
          sums1[output] += input * float(weight1[row * p.k + i]);
        if (row < p.n2)
          sums2[output] += input * float(weight2[row * p.k + i]);
      }
    }
  }
  for (uint output = 0; output < rows_per_simdgroup; ++output) {
    float total0 = simd_sum(sums0[output]);
    float total1 = simd_sum(sums1[output]);
    float total2 = simd_sum(sums2[output]);
    uint row = first_row + output;
    if (lane == 0) {
      if (row < p.n0)
        out0[row] = half(total0);
      if (row < p.n1)
        out1[row] = half(total1);
      if (row < p.n2)
        out2[row] = half(total2);
    }
  }
}

kernel void rms_norm_f16(device const half *x [[buffer(0)]],
                         device const half *weight [[buffer(1)]],
                         device half *out [[buffer(2)]],
                         constant NormParams &p [[buffer(3)]],
                         uint group [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simdgroup_index
                         [[simdgroup_index_in_threadgroup]]) {
  constexpr uint simdgroups_per_threadgroup = 8;
  uint row = group * simdgroups_per_threadgroup + simdgroup_index;
  if (row >= p.rows)
    return;
  float sum = 0.0f;
  uint base = row * p.width;
  for (uint i = lane; i < p.width; i += 32) {
    float v = float(x[base + i]);
    sum += v * v;
  }
  sum = simd_sum(sum);
  float scale = rsqrt(sum / float(p.width) + p.epsilon);
  for (uint i = lane; i < p.width; i += 32)
    out[base + i] = half(float(x[base + i]) * scale * float(weight[i]));
}

kernel void add_rms_norm_f16(device const half *left [[buffer(0)]],
                             device const half *right [[buffer(1)]],
                             device const half *weight [[buffer(2)]],
                             device half *residual [[buffer(3)]],
                             device half *normalized [[buffer(4)]],
                             constant NormParams &p [[buffer(5)]],
                             uint group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simdgroup_index
                             [[simdgroup_index_in_threadgroup]]) {
  constexpr uint simdgroups_per_threadgroup = 8;
  uint row = group * simdgroups_per_threadgroup + simdgroup_index;
  if (row >= p.rows)
    return;
  uint base = row * p.width;
  float sum = 0.0f;
  for (uint i = lane; i < p.width; i += 32) {
    float value = float(left[base + i]) + float(right[base + i]);
    residual[base + i] = half(value);
    sum += value * value;
  }
  sum = simd_sum(sum);
  float scale = rsqrt(sum / float(p.width) + p.epsilon);
  for (uint i = lane; i < p.width; i += 32) {
    float value = float(residual[base + i]);
    normalized[base + i] = half(value * scale * float(weight[i]));
  }
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

kernel void qk_norm_rope_cache_f16(device const half *query [[buffer(0)]],
                                   device const half *key [[buffer(1)]],
                                   device const half *query_weight
                                   [[buffer(2)]],
                                   device const half *key_weight [[buffer(3)]],
                                   device half *query_out [[buffer(4)]],
                                   device half *key_cache [[buffer(5)]],
                                   constant QkTransformParams &p [[buffer(6)]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
  uint query_rows = p.tokens * p.q_heads;
  bool is_query = group < query_rows;
  uint row = is_query ? group : group - query_rows;
  uint heads = is_query ? p.q_heads : p.kv_heads;
  if (row >= p.tokens * heads)
    return;
  uint token = row / heads;
  uint head = row - token * heads;
  device const half *input = is_query ? query : key;
  device const half *weight = is_query ? query_weight : key_weight;
  uint base = row * p.head_dim;
  float sum = 0.0f;
  for (uint i = lane; i < p.head_dim; i += 32) {
    float value = float(input[base + i]);
    sum += value * value;
  }
  float scale = rsqrt(simd_sum(sum) / float(p.head_dim) + p.epsilon);
  for (uint pair = lane; pair < p.head_dim / 2; pair += 32) {
    uint second = pair + p.head_dim / 2;
    float x = float(input[base + pair]) * scale * float(weight[pair]);
    float y = float(input[base + second]) * scale * float(weight[second]);
    float frequency = pow(p.theta, -float(pair * 2) / float(p.head_dim));
    float angle = float(token + p.offset) * frequency;
    half rotated_x = half(x * cos(angle) - y * sin(angle));
    half rotated_y = half(x * sin(angle) + y * cos(angle));
    if (is_query) {
      query_out[base + pair] = rotated_x;
      query_out[base + second] = rotated_y;
    } else {
      uint cache_base = ((p.offset + token) * p.kv_heads + head) * p.head_dim;
      key_cache[cache_base + pair] = rotated_x;
      key_cache[cache_base + second] = rotated_y;
    }
  }
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
                                uint group [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  uint qi = group / p.q_heads;
  uint h = group - qi * p.q_heads;
  if (h >= p.q_heads || qi >= p.tokens)
    return;
  uint kvh = h / (p.q_heads / p.kv_heads);
  uint available =
      p.causal != 0 ? min(p.kv_length, p.query_offset + qi + 1) : p.kv_length;
  float running_max = -INFINITY;
  float running_sum = 0.0f;
  float accumulator[8];
  for (uint component = 0; component < 8; ++component)
    accumulator[component] = 0.0f;
  float scale = rsqrt(float(p.head_dim));
  for (uint j = 0; j < available; ++j) {
    float partial = 0.0f;
    for (uint component = 0; component < 8; ++component) {
      uint d = lane + component * 32;
      if (d < p.head_dim) {
        partial += float(q[(qi * p.q_heads + h) * p.head_dim + d]) *
                   float(k[(j * p.kv_heads + kvh) * p.head_dim + d]);
      }
    }
    float score = simd_sum(partial) * scale;
    float next_max = max(running_max, score);
    float previous_scale = exp(running_max - next_max);
    float current_scale = exp(score - next_max);
    for (uint component = 0; component < 8; ++component) {
      uint d = lane + component * 32;
      if (d < p.head_dim) {
        accumulator[component] =
            accumulator[component] * previous_scale +
            current_scale * float(v[(j * p.kv_heads + kvh) * p.head_dim + d]);
      }
    }
    running_sum = running_sum * previous_scale + current_scale;
    running_max = next_max;
  }
  for (uint component = 0; component < 8; ++component) {
    uint d = lane + component * 32;
    if (d < p.head_dim) {
      out[(qi * p.q_heads + h) * p.head_dim + d] =
          half(accumulator[component] / running_sum);
    }
  }
}

kernel void copy_kv_f16(device const half *source [[buffer(0)]],
                        device half *cache [[buffer(1)]],
                        constant uint3 &p [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
  uint count = p.x * p.z;
  if (id < count)
    cache[p.y * p.z + id] = source[id];
}
