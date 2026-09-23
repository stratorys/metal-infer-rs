#include "prelude.metal"
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

// Four SIMD-groups own eight query rows each. The score and probability tiles
// bridge matrix fragments and the row-wise online softmax.
kernel void attention_flash_prefill_f16(
    device const half *q [[buffer(0)]], device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]], device half *out [[buffer(3)]],
    constant AttentionParams &p [[buffer(4)]],
    uint2 group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]]) {
  constexpr uint tile = 32;
  constexpr uint matrix_width = 8;
  threadgroup half q_tile[tile * matrix_width];
  threadgroup half kv_tile[tile * matrix_width];
  threadgroup float scores[tile * tile];
  threadgroup half probabilities[tile * tile];
  threadgroup float row_max[tile];
  threadgroup float row_sum[tile];
  threadgroup float row_scale[tile];

  uint query_base = group.x * tile;
  uint head = group.y;
  uint kv_head = head / (p.q_heads / p.kv_heads);
  uint query_row = simdgroup_index * matrix_width;
  uint matrix_row = ((simd_lane / 4) & 4) + (simd_lane / 2) % 4;
  uint matrix_column = ((simd_lane / 4) & 2) * 2 + (simd_lane % 2) * 2;
  uint dimensions = (p.head_dim + matrix_width - 1) / matrix_width;
  simdgroup_float8x8 output_tiles[32];
  for (uint d = 0; d < dimensions; ++d)
    output_tiles[d] = make_filled_simdgroup_matrix<float, 8>(0.0f);
  if (thread_index < tile) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint last_query = min(query_base + tile, p.tokens) - 1;
  uint key_limit = p.causal != 0
                       ? min(p.kv_length, p.query_offset + last_query + 1)
                       : p.kv_length;
  float score_scale = rsqrt(float(p.head_dim));
  for (uint key_base = 0; key_base < key_limit; key_base += tile) {
    simdgroup_float8x8 score_tiles[4];
    for (uint n = 0; n < 4; ++n)
      score_tiles[n] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (uint d = 0; d < dimensions; ++d) {
      for (uint slot = thread_index; slot < tile * matrix_width; slot += 128) {
        uint row = slot / matrix_width;
        uint component = d * matrix_width + slot % matrix_width;
        q_tile[slot] =
            query_base + row < p.tokens && component < p.head_dim
                ? q[((query_base + row) * p.q_heads + head) * p.head_dim +
                    component]
                : half(0.0f);
        kv_tile[slot] =
            key_base + row < p.kv_length && component < p.head_dim
                ? k[((key_base + row) * p.kv_heads + kv_head) * p.head_dim +
                    component]
                : half(0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_half8x8 q_matrix;
      simdgroup_load(q_matrix, q_tile + query_row * matrix_width, matrix_width);
      for (uint n = 0; n < 4; ++n) {
        simdgroup_half8x8 k_matrix;
        simdgroup_load(k_matrix, kv_tile + n * matrix_width * matrix_width,
                       matrix_width, ulong2(0), true);
        simdgroup_multiply_accumulate(score_tiles[n], q_matrix, k_matrix,
                                      score_tiles[n]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint n = 0; n < 4; ++n) {
      uint offset =
          (query_row + matrix_row) * tile + n * matrix_width + matrix_column;
      scores[offset] = score_tiles[n].thread_elements()[0] * score_scale;
      scores[offset + 1] = score_tiles[n].thread_elements()[1] * score_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (thread_index < tile) {
      uint query = query_base + thread_index;
      uint available = p.causal != 0
                           ? min(p.kv_length, p.query_offset + query + 1)
                           : p.kv_length;
      float block_max = -INFINITY;
      if (query < p.tokens) {
        for (uint column = 0; column < tile; ++column) {
          if (key_base + column < available)
            block_max = max(block_max, scores[thread_index * tile + column]);
        }
      }
      float next_max = max(row_max[thread_index], block_max);
      float alpha = isinf(row_max[thread_index])
                        ? 0.0f
                        : exp(row_max[thread_index] - next_max);
      row_scale[thread_index] = alpha;
      float block_sum = 0.0f;
      for (uint column = 0; column < tile; ++column) {
        float weight =
            query < p.tokens && key_base + column < available
                ? exp(scores[thread_index * tile + column] - next_max)
                : 0.0f;
        probabilities[thread_index * tile + column] = half(weight);
        block_sum += weight;
      }
      row_sum[thread_index] = row_sum[thread_index] * alpha + block_sum;
      row_max[thread_index] = next_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float alpha = row_scale[query_row + matrix_row];
    for (uint d = 0; d < dimensions; ++d) {
      output_tiles[d].thread_elements()[0] *= alpha;
      output_tiles[d].thread_elements()[1] *= alpha;
      for (uint slot = thread_index; slot < tile * matrix_width; slot += 128) {
        uint row = slot / matrix_width;
        uint component = d * matrix_width + slot % matrix_width;
        kv_tile[slot] =
            key_base + row < p.kv_length && component < p.head_dim
                ? v[((key_base + row) * p.kv_heads + kv_head) * p.head_dim +
                    component]
                : half(0.0f);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint n = 0; n < 4; ++n) {
        simdgroup_half8x8 p_matrix;
        simdgroup_half8x8 v_matrix;
        simdgroup_load(p_matrix,
                       probabilities + query_row * tile + n * matrix_width,
                       tile);
        simdgroup_load(v_matrix, kv_tile + n * matrix_width * matrix_width,
                       matrix_width);
        simdgroup_multiply_accumulate(output_tiles[d], p_matrix, v_matrix,
                                      output_tiles[d]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  uint query = query_base + query_row + matrix_row;
  if (query < p.tokens) {
    float denominator = row_sum[query_row + matrix_row];
    for (uint d = 0; d < dimensions; ++d) {
      uint component = d * matrix_width + matrix_column;
      uint offset = (query * p.q_heads + head) * p.head_dim + component;
      if (component < p.head_dim)
        out[offset] = half(output_tiles[d].thread_elements()[0] / denominator);
      if (component + 1 < p.head_dim)
        out[offset + 1] =
            half(output_tiles[d].thread_elements()[1] / denominator);
    }
  }
}

kernel void attention_decode_f16(device const half *q [[buffer(0)]],
                                 device const half *k [[buffer(1)]],
                                 device const half *v [[buffer(2)]],
                                 device half *out [[buffer(3)]],
                                 constant AttentionParams &p [[buffer(4)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simdgroup_index
                                 [[simdgroup_index_in_threadgroup]]) {
  constexpr uint simdgroups_per_threadgroup = 8;
  threadgroup float partial_max[simdgroups_per_threadgroup];
  threadgroup float partial_sum[simdgroups_per_threadgroup];
  threadgroup float partial_values[simdgroups_per_threadgroup][256];

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

  for (uint j = simdgroup_index; j < available;
       j += simdgroups_per_threadgroup) {
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

  if (lane == 0) {
    partial_max[simdgroup_index] = running_max;
    partial_sum[simdgroup_index] = running_sum;
  }
  for (uint component = 0; component < 8; ++component) {
    uint d = lane + component * 32;
    if (d < p.head_dim)
      partial_values[simdgroup_index][d] = accumulator[component];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_index == 0) {
    float maximum = -INFINITY;
    for (uint index = 0; index < simdgroups_per_threadgroup; ++index)
      maximum = max(maximum, partial_max[index]);
    float denominator = 0.0f;
    for (uint index = 0; index < simdgroups_per_threadgroup; ++index)
      denominator += partial_sum[index] * exp(partial_max[index] - maximum);
    for (uint component = 0; component < 8; ++component) {
      uint d = lane + component * 32;
      if (d < p.head_dim) {
        float numerator = 0.0f;
        for (uint index = 0; index < simdgroups_per_threadgroup; ++index) {
          numerator +=
              partial_values[index][d] * exp(partial_max[index] - maximum);
        }
        out[(qi * p.q_heads + h) * p.head_dim + d] =
            half(numerator / denominator);
      }
    }
  }
}

kernel void attention_flash_decode_partial_f16(
    device const half *q [[buffer(0)]], device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]], device float *scratch [[buffer(3)]],
    constant AttentionParams &p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]) {
  constexpr uint head_dim = 128;
  constexpr uint max_block = 256;
  constexpr uint key_lanes = 8;
  constexpr uint key_rows = 16;
  constexpr uint value_lanes = 16;
  constexpr uint value_rows = 8;
  uint block_keys = p.padding;
  uint available = p.causal != 0 && p.query_offset < p.kv_length
                       ? p.query_offset + 1
                       : p.kv_length;
  uint blocks = (available - 1) / block_keys + 1;
  uint kvh = group / blocks;
  uint block = group - kvh * blocks;
  uint first = block * block_keys;
  uint block_length = min(block_keys, available - first);
  uint row_stride = p.kv_heads * head_dim / 4;
  device const half4 *keys =
      (device const half4 *)(k + (first * p.kv_heads + kvh) * head_dim);
  device const half4 *values =
      (device const half4 *)(v + (first * p.kv_heads + kvh) * head_dim);
  float scale = rsqrt(float(head_dim));

  threadgroup float weights[2][max_block];
  threadgroup float totals[2][2];
  threadgroup float4 partial_values[4][2][head_dim / 4];

  uint key_part = thread_index % key_lanes;
  uint key_row = thread_index / key_lanes;
  float4 query_values[2][4];
  for (uint head = 0; head < 2; ++head) {
    device const half4 *query =
        (device const half4 *)(q + (kvh * 2 + head) * head_dim);
    for (uint index = 0; index < 4; ++index)
      query_values[head][index] = float4(query[index * key_lanes + key_part]);
  }

  for (uint row_base = 0; row_base < block_length; row_base += key_rows) {
    uint row = row_base + key_row;
    bool valid = row < block_length;
    float dot0 = 0.0f;
    float dot1 = 0.0f;
    if (valid) {
      device const half4 *key = keys + row * row_stride;
      for (uint index = 0; index < 4; ++index) {
        float4 key_value = float4(key[index * key_lanes + key_part]);
        dot0 += dot(query_values[0][index], key_value);
        dot1 += dot(query_values[1][index], key_value);
      }
    }
    for (ushort offset = key_lanes / 2; offset > 0; offset /= 2) {
      dot0 += simd_shuffle_xor(dot0, offset);
      dot1 += simd_shuffle_xor(dot1, offset);
    }
    if (valid && key_part == 0) {
      weights[0][row] = dot0 * scale;
      weights[1][row] = dot1 * scale;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_index < 2) {
    uint head = simdgroup_index;
    float maximum = -INFINITY;
    for (uint row = lane; row < block_length; row += 32)
      maximum = max(maximum, weights[head][row]);
    maximum = simd_max(maximum);
    float sum = 0.0f;
    for (uint row = lane; row < block_length; row += 32) {
      float weight = exp(weights[head][row] - maximum);
      weights[head][row] = weight;
      sum += weight;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
      totals[head][0] = maximum;
      totals[head][1] = sum;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint value_part = thread_index % value_lanes;
  uint value_row = thread_index / value_lanes;
  float4 accumulator[2][2] = {};
  for (uint row = value_row; row < block_length; row += value_rows) {
    device const half4 *value = values + row * row_stride;
    float weight0 = weights[0][row];
    float weight1 = weights[1][row];
    for (uint index = 0; index < 2; ++index) {
      float4 value_value = float4(value[index * value_lanes + value_part]);
      accumulator[0][index] += weight0 * value_value;
      accumulator[1][index] += weight1 * value_value;
    }
  }
  for (uint head = 0; head < 2; ++head) {
    for (uint index = 0; index < 2; ++index) {
      accumulator[head][index] +=
          simd_shuffle_xor(accumulator[head][index], ushort(value_lanes));
      if (lane < value_lanes)
        partial_values[simdgroup_index][head]
                      [index * value_lanes + value_part] =
                          accumulator[head][index];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint stride = head_dim + 2;
  for (uint head = 0; head < 2; ++head) {
    float numerator = 0.0f;
    for (uint index = 0; index < 4; ++index)
      numerator +=
          partial_values[index][head][thread_index / 4][thread_index % 4];
    uint destination = ((kvh * blocks + block) * 2 + head) * stride;
    if (thread_index == 0) {
      scratch[destination] = totals[head][0];
      scratch[destination + 1] = totals[head][1];
    }
    scratch[destination + 2 + thread_index] = numerator;
  }
}

kernel void
attention_flash_decode_reduce_f16(device const float *scratch [[buffer(0)]],
                                  device half *out [[buffer(1)]],
                                  constant AttentionParams &p [[buffer(2)]],
                                  uint qh [[threadgroup_position_in_grid]],
                                  uint d [[thread_index_in_threadgroup]]) {
  uint block_keys = p.padding;
  uint available = p.causal != 0 && p.query_offset < p.kv_length
                       ? p.query_offset + 1
                       : p.kv_length;
  uint blocks = (available - 1) / block_keys + 1;
  uint kvh = qh / 2;
  uint head = qh - kvh * 2;
  uint stride = p.head_dim + 2;
  float maximum = -INFINITY;
  for (uint block = 0; block < blocks; ++block) {
    uint source = ((kvh * blocks + block) * 2 + head) * stride;
    maximum = max(maximum, scratch[source]);
  }
  float denominator = 0.0f;
  float numerator = 0.0f;
  for (uint block = 0; block < blocks; ++block) {
    uint source = ((kvh * blocks + block) * 2 + head) * stride;
    float factor = exp(scratch[source] - maximum);
    denominator += scratch[source + 1] * factor;
    numerator += scratch[source + 2 + d] * factor;
  }
  out[qh * p.head_dim + d] = half(numerator / denominator);
}
