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

kernel void attention_flash_decode_partial_legacy_f16(
    device const half *q [[buffer(0)]], device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]], device float *scratch [[buffer(3)]],
    constant AttentionParams &p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]) {
  uint block_keys = p.padding;
  constexpr uint tile_keys = 8;
  uint available = p.causal != 0 && p.query_offset < p.kv_length
                       ? p.query_offset + 1
                       : p.kv_length;
  uint blocks = (available - 1) / block_keys + 1;
  uint kvh = group / blocks;
  uint block = group - kvh * blocks;
  uint first = block * block_keys;
  uint block_length = min(block_keys, available - first);
  uint stride = p.head_dim + 2;
  float scale = rsqrt(float(p.head_dim));

  threadgroup float partial_max[2][tile_keys];
  threadgroup float partial_sum[2][tile_keys];
  threadgroup float partial_values[2][tile_keys][256];
  float running_max[2] = {-INFINITY, -INFINITY};
  float running_sum[2] = {0.0f, 0.0f};
  float accumulator[2][8];
  for (uint head = 0; head < 2; ++head)
    for (uint component = 0; component < 8; ++component)
      accumulator[head][component] = 0.0f;

  for (uint tile_key = simdgroup_index; tile_key < block_length;
       tile_key += tile_keys) {
    uint source = ((first + tile_key) * p.kv_heads + kvh) * p.head_dim;
    float key_values[8];
    float value_values[8];
    for (uint component = 0; component < 8; ++component) {
      uint d = lane + component * 32;
      key_values[component] = d < p.head_dim ? float(k[source + d]) : 0.0f;
      value_values[component] = d < p.head_dim ? float(v[source + d]) : 0.0f;
    }
    for (uint head = 0; head < 2; ++head) {
      uint qh = kvh * 2 + head;
      float dot = 0.0f;
      for (uint component = 0; component < 8; ++component) {
        uint d = lane + component * 32;
        if (d < p.head_dim)
          dot += float(q[qh * p.head_dim + d]) * key_values[component];
      }
      float score = simd_sum(dot) * scale;
      float next_max = max(running_max[head], score);
      float previous_scale = exp(running_max[head] - next_max);
      float current_scale = exp(score - next_max);
      for (uint component = 0; component < 8; ++component) {
        uint d = lane + component * 32;
        if (d < p.head_dim) {
          accumulator[head][component] =
              accumulator[head][component] * previous_scale +
              current_scale * value_values[component];
        }
      }
      running_sum[head] = running_sum[head] * previous_scale + current_scale;
      running_max[head] = next_max;
    }
  }

  for (uint head = 0; head < 2; ++head) {
    if (lane == 0) {
      partial_max[head][simdgroup_index] = running_max[head];
      partial_sum[head][simdgroup_index] = running_sum[head];
    }
    for (uint component = 0; component < 8; ++component) {
      uint d = lane + component * 32;
      if (d < p.head_dim)
        partial_values[head][simdgroup_index][d] = accumulator[head][component];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_index == 0) {
    for (uint head = 0; head < 2; ++head) {
      float maximum = -INFINITY;
      for (uint index = 0; index < tile_keys; ++index)
        maximum = max(maximum, partial_max[head][index]);
      float denominator = 0.0f;
      for (uint index = 0; index < tile_keys; ++index)
        denominator +=
            partial_sum[head][index] * exp(partial_max[head][index] - maximum);
      uint destination = ((kvh * blocks + block) * 2 + head) * stride;
      if (lane == 0) {
        scratch[destination] = maximum;
        scratch[destination + 1] = denominator;
      }
      for (uint component = 0; component < 8; ++component) {
        uint d = lane + component * 32;
        if (d < p.head_dim) {
          float numerator = 0.0f;
          for (uint index = 0; index < tile_keys; ++index)
            numerator += partial_values[head][index][d] *
                         exp(partial_max[head][index] - maximum);
          scratch[destination + 2 + d] = numerator;
        }
      }
    }
  }
}

kernel void attention_flash_decode_partial_128_legacy_f16(
    device const half *q [[buffer(0)]], device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]], device float *scratch [[buffer(3)]],
    constant AttentionParams &p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]) {
  uint block_keys = p.padding;
  constexpr uint tile_keys = 4;
  uint available = p.causal != 0 && p.query_offset < p.kv_length
                       ? p.query_offset + 1
                       : p.kv_length;
  uint blocks = (available - 1) / block_keys + 1;
  uint kvh = group / blocks;
  uint block = group - kvh * blocks;
  uint first = block * block_keys;
  uint block_length = min(block_keys, available - first);
  uint stride = p.head_dim + 2;
  float scale = rsqrt(float(p.head_dim));

  threadgroup float partial_max[2][tile_keys];
  threadgroup float partial_sum[2][tile_keys];
  threadgroup float partial_values[2][tile_keys][256];
  float running_max[2] = {-INFINITY, -INFINITY};
  float running_sum[2] = {0.0f, 0.0f};
  float accumulator[2][8];
  for (uint head = 0; head < 2; ++head)
    for (uint component = 0; component < 8; ++component)
      accumulator[head][component] = 0.0f;

  for (uint tile_key = simdgroup_index; tile_key < block_length;
       tile_key += tile_keys) {
    uint source = ((first + tile_key) * p.kv_heads + kvh) * p.head_dim;
    float key_values[8];
    float value_values[8];
    for (uint component = 0; component < 8; ++component) {
      uint d = lane + component * 32;
      key_values[component] = d < p.head_dim ? float(k[source + d]) : 0.0f;
      value_values[component] = d < p.head_dim ? float(v[source + d]) : 0.0f;
    }
    for (uint head = 0; head < 2; ++head) {
      uint qh = kvh * 2 + head;
      float dot = 0.0f;
      for (uint component = 0; component < 8; ++component) {
        uint d = lane + component * 32;
        if (d < p.head_dim)
          dot += float(q[qh * p.head_dim + d]) * key_values[component];
      }
      float score = simd_sum(dot) * scale;
      float next_max = max(running_max[head], score);
      float previous_scale = exp(running_max[head] - next_max);
      float current_scale = exp(score - next_max);
      for (uint component = 0; component < 8; ++component) {
        uint d = lane + component * 32;
        if (d < p.head_dim) {
          accumulator[head][component] =
              accumulator[head][component] * previous_scale +
              current_scale * value_values[component];
        }
      }
      running_sum[head] = running_sum[head] * previous_scale + current_scale;
      running_max[head] = next_max;
    }
  }

  for (uint head = 0; head < 2; ++head) {
    if (lane == 0) {
      partial_max[head][simdgroup_index] = running_max[head];
      partial_sum[head][simdgroup_index] = running_sum[head];
    }
    for (uint component = 0; component < 8; ++component) {
      uint d = lane + component * 32;
      if (d < p.head_dim)
        partial_values[head][simdgroup_index][d] = accumulator[head][component];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_index == 0) {
    for (uint head = 0; head < 2; ++head) {
      float maximum = -INFINITY;
      for (uint index = 0; index < tile_keys; ++index)
        maximum = max(maximum, partial_max[head][index]);
      float denominator = 0.0f;
      for (uint index = 0; index < tile_keys; ++index)
        denominator +=
            partial_sum[head][index] * exp(partial_max[head][index] - maximum);
      uint destination = ((kvh * blocks + block) * 2 + head) * stride;
      if (lane == 0) {
        scratch[destination] = maximum;
        scratch[destination + 1] = denominator;
      }
      for (uint component = 0; component < 8; ++component) {
        uint d = lane + component * 32;
        if (d < p.head_dim) {
          float numerator = 0.0f;
          for (uint index = 0; index < tile_keys; ++index)
            numerator += partial_values[head][index][d] *
                         exp(partial_max[head][index] - maximum);
          scratch[destination + 2 + d] = numerator;
        }
      }
    }
  }
}

// Kept as a correctness reference for the specialized 128-thread kernel.
// The implementation above remains the legacy path; the optimized kernel is
// deliberately separate so GPU tests can compare both implementations.

kernel void attention_flash_decode_partial_128_opt_f16(
    device const half *q [[buffer(0)]], device const half *k [[buffer(1)]],
    device const half *v [[buffer(2)]], device float *scratch [[buffer(3)]],
    constant AttentionParams &p [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]) {
  uint block_keys = p.padding;
  constexpr uint tile_keys = 4;
  uint available = p.causal != 0 && p.query_offset < p.kv_length
                       ? p.query_offset + 1
                       : p.kv_length;
  uint blocks = (available - 1) / block_keys + 1;
  uint kvh = group / blocks;
  uint block = group - kvh * blocks;
  uint first = block * block_keys;
  uint block_length = min(block_keys, available - first);
  uint stride = p.head_dim + 2;
  float scale = rsqrt(float(p.head_dim));

  threadgroup float partial_max[2][tile_keys];
  threadgroup float partial_sum[2][tile_keys];
  threadgroup float partial_values[2][tile_keys][256];
  float running_max[2] = {-INFINITY, -INFINITY};
  float running_sum[2] = {0.0f, 0.0f};
  float accumulator[2][4] = {};
  float query_values[2][4];
  for (uint head = 0; head < 2; ++head) {
    uint qh = kvh * 2 + head;
    for (uint component = 0; component < 4; ++component) {
      uint d = lane + component * 32;
      query_values[head][component] = float(q[qh * p.head_dim + d]);
    }
  }

  float key_values[4];
  float value_values[4];
  uint initial_source =
      ((first + simdgroup_index) * p.kv_heads + kvh) * p.head_dim;
  for (uint component = 0; component < 4; ++component) {
    uint d = lane + component * 32;
    key_values[component] =
        simdgroup_index < block_length ? float(k[initial_source + d]) : 0.0f;
    value_values[component] =
        simdgroup_index < block_length ? float(v[initial_source + d]) : 0.0f;
  }
  for (uint tile_key = simdgroup_index; tile_key < block_length;
       tile_key += tile_keys) {
    float next_key_values[4];
    float next_value_values[4];
    uint next_tile_key = tile_key + tile_keys;
    uint next_source =
        ((first + next_tile_key) * p.kv_heads + kvh) * p.head_dim;
    bool has_next = next_tile_key < block_length;
    for (uint component = 0; component < 4; ++component) {
      uint d = lane + component * 32;
      next_key_values[component] = 0.0f;
      next_value_values[component] = 0.0f;
      if (has_next) {
        next_key_values[component] = float(k[next_source + d]);
        next_value_values[component] = float(v[next_source + d]);
      }
    }
    for (uint head = 0; head < 2; ++head) {
      float dot = 0.0f;
      for (uint component = 0; component < 4; ++component)
        dot += query_values[head][component] * key_values[component];
      float score = simd_sum(dot) * scale;
      float next_max = max(running_max[head], score);
      float previous_scale = exp(running_max[head] - next_max);
      float current_scale = exp(score - next_max);
      for (uint component = 0; component < 4; ++component) {
        accumulator[head][component] =
            accumulator[head][component] * previous_scale +
            current_scale * value_values[component];
      }
      running_sum[head] = running_sum[head] * previous_scale + current_scale;
      running_max[head] = next_max;
    }
    // Keep the next tile's loads in registers for the following iteration.
    for (uint component = 0; component < 4; ++component) {
      key_values[component] = next_key_values[component];
      value_values[component] = next_value_values[component];
    }
  }

  for (uint head = 0; head < 2; ++head) {
    if (lane == 0) {
      partial_max[head][simdgroup_index] = running_max[head];
      partial_sum[head][simdgroup_index] = running_sum[head];
    }
    for (uint component = 0; component < 4; ++component) {
      uint d = lane + component * 32;
      partial_values[head][simdgroup_index][d] = accumulator[head][component];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simdgroup_index == 0) {
    for (uint head = 0; head < 2; ++head) {
      float maximum = -INFINITY;
      for (uint index = 0; index < tile_keys; ++index)
        maximum = max(maximum, partial_max[head][index]);
      float denominator = 0.0f;
      for (uint index = 0; index < tile_keys; ++index)
        denominator +=
            partial_sum[head][index] * exp(partial_max[head][index] - maximum);
      uint destination = ((kvh * blocks + block) * 2 + head) * stride;
      if (lane == 0) {
        scratch[destination] = maximum;
        scratch[destination + 1] = denominator;
      }
      for (uint component = 0; component < 4; ++component) {
        uint d = lane + component * 32;
        float numerator = 0.0f;
        for (uint index = 0; index < tile_keys; ++index)
          numerator += partial_values[head][index][d] *
                       exp(partial_max[head][index] - maximum);
        scratch[destination + 2 + d] = numerator;
      }
    }
  }
}

kernel void attention_flash_decode_reduce_legacy_f16(
    device const float *scratch [[buffer(0)]], device half *out [[buffer(1)]],
    constant AttentionParams &p [[buffer(2)]],
    uint qh [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
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
  float numerator[8] = {};
  for (uint block = 0; block < blocks; ++block) {
    uint source = ((kvh * blocks + block) * 2 + head) * stride;
    float factor = exp(scratch[source] - maximum);
    denominator += scratch[source + 1] * factor;
    for (uint component = 0; component < 8; ++component) {
      uint d = lane + component * 32;
      if (d < p.head_dim)
        numerator[component] += scratch[source + 2 + d] * factor;
    }
  }
  for (uint component = 0; component < 8; ++component) {
    uint d = lane + component * 32;
    if (d < p.head_dim)
      out[qh * p.head_dim + d] = half(numerator[component] / denominator);
  }
}

kernel void
attention_flash_decode_reduce_128_f16(device const float *scratch [[buffer(0)]],
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
