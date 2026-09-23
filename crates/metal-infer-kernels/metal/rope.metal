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
