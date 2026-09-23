#include "prelude.metal"
inline bool argmax_better(float candidate, uint candidate_index, float current,
                          uint current_index) {
  if (candidate_index == UINT_MAX)
    return false;
  if (current_index == UINT_MAX)
    return true;
  uint candidate_bits = as_type<uint>(candidate);
  uint current_bits = as_type<uint>(current);
  uint candidate_key = (candidate_bits & 0x80000000u)
                           ? ~candidate_bits
                           : candidate_bits ^ 0x80000000u;
  uint current_key =
      (current_bits & 0x80000000u) ? ~current_bits : current_bits ^ 0x80000000u;
  return candidate_key > current_key ||
         (candidate_key == current_key && candidate_index > current_index);
}

kernel void argmax_f16_partial(device const half *logits [[buffer(0)]],
                               device float *partial_values [[buffer(1)]],
                               device uint *partial_indices [[buffer(2)]],
                               constant uint &count [[buffer(3)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float group_values[8];
  threadgroup uint group_indices[8];
  float best = -INFINITY;
  uint best_index = UINT_MAX;
  for (uint j = 0; j < 8; ++j) {
    uint index = group * 2048 + lane + j * 256;
    if (index < count) {
      float value = float(logits[index]);
      if (isfinite(value) && argmax_better(value, index, best, best_index)) {
        best = value;
        best_index = index;
      }
    }
  }
  for (uint offset = 16; offset > 0; offset >>= 1) {
    float other_value = simd_shuffle_down(best, offset);
    uint other_index = simd_shuffle_down(best_index, offset);
    if ((lane & 31) < offset &&
        argmax_better(other_value, other_index, best, best_index)) {
      best = other_value;
      best_index = other_index;
    }
  }
  if ((lane & 31) == 0) {
    group_values[lane / 32] = best;
    group_indices[lane / 32] = best_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < 32) {
    best = lane < 8 ? group_values[lane] : -INFINITY;
    best_index = lane < 8 ? group_indices[lane] : UINT_MAX;
    for (uint offset = 16; offset > 0; offset >>= 1) {
      float other_value = simd_shuffle_down(best, offset);
      uint other_index = simd_shuffle_down(best_index, offset);
      if (lane < offset &&
          argmax_better(other_value, other_index, best, best_index)) {
        best = other_value;
        best_index = other_index;
      }
    }
    if (lane == 0) {
      partial_values[group] = best;
      partial_indices[group] = best_index;
    }
  }
}

kernel void argmax_f16_reduce(device const float *partial_values [[buffer(0)]],
                              device const uint *partial_indices [[buffer(1)]],
                              device uint *output [[buffer(2)]],
                              constant uint &groups [[buffer(3)]],
                              uint lane [[thread_index_in_threadgroup]]) {
  threadgroup float group_values[8];
  threadgroup uint group_indices[8];
  float best = -INFINITY;
  uint best_index = UINT_MAX;
  for (uint index = lane; index < groups; index += 256) {
    float value = partial_values[index];
    uint token_index = partial_indices[index];
    if (argmax_better(value, token_index, best, best_index)) {
      best = value;
      best_index = token_index;
    }
  }
  for (uint offset = 16; offset > 0; offset >>= 1) {
    float other_value = simd_shuffle_down(best, offset);
    uint other_index = simd_shuffle_down(best_index, offset);
    if ((lane & 31) < offset &&
        argmax_better(other_value, other_index, best, best_index)) {
      best = other_value;
      best_index = other_index;
    }
  }
  if ((lane & 31) == 0) {
    group_values[lane / 32] = best;
    group_indices[lane / 32] = best_index;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane < 32) {
    best = lane < 8 ? group_values[lane] : -INFINITY;
    best_index = lane < 8 ? group_indices[lane] : UINT_MAX;
    for (uint offset = 16; offset > 0; offset >>= 1) {
      float other_value = simd_shuffle_down(best, offset);
      uint other_index = simd_shuffle_down(best_index, offset);
      if (lane < offset &&
          argmax_better(other_value, other_index, best, best_index)) {
        best = other_value;
        best_index = other_index;
      }
    }
    if (lane == 0)
      output[0] = best_index;
  }
}
