#include "prelude.metal"
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
