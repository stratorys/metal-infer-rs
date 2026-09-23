#include "prelude.metal"
template <ushort ROWS, ushort GROUPS>
void matvec3_rms_impl(device const half *x, device const half *norm_weight,
                      device const half *weight0, device const half *weight1,
                      device const half *weight2, device half *out0,
                      device half *out1, device half *out2,
                      constant NormMultiMatrixParams &p, uint group, uint lane,
                      uint simdgroup_index, threadgroup float *partial,
                      threadgroup float *scale) {
  uint thread_id = simdgroup_index * 32 + lane;
  float square_sum = 0.0f;
  for (uint i = thread_id; i < p.k; i += GROUPS * 32) {
    float value = float(x[i]);
    square_sum += value * value;
  }
  float simd_total = simd_sum(square_sum);
  if (lane == 0)
    partial[simdgroup_index] = simd_total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_id == 0) {
    float total = 0.0f;
    for (uint index = 0; index < GROUPS; ++index)
      total += partial[index];
    *scale = rsqrt(total / float(p.k) + p.epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint first_row = (group * GROUPS + simdgroup_index) * ROWS;
  device const half4 *x4 = reinterpret_cast<device const half4 *>(x);
  device const half4 *norm4 =
      reinterpret_cast<device const half4 *>(norm_weight);
  float sums0[ROWS] = {};
  float sums1[ROWS] = {};
  float sums2[ROWS] = {};
  for (uint i = lane * 2; i < p.k / 4; i += 64) {
    float4 input0 = float4(half4(float4(x4[i]) * (*scale) * float4(norm4[i])));
    float4 input1 =
        float4(half4(float4(x4[i + 1]) * (*scale) * float4(norm4[i + 1])));
    for (uint row_offset = 0; row_offset < ROWS; ++row_offset) {
      uint row = first_row + row_offset;
      if (row < p.n0) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight0 + row * p.k);
        sums0[row_offset] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
      if (row < p.n1) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight1 + row * p.k);
        sums1[row_offset] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
      if (row < p.n2) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight2 + row * p.k);
        sums2[row_offset] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
    }
  }
  for (uint row_offset = 0; row_offset < ROWS; ++row_offset) {
    uint row = first_row + row_offset;
    float total0 = simd_sum(sums0[row_offset]);
    float total1 = simd_sum(sums1[row_offset]);
    float total2 = simd_sum(sums2[row_offset]);
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

kernel void matvec3_rms_r1_f16(device const half *x [[buffer(0)]],
                               device const half *norm_weight [[buffer(1)]],
                               device const half *weight0 [[buffer(2)]],
                               device const half *weight1 [[buffer(3)]],
                               device const half *weight2 [[buffer(4)]],
                               device half *out0 [[buffer(5)]],
                               device half *out1 [[buffer(6)]],
                               device half *out2 [[buffer(7)]],
                               constant NormMultiMatrixParams &p [[buffer(8)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simdgroup_index
                               [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[8];
  threadgroup float scale;
  matvec3_rms_impl<1, 8>(x, norm_weight, weight0, weight1, weight2, out0, out1,
                         out2, p, group, lane, simdgroup_index, partial,
                         &scale);
}

kernel void matvec3_rms_f16(device const half *x [[buffer(0)]],
                            device const half *norm_weight [[buffer(1)]],
                            device const half *weight0 [[buffer(2)]],
                            device const half *weight1 [[buffer(3)]],
                            device const half *weight2 [[buffer(4)]],
                            device half *out0 [[buffer(5)]],
                            device half *out1 [[buffer(6)]],
                            device half *out2 [[buffer(7)]],
                            constant NormMultiMatrixParams &p [[buffer(8)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simdgroup_index
                            [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[8];
  threadgroup float scale;
  matvec3_rms_impl<2, 4>(x, norm_weight, weight0, weight1, weight2, out0, out1,
                         out2, p, group, lane, simdgroup_index, partial,
                         &scale);
}

kernel void matvec3_rms_r4_f16(device const half *x [[buffer(0)]],
                               device const half *norm_weight [[buffer(1)]],
                               device const half *weight0 [[buffer(2)]],
                               device const half *weight1 [[buffer(3)]],
                               device const half *weight2 [[buffer(4)]],
                               device half *out0 [[buffer(5)]],
                               device half *out1 [[buffer(6)]],
                               device half *out2 [[buffer(7)]],
                               constant NormMultiMatrixParams &p [[buffer(8)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simdgroup_index
                               [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[8];
  threadgroup float scale;
  matvec3_rms_impl<4, 4>(x, norm_weight, weight0, weight1, weight2, out0, out1,
                         out2, p, group, lane, simdgroup_index, partial,
                         &scale);
}

kernel void matvec3_rms_r8_f16(device const half *x [[buffer(0)]],
                               device const half *norm_weight [[buffer(1)]],
                               device const half *weight0 [[buffer(2)]],
                               device const half *weight1 [[buffer(3)]],
                               device const half *weight2 [[buffer(4)]],
                               device half *out0 [[buffer(5)]],
                               device half *out1 [[buffer(6)]],
                               device half *out2 [[buffer(7)]],
                               constant NormMultiMatrixParams &p [[buffer(8)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simdgroup_index
                               [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[8];
  threadgroup float scale;
  matvec3_rms_impl<8, 4>(x, norm_weight, weight0, weight1, weight2, out0, out1,
                         out2, p, group, lane, simdgroup_index, partial,
                         &scale);
}

template <ushort ROWS, ushort GROUPS>
void matvec2_add_rms_impl(device const half *left, device const half *right,
                          device const half *norm_weight,
                          device const half *weight0,
                          device const half *weight1, device half *residual,
                          device half *out0, device half *out1,
                          constant NormMultiMatrixParams &p, uint group,
                          uint lane, uint simdgroup_index,
                          threadgroup float *partial,
                          threadgroup float *scale) {
  uint thread_id = simdgroup_index * 32 + lane;
  float square_sum = 0.0f;
  for (uint i = thread_id; i < p.k; i += GROUPS * 32) {
    float value = float(left[i]) + float(right[i]);
    square_sum += value * value;
    if (group == 0)
      residual[i] = half(value);
  }
  float simd_total = simd_sum(square_sum);
  if (lane == 0)
    partial[simdgroup_index] = simd_total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_id == 0) {
    float total = 0.0f;
    for (uint index = 0; index < GROUPS; ++index)
      total += partial[index];
    *scale = rsqrt(total / float(p.k) + p.epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint first_row = (group * GROUPS + simdgroup_index) * ROWS;
  device const half4 *left4 = reinterpret_cast<device const half4 *>(left);
  device const half4 *right4 = reinterpret_cast<device const half4 *>(right);
  device const half4 *norm4 =
      reinterpret_cast<device const half4 *>(norm_weight);
  float sums0[ROWS] = {};
  float sums1[ROWS] = {};
  for (uint i = lane * 2; i < p.k / 4; i += 64) {
    float4 residual0 = float4(half4(float4(left4[i]) + float4(right4[i])));
    float4 residual1 =
        float4(half4(float4(left4[i + 1]) + float4(right4[i + 1])));
    float4 input0 = float4(half4(residual0 * (*scale) * float4(norm4[i])));
    float4 input1 = float4(half4(residual1 * (*scale) * float4(norm4[i + 1])));
    for (uint row_offset = 0; row_offset < ROWS; ++row_offset) {
      uint row = first_row + row_offset;
      if (row < p.n0) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight0 + row * p.k);
        sums0[row_offset] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
      if (row < p.n1) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight1 + row * p.k);
        sums1[row_offset] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
    }
  }
  for (uint row_offset = 0; row_offset < ROWS; ++row_offset) {
    uint row = first_row + row_offset;
    float total0 = simd_sum(sums0[row_offset]);
    float total1 = simd_sum(sums1[row_offset]);
    if (lane == 0) {
      if (row < p.n0)
        out0[row] = half(total0);
      if (row < p.n1)
        out1[row] = half(total1);
    }
  }
}

kernel void matvec2_add_rms_r1_f16(device const half *left [[buffer(0)]],
                                   device const half *right [[buffer(1)]],
                                   device const half *norm_weight [[buffer(2)]],
                                   device const half *weight0 [[buffer(3)]],
                                   device const half *weight1 [[buffer(4)]],
                                   device half *residual [[buffer(5)]],
                                   device half *out0 [[buffer(6)]],
                                   device half *out1 [[buffer(7)]],
                                   constant NormMultiMatrixParams &p
                                   [[buffer(8)]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simdgroup_index
                                   [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[8];
  threadgroup float scale;
  matvec2_add_rms_impl<1, 8>(left, right, norm_weight, weight0, weight1,
                             residual, out0, out1, p, group, lane,
                             simdgroup_index, partial, &scale);
}

kernel void matvec2_add_rms_f16(device const half *left [[buffer(0)]],
                                device const half *right [[buffer(1)]],
                                device const half *norm_weight [[buffer(2)]],
                                device const half *weight0 [[buffer(3)]],
                                device const half *weight1 [[buffer(4)]],
                                device half *residual [[buffer(5)]],
                                device half *out0 [[buffer(6)]],
                                device half *out1 [[buffer(7)]],
                                constant NormMultiMatrixParams &p [[buffer(8)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint simdgroup_index
                                [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[8];
  threadgroup float scale;
  matvec2_add_rms_impl<2, 4>(left, right, norm_weight, weight0, weight1,
                             residual, out0, out1, p, group, lane,
                             simdgroup_index, partial, &scale);
}

kernel void matvec2_add_rms_r4_f16(device const half *left [[buffer(0)]],
                                   device const half *right [[buffer(1)]],
                                   device const half *norm_weight [[buffer(2)]],
                                   device const half *weight0 [[buffer(3)]],
                                   device const half *weight1 [[buffer(4)]],
                                   device half *residual [[buffer(5)]],
                                   device half *out0 [[buffer(6)]],
                                   device half *out1 [[buffer(7)]],
                                   constant NormMultiMatrixParams &p
                                   [[buffer(8)]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simdgroup_index
                                   [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[8];
  threadgroup float scale;
  matvec2_add_rms_impl<4, 4>(left, right, norm_weight, weight0, weight1,
                             residual, out0, out1, p, group, lane,
                             simdgroup_index, partial, &scale);
}

kernel void matvec2_add_rms_r8_f16(device const half *left [[buffer(0)]],
                                   device const half *right [[buffer(1)]],
                                   device const half *norm_weight [[buffer(2)]],
                                   device const half *weight0 [[buffer(3)]],
                                   device const half *weight1 [[buffer(4)]],
                                   device half *residual [[buffer(5)]],
                                   device half *out0 [[buffer(6)]],
                                   device half *out1 [[buffer(7)]],
                                   constant NormMultiMatrixParams &p
                                   [[buffer(8)]],
                                   uint group [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simdgroup_index
                                   [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[8];
  threadgroup float scale;
  matvec2_add_rms_impl<8, 4>(left, right, norm_weight, weight0, weight1,
                             residual, out0, out1, p, group, lane,
                             simdgroup_index, partial, &scale);
}
