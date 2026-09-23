#include "prelude.metal"
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

template <ushort ROWS>
void matvec_tuned_impl(device const half *x, device const half *weight,
                       device half *out, constant MatrixParams &p, uint group,
                       uint lane, uint simdgroup_index) {
  constexpr uint groups = 4;
  uint first_row = (group * groups + simdgroup_index) * ROWS;
  device const half4 *x4 = reinterpret_cast<device const half4 *>(x);
  uint vectors = p.k / 4;
  float sums[ROWS] = {};
  for (uint i = lane * 2; i < vectors; i += 64) {
    float4 input0 = float4(x4[i]);
    float4 input1 = float4(x4[i + 1]);
    for (uint output = 0; output < ROWS; ++output) {
      uint row = first_row + output;
      if (row < p.n) {
        device const half4 *weight4 =
            reinterpret_cast<device const half4 *>(weight + row * p.k);
        sums[output] += dot(input0, float4(weight4[i]));
        sums[output] += dot(input1, float4(weight4[i + 1]));
      }
    }
  }
  for (uint output = 0; output < ROWS; ++output) {
    float total = simd_sum(sums[output]);
    uint row = first_row + output;
    if (lane == 0 && row < p.n)
      out[row] = half(total);
  }
}

kernel void matvec_tuned_f16(device const half *x [[buffer(0)]],
                             device const half *weight [[buffer(1)]],
                             device half *out [[buffer(2)]],
                             constant MatrixParams &p [[buffer(3)]],
                             uint group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simdgroup_index
                             [[simdgroup_index_in_threadgroup]]) {
  matvec_tuned_impl<4>(x, weight, out, p, group, lane, simdgroup_index);
}

kernel void matvec_one_row_f16(device const half *x [[buffer(0)]],
                               device const half *weight [[buffer(1)]],
                               device half *out [[buffer(2)]],
                               constant MatrixParams &p [[buffer(3)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simdgroup_index
                               [[simdgroup_index_in_threadgroup]]) {
  uint row = group * 8 + simdgroup_index;
  if (row >= p.n)
    return;
  device const half4 *x4 = reinterpret_cast<device const half4 *>(x);
  device const half4 *w4 =
      reinterpret_cast<device const half4 *>(weight + row * p.k);
  float sum = 0.0f;
  for (uint i = lane * 2; i < p.k / 4; i += 64) {
    sum += dot(float4(x4[i]), float4(w4[i]));
    sum += dot(float4(x4[i + 1]), float4(w4[i + 1]));
  }
  float total = simd_sum(sum);
  if (lane == 0)
    out[row] = half(total);
}

struct Half8Load {
  half4 low;
  half4 high;
};

kernel void matvec_one_row_half8_f16(
    device const half *x [[buffer(0)]], device const half *weight [[buffer(1)]],
    device half *out [[buffer(2)]], constant MatrixParams &p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]) {
  uint row = group * 8 + simdgroup_index;
  if (row >= p.n)
    return;
  device const Half8Load *x8 = reinterpret_cast<device const Half8Load *>(x);
  device const Half8Load *w8 =
      reinterpret_cast<device const Half8Load *>(weight + row * p.k);
  float sum = 0.0f;
  for (uint i = lane; i < p.k / 8; i += 32) {
    Half8Load inputs = x8[i];
    Half8Load weights = w8[i];
    sum += dot(float4(inputs.low), float4(weights.low));
    sum += dot(float4(inputs.high), float4(weights.high));
  }
  float total = simd_sum(sum);
  if (lane == 0)
    out[row] = half(total);
}

kernel void matvec_splitk_partial_f16(
    device const half *x [[buffer(0)]], device const half *weight [[buffer(1)]],
    device float *partial [[buffer(2)]], constant MatrixParams &p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]) {
  uint groups_per_split = (p.n + 7) / 8;
  uint split = group / groups_per_split;
  uint row = (group % groups_per_split) * 8 + simdgroup_index;
  if (row >= p.n)
    return;
  uint vectors = p.k / 4;
  uint vectors_per_split = (vectors + p.padding - 1) / p.padding;
  uint begin = split * vectors_per_split;
  uint end = min(begin + vectors_per_split, vectors);
  device const half4 *x4 = reinterpret_cast<device const half4 *>(x);
  device const half4 *w4 =
      reinterpret_cast<device const half4 *>(weight + row * p.k);
  float sum = 0.0f;
  for (uint i = begin + lane; i < end; i += 32)
    sum += dot(float4(x4[i]), float4(w4[i]));
  float total = simd_sum(sum);
  if (lane == 0)
    partial[split * p.n + row] = total;
}

kernel void matvec_splitk_reduce_f16(device const float *partial [[buffer(0)]],
                                     device half *out [[buffer(1)]],
                                     constant MatrixParams &p [[buffer(2)]],
                                     uint row [[thread_position_in_grid]]) {
  if (row >= p.n)
    return;
  float sum = 0.0f;
  for (uint split = 0; split < p.padding; ++split)
    sum += partial[split * p.n + row];
  out[row] = half(sum);
}

kernel void matvec_tuned2_f16(device const half *x [[buffer(0)]],
                              device const half *weight [[buffer(1)]],
                              device half *out [[buffer(2)]],
                              constant MatrixParams &p [[buffer(3)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simdgroup_index
                              [[simdgroup_index_in_threadgroup]]) {
  matvec_tuned_impl<2>(x, weight, out, p, group, lane, simdgroup_index);
}

kernel void matvec_tuned8_f16(device const half *x [[buffer(0)]],
                              device const half *weight [[buffer(1)]],
                              device half *out [[buffer(2)]],
                              constant MatrixParams &p [[buffer(3)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simdgroup_index
                              [[simdgroup_index_in_threadgroup]]) {
  matvec_tuned_impl<8>(x, weight, out, p, group, lane, simdgroup_index);
}

kernel void matvec_vocab_f16(device const half *x [[buffer(0)]],
                             device const half *weight [[buffer(1)]],
                             device half *out [[buffer(2)]],
                             constant MatrixParams &p [[buffer(3)]],
                             uint group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simdgroup_index
                             [[simdgroup_index_in_threadgroup]]) {
  matvec_tuned_impl<2>(x, weight, out, p, group, lane, simdgroup_index);
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

// Fused decode projections use two outputs per SIMD-group to keep the number
// of live FP32 accumulators bounded when two or three matrices are present.
kernel void matvec2_tuned_f16(device const half *x [[buffer(0)]],
                              device const half *weight0 [[buffer(1)]],
                              device const half *weight1 [[buffer(2)]],
                              device half *out0 [[buffer(3)]],
                              device half *out1 [[buffer(4)]],
                              constant MultiMatrixParams &p [[buffer(5)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simdgroup_index
                              [[simdgroup_index_in_threadgroup]]) {
  constexpr uint rows = 2;
  uint first_row = (group * 4 + simdgroup_index) * rows;
  device const half4 *x4 = reinterpret_cast<device const half4 *>(x);
  float sums0[rows] = {};
  float sums1[rows] = {};
  for (uint i = lane * 2; i < p.k / 4; i += 64) {
    float4 input0 = float4(x4[i]);
    float4 input1 = float4(x4[i + 1]);
    for (uint output = 0; output < rows; ++output) {
      uint row = first_row + output;
      if (row < p.n0) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight0 + row * p.k);
        sums0[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
      if (row < p.n1) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight1 + row * p.k);
        sums1[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
    }
  }
  for (uint output = 0; output < rows; ++output) {
    uint row = first_row + output;
    float total0 = simd_sum(sums0[output]);
    float total1 = simd_sum(sums1[output]);
    if (lane == 0) {
      if (row < p.n0)
        out0[row] = half(total0);
      if (row < p.n1)
        out1[row] = half(total1);
    }
  }
}

kernel void matvec3_tuned_f16(device const half *x [[buffer(0)]],
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
  constexpr uint rows = 2;
  uint first_row = (group * 4 + simdgroup_index) * rows;
  device const half4 *x4 = reinterpret_cast<device const half4 *>(x);
  float sums0[rows] = {};
  float sums1[rows] = {};
  float sums2[rows] = {};
  for (uint i = lane * 2; i < p.k / 4; i += 64) {
    float4 input0 = float4(x4[i]);
    float4 input1 = float4(x4[i + 1]);
    for (uint output = 0; output < rows; ++output) {
      uint row = first_row + output;
      if (row < p.n0) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight0 + row * p.k);
        sums0[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
      if (row < p.n1) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight1 + row * p.k);
        sums1[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
      if (row < p.n2) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight2 + row * p.k);
        sums2[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
    }
  }
  for (uint output = 0; output < rows; ++output) {
    uint row = first_row + output;
    float total0 = simd_sum(sums0[output]);
    float total1 = simd_sum(sums1[output]);
    float total2 = simd_sum(sums2[output]);
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

kernel void matvec2_tuned4_f16(device const half *x [[buffer(0)]],
                               device const half *weight0 [[buffer(1)]],
                               device const half *weight1 [[buffer(2)]],
                               device half *out0 [[buffer(3)]],
                               device half *out1 [[buffer(4)]],
                               constant MultiMatrixParams &p [[buffer(5)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simdgroup_index
                               [[simdgroup_index_in_threadgroup]]) {
  constexpr uint rows = 4;
  uint first_row = (group * 4 + simdgroup_index) * rows;
  device const half4 *x4 = reinterpret_cast<device const half4 *>(x);
  float sums0[rows] = {};
  float sums1[rows] = {};
  for (uint i = lane * 2; i < p.k / 4; i += 64) {
    float4 input0 = float4(x4[i]);
    float4 input1 = float4(x4[i + 1]);
    for (uint output = 0; output < rows; ++output) {
      uint row = first_row + output;
      if (row < p.n0) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight0 + row * p.k);
        sums0[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
      if (row < p.n1) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight1 + row * p.k);
        sums1[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
    }
  }
  for (uint output = 0; output < rows; ++output) {
    uint row = first_row + output;
    float total0 = simd_sum(sums0[output]);
    float total1 = simd_sum(sums1[output]);
    if (lane == 0) {
      if (row < p.n0)
        out0[row] = half(total0);
      if (row < p.n1)
        out1[row] = half(total1);
    }
  }
}

kernel void matvec3_tuned4_f16(device const half *x [[buffer(0)]],
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
  constexpr uint rows = 4;
  uint first_row = (group * 4 + simdgroup_index) * rows;
  device const half4 *x4 = reinterpret_cast<device const half4 *>(x);
  float sums0[rows] = {};
  float sums1[rows] = {};
  float sums2[rows] = {};
  for (uint i = lane * 2; i < p.k / 4; i += 64) {
    float4 input0 = float4(x4[i]);
    float4 input1 = float4(x4[i + 1]);
    for (uint output = 0; output < rows; ++output) {
      uint row = first_row + output;
      if (row < p.n0) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight0 + row * p.k);
        sums0[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
      if (row < p.n1) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight1 + row * p.k);
        sums1[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
      if (row < p.n2) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight2 + row * p.k);
        sums2[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
    }
  }
  for (uint output = 0; output < rows; ++output) {
    uint row = first_row + output;
    float total0 = simd_sum(sums0[output]);
    float total1 = simd_sum(sums1[output]);
    float total2 = simd_sum(sums2[output]);
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

kernel void matvec2_tuned8_f16(device const half *x [[buffer(0)]],
                               device const half *weight0 [[buffer(1)]],
                               device const half *weight1 [[buffer(2)]],
                               device half *out0 [[buffer(3)]],
                               device half *out1 [[buffer(4)]],
                               constant MultiMatrixParams &p [[buffer(5)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simdgroup_index
                               [[simdgroup_index_in_threadgroup]]) {
  constexpr uint rows = 8;
  uint first_row = (group * 4 + simdgroup_index) * rows;
  device const half4 *x4 = reinterpret_cast<device const half4 *>(x);
  float sums0[rows] = {};
  float sums1[rows] = {};
  for (uint i = lane * 2; i < p.k / 4; i += 64) {
    float4 input0 = float4(x4[i]);
    float4 input1 = float4(x4[i + 1]);
    for (uint output = 0; output < rows; ++output) {
      uint row = first_row + output;
      if (row < p.n0) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight0 + row * p.k);
        sums0[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
      if (row < p.n1) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight1 + row * p.k);
        sums1[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
    }
  }
  for (uint output = 0; output < rows; ++output) {
    uint row = first_row + output;
    float total0 = simd_sum(sums0[output]);
    float total1 = simd_sum(sums1[output]);
    if (lane == 0) {
      if (row < p.n0)
        out0[row] = half(total0);
      if (row < p.n1)
        out1[row] = half(total1);
    }
  }
}

kernel void matvec3_tuned8_f16(device const half *x [[buffer(0)]],
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
  constexpr uint rows = 8;
  uint first_row = (group * 4 + simdgroup_index) * rows;
  device const half4 *x4 = reinterpret_cast<device const half4 *>(x);
  float sums0[rows] = {};
  float sums1[rows] = {};
  float sums2[rows] = {};
  for (uint i = lane * 2; i < p.k / 4; i += 64) {
    float4 input0 = float4(x4[i]);
    float4 input1 = float4(x4[i + 1]);
    for (uint output = 0; output < rows; ++output) {
      uint row = first_row + output;
      if (row < p.n0) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight0 + row * p.k);
        sums0[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
      if (row < p.n1) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight1 + row * p.k);
        sums1[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
      if (row < p.n2) {
        device const half4 *w =
            reinterpret_cast<device const half4 *>(weight2 + row * p.k);
        sums2[output] +=
            dot(input0, float4(w[i])) + dot(input1, float4(w[i + 1]));
      }
    }
  }
  for (uint output = 0; output < rows; ++output) {
    uint row = first_row + output;
    float total0 = simd_sum(sums0[output]);
    float total1 = simd_sum(sums1[output]);
    float total2 = simd_sum(sums2[output]);
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
