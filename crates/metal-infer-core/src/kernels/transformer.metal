#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

struct MatrixParams {
  uint m;
  uint n;
  uint k;
  uint padding;
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
struct NormMultiMatrixParams {
  uint n0;
  uint n1;
  uint n2;
  uint k;
  float epsilon;
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
  uint padding;
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

// A 32x32 output tile is shared by four SIMD-groups. Each SIMD-group computes
// a 16x16 quadrant using 8x8 matrix instructions and accumulates in FP32.
kernel void matmul_simd_f16(device const half *x [[buffer(0)]],
                            device const half *weight [[buffer(1)]],
                            device half *out [[buffer(2)]],
                            constant MatrixParams &p [[buffer(3)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            ushort thread_index [[thread_index_in_threadgroup]],
                            ushort simdgroup_index
                            [[simdgroup_index_in_threadgroup]],
                            ushort simd_lane [[thread_index_in_simdgroup]]) {
  constexpr uint tile_m = 32;
  constexpr uint tile_n = 32;
  constexpr uint tile_k = 16;
  constexpr uint tile_stride = 24;
  threadgroup half x_tile[tile_m * tile_stride];
  threadgroup half weight_tile[tile_n * tile_stride];

  uint output_row = group.y * tile_m;
  uint output_column = group.x * tile_n;
  uint simd_row = (simdgroup_index / 2) * 16;
  uint simd_column = (simdgroup_index % 2) * 16;
  simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8>(0.0f);
  simdgroup_float8x8 c01 = make_filled_simdgroup_matrix<float, 8>(0.0f);
  simdgroup_float8x8 c10 = make_filled_simdgroup_matrix<float, 8>(0.0f);
  simdgroup_float8x8 c11 = make_filled_simdgroup_matrix<float, 8>(0.0f);

  for (uint base = 0; base < p.k; base += tile_k) {
    uint load_row = thread_index / 4;
    uint load_column = (thread_index % 4) * 4;
    *reinterpret_cast<threadgroup half4 *>(x_tile + load_row * tile_stride +
                                           load_column) =
        *reinterpret_cast<device const half4 *>(
            x + (output_row + load_row) * p.k + base + load_column);
    *reinterpret_cast<threadgroup half4 *>(
        weight_tile + load_row * tile_stride + load_column) =
        *reinterpret_cast<device const half4 *>(
            weight + (output_column + load_row) * p.k + base + load_column);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint inner = 0; inner < tile_k; inner += 8) {
      simdgroup_half8x8 a0;
      simdgroup_half8x8 a1;
      simdgroup_half8x8 b0;
      simdgroup_half8x8 b1;
      simdgroup_load(a0, x_tile + simd_row * tile_stride + inner, tile_stride);
      simdgroup_load(a1, x_tile + (simd_row + 8) * tile_stride + inner,
                     tile_stride);
      simdgroup_load(b0, weight_tile + simd_column * tile_stride + inner,
                     tile_stride, ulong2(0), true);
      simdgroup_load(b1, weight_tile + (simd_column + 8) * tile_stride + inner,
                     tile_stride, ulong2(0), true);
      simdgroup_multiply_accumulate(c00, a0, b0, c00);
      simdgroup_multiply_accumulate(c01, a0, b1, c01);
      simdgroup_multiply_accumulate(c10, a1, b0, c10);
      simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  ushort quad = simd_lane / 4;
  ushort matrix_row = (quad & 4) + (simd_lane / 2) % 4;
  ushort matrix_column = (quad & 2) * 2 + (simd_lane % 2) * 2;
  device half *destination = out + (output_row + simd_row + matrix_row) * p.n +
                             output_column + simd_column + matrix_column;
  destination[0] = half(c00.thread_elements()[0]);
  destination[1] = half(c00.thread_elements()[1]);
  destination[8] = half(c01.thread_elements()[0]);
  destination[9] = half(c01.thread_elements()[1]);
  destination[8 * p.n] = half(c10.thread_elements()[0]);
  destination[8 * p.n + 1] = half(c10.thread_elements()[1]);
  destination[8 * p.n + 8] = half(c11.thread_elements()[0]);
  destination[8 * p.n + 9] = half(c11.thread_elements()[1]);
}

// Adjacent threadgroups reuse a weight tile across up to eight M tiles. The
// two threadgroup buffers let the next K tile be fetched before matrix work on
// the current tile has finished.
kernel void matmul_simd_db_f16(
    device const half *x [[buffer(0)]], device const half *weight [[buffer(1)]],
    device half *out [[buffer(2)]], constant MatrixParams &p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    ushort thread_index [[thread_index_in_threadgroup]],
    ushort simdgroup_index [[simdgroup_index_in_threadgroup]],
    ushort simd_lane [[thread_index_in_simdgroup]]) {
  constexpr uint tile_m = 32;
  constexpr uint tile_n = 32;
  constexpr uint tile_k = 32;
  constexpr uint stride = 40;
  constexpr uint swizzle_m = 8;
  threadgroup half x_tile[2][tile_m * stride];
  threadgroup half weight_tile[2][tile_n * stride];

  uint m_tiles = p.m / tile_m;
  uint n_tiles = p.n / tile_n;
  uint stripe = group / (swizzle_m * n_tiles);
  uint first_m = stripe * swizzle_m;
  uint stripe_rows = min(swizzle_m, m_tiles - first_m);
  uint within = group - stripe * swizzle_m * n_tiles;
  uint output_row = (first_m + within % stripe_rows) * tile_m;
  uint output_column = (within / stripe_rows) * tile_n;
  uint simd_row = (simdgroup_index / 2) * 16;
  uint simd_column = (simdgroup_index % 2) * 16;
  uint load_row = thread_index / 4;
  uint load_column = (thread_index % 4) * 8;

  simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8>(0.0f);
  simdgroup_float8x8 c01 = make_filled_simdgroup_matrix<float, 8>(0.0f);
  simdgroup_float8x8 c10 = make_filled_simdgroup_matrix<float, 8>(0.0f);
  simdgroup_float8x8 c11 = make_filled_simdgroup_matrix<float, 8>(0.0f);

  // Each lane copies eight halves as two aligned half4 loads.
  for (uint part = 0; part < 2; ++part) {
    uint column = load_column + part * 4;
    *reinterpret_cast<threadgroup half4 *>(x_tile[0] + load_row * stride +
                                           column) =
        *reinterpret_cast<device const half4 *>(
            x + (output_row + load_row) * p.k + column);
    *reinterpret_cast<threadgroup half4 *>(weight_tile[0] + load_row * stride +
                                           column) =
        *reinterpret_cast<device const half4 *>(
            weight + (output_column + load_row) * p.k + column);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint base = 0; base < p.k; base += tile_k) {
    uint current = (base / tile_k) & 1;
    uint next = current ^ 1;
    if (base + tile_k < p.k) {
      for (uint part = 0; part < 2; ++part) {
        uint column = load_column + part * 4;
        *reinterpret_cast<threadgroup half4 *>(x_tile[next] +
                                               load_row * stride + column) =
            *reinterpret_cast<device const half4 *>(
                x + (output_row + load_row) * p.k + base + tile_k + column);
        *reinterpret_cast<threadgroup half4 *>(weight_tile[next] +
                                               load_row * stride + column) =
            *reinterpret_cast<device const half4 *>(
                weight + (output_column + load_row) * p.k + base + tile_k +
                column);
      }
    }
    for (uint inner = 0; inner < tile_k; inner += 8) {
      simdgroup_half8x8 a0;
      simdgroup_half8x8 a1;
      simdgroup_half8x8 b0;
      simdgroup_half8x8 b1;
      simdgroup_load(a0, x_tile[current] + simd_row * stride + inner, stride);
      simdgroup_load(a1, x_tile[current] + (simd_row + 8) * stride + inner,
                     stride);
      simdgroup_load(b0, weight_tile[current] + simd_column * stride + inner,
                     stride, ulong2(0), true);
      simdgroup_load(b1,
                     weight_tile[current] + (simd_column + 8) * stride + inner,
                     stride, ulong2(0), true);
      simdgroup_multiply_accumulate(c00, a0, b0, c00);
      simdgroup_multiply_accumulate(c01, a0, b1, c01);
      simdgroup_multiply_accumulate(c10, a1, b0, c10);
      simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  ushort quad = simd_lane / 4;
  ushort matrix_row = (quad & 4) + (simd_lane / 2) % 4;
  ushort matrix_column = (quad & 2) * 2 + (simd_lane % 2) * 2;
  device half *destination = out + (output_row + simd_row + matrix_row) * p.n +
                             output_column + simd_column + matrix_column;
  destination[0] = half(c00.thread_elements()[0]);
  destination[1] = half(c00.thread_elements()[1]);
  destination[8] = half(c01.thread_elements()[0]);
  destination[9] = half(c01.thread_elements()[1]);
  destination[8 * p.n] = half(c10.thread_elements()[0]);
  destination[8 * p.n + 1] = half(c10.thread_elements()[1]);
  destination[8 * p.n + 8] = half(c11.thread_elements()[0]);
  destination[8 * p.n + 9] = half(c11.thread_elements()[1]);
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

// K=1024 decode variant: each threadgroup builds the normalized input once
// and its eight SIMD groups reuse it while streaming the projection weights.
kernel void matvec2_add_rms_shared_r1_f16(
    device const half *left [[buffer(0)]],
    device const half *right [[buffer(1)]],
    device const half *norm_weight [[buffer(2)]],
    device const half *weight0 [[buffer(3)]],
    device const half *weight1 [[buffer(4)]],
    device half *residual [[buffer(5)]], device half *out0 [[buffer(6)]],
    device half *out1 [[buffer(7)]],
    constant NormMultiMatrixParams &p [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[8];
  threadgroup float scale;
  threadgroup half4 input4[256];
  threadgroup half *shared_input = reinterpret_cast<threadgroup half *>(input4);
  uint thread_id = simdgroup_index * 32 + lane;
  float square_sum = 0.0f;
  for (uint i = thread_id; i < 1024; i += 256) {
    float value = float(left[i]) + float(right[i]);
    square_sum += value * value;
    half rounded = half(value);
    shared_input[i] = rounded;
    if (group == 0)
      residual[i] = rounded;
  }
  float simd_total = simd_sum(square_sum);
  if (lane == 0)
    partial[simdgroup_index] = simd_total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_id == 0) {
    float total = 0.0f;
    for (uint index = 0; index < 8; ++index)
      total += partial[index];
    scale = rsqrt(total / float(p.k) + p.epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  device const half4 *norm4 =
      reinterpret_cast<device const half4 *>(norm_weight);
  input4[thread_id] =
      half4(float4(input4[thread_id]) * scale * float4(norm4[thread_id]));
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint row = group * 8 + simdgroup_index;
  float sum0 = 0.0f;
  float sum1 = 0.0f;
  for (uint i = lane * 2; i < 256; i += 64) {
    float4 input0 = float4(input4[i]);
    float4 input1 = float4(input4[i + 1]);
    if (row < p.n0) {
      device const half4 *w0 =
          reinterpret_cast<device const half4 *>(weight0 + row * p.k);
      sum0 += dot(input0, float4(w0[i])) + dot(input1, float4(w0[i + 1]));
    }
    if (row < p.n1) {
      device const half4 *w1 =
          reinterpret_cast<device const half4 *>(weight1 + row * p.k);
      sum1 += dot(input0, float4(w1[i])) + dot(input1, float4(w1[i + 1]));
    }
  }
  float total0 = simd_sum(sum0);
  float total1 = simd_sum(sum1);
  if (lane == 0) {
    if (row < p.n0)
      out0[row] = half(total0);
    if (row < p.n1)
      out1[row] = half(total1);
  }
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
  if (tokens[id.y] == UINT_MAX) {
    out[id.y * p.y + id.x] = half(0.0f);
    return;
  }
  out[id.y * p.y + id.x] = table[tokens[id.y] * p.y + id.x];
}

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

kernel void attention_flash_decode_partial_128_f16(
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

kernel void
attention_flash_decode_reduce_f16(device const float *scratch [[buffer(0)]],
                                  device half *out [[buffer(1)]],
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

kernel void copy_kv_f16(device const half *source [[buffer(0)]],
                        device half *cache [[buffer(1)]],
                        constant uint3 &p [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
  uint count = p.x * p.z;
  if (id < count)
    cache[p.y * p.z + id] = source[id];
}
