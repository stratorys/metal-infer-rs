#include "prelude.metal"
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
