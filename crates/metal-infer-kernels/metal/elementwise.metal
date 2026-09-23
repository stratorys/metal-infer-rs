#include "prelude.metal"
kernel void copy_row_f16(device const half *source [[buffer(0)]],
                         device half *destination [[buffer(1)]],
                         constant uint3 &p [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
  if (id < p.x)
    destination[p.y * p.x + id] = source[id];
}
kernel void add_f16(device const half *a [[buffer(0)]],
                    device const half *b [[buffer(1)]],
                    device half *out [[buffer(2)]],
                    constant uint &count [[buffer(3)]],
                    uint id [[thread_position_in_grid]]) {
  if (id < count)
    out[id] = a[id] + b[id];
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
