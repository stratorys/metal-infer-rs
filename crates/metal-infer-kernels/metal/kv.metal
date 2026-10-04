#include "prelude.metal"
kernel void copy_kv_f16(device const half *source [[buffer(0)]],
                        device half *cache [[buffer(1)]],
                        constant uint3 &p [[buffer(2)]],
                        uint id [[thread_position_in_grid]]) {
  uint count = p.x * p.z;
  if (id < count)
    cache[p.y * p.z + id] = source[id];
}
