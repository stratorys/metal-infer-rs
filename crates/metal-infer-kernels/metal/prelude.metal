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
