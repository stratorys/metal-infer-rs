use std::ffi::c_void;
use std::ptr::NonNull;

use objc2_metal::{MTLComputeCommandEncoder, MTLSize};

use crate::{CommandBatch, CoreError, DType, MetalContext, Tensor};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionKind {
    Reference,
    Tiled,
    DecodeSplitKv,
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionConfig {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub query_offset: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct QkNormRopeCacheConfig {
    pub offset: usize,
    pub theta: f32,
    pub epsilon: f32,
}

#[repr(C)]
struct MatrixParams {
    m: u32,
    n: u32,
    k: u32,
    padding: u32,
}

#[repr(C)]
struct NormParams {
    rows: u32,
    width: u32,
    epsilon: f32,
    padding: u32,
}

#[repr(C)]
struct MultiMatrixParams {
    n0: u32,
    n1: u32,
    n2: u32,
    k: u32,
}

#[repr(C)]
struct QkTransformParams {
    tokens: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    offset: u32,
    theta: f32,
    cache_capacity: u32,
    epsilon: f32,
}

#[repr(C)]
struct RopeParams {
    tokens: u32,
    heads: u32,
    head_dim: u32,
    offset: u32,
    theta: f32,
    padding: [u32; 3],
}

#[repr(C)]
struct AttentionParams {
    tokens: u32,
    q_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    causal: u32,
    query_offset: u32,
    kv_length: u32,
    padding: u32,
}

impl MetalContext {
    fn immediate<T>(
        &self,
        encode: impl FnOnce(&mut CommandBatch<'_>) -> Result<T, CoreError>,
    ) -> Result<T, CoreError> {
        let mut batch = self.begin_batch()?;
        let output = encode(&mut batch)?;
        batch.finish()?;
        Ok(output)
    }

    pub fn add(
        &self,
        left: &Tensor,
        right: &Tensor,
    ) -> Result<Tensor, CoreError> {
        self.immediate(|batch| batch.add(left, right))
    }

    pub fn matmul(
        &self,
        input: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor, CoreError> {
        self.immediate(|batch| batch.matmul(input, weight))
    }

    pub fn rms_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        epsilon: f32,
    ) -> Result<Tensor, CoreError> {
        self.immediate(|batch| batch.rms_norm(input, weight, epsilon))
    }

    pub fn swiglu(
        &self,
        gate: &Tensor,
        up: &Tensor,
    ) -> Result<Tensor, CoreError> {
        self.immediate(|batch| batch.swiglu(gate, up))
    }

    pub fn embedding(
        &self,
        tokens: &Tensor,
        table: &Tensor,
    ) -> Result<Tensor, CoreError> {
        self.immediate(|batch| batch.embedding(tokens, table))
    }

    pub fn rope(
        &self,
        input: &Tensor,
        offset: usize,
        theta: f32,
    ) -> Result<Tensor, CoreError> {
        self.immediate(|batch| batch.rope(input, offset, theta))
    }

    pub fn attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        config: AttentionConfig,
        kind: AttentionKind,
    ) -> Result<Tensor, CoreError> {
        self.immediate(|batch| batch.attention(query, key, value, config, kind))
    }

    pub fn copy_into_cache(
        &self,
        source: &Tensor,
        cache: &Tensor,
        offset: usize,
    ) -> Result<(), CoreError> {
        self.immediate(|batch| batch.copy_into_cache(source, cache, offset))
    }
}

impl CommandBatch<'_> {
    pub fn add(
        &mut self,
        left: &Tensor,
        right: &Tensor,
    ) -> Result<Tensor, CoreError> {
        require_f16(left)?;
        require_f16(right)?;
        require_same_shape(left, right)?;
        let out = self.empty(left.shape(), DType::F16)?;
        let count = to_u32(left.len(), "element count")?;
        self.dispatch(
            "add_f16",
            &[left, right, &out],
            &count,
            size(left.len(), 1, 1),
            size(left.len().min(256), 1, 1),
        )?;
        Ok(out)
    }

    pub fn matmul(
        &mut self,
        input: &Tensor,
        weight: &Tensor,
    ) -> Result<Tensor, CoreError> {
        require_f16(input)?;
        require_f16(weight)?;
        let [m, k] = matrix_shape(input)?;
        let [n, weight_k] = matrix_shape(weight)?;
        if k != weight_k {
            return Err(CoreError::Shape(format!(
                "matmul inner dimensions differ: {k} and {weight_k}"
            )));
        }
        let out = self.empty(&[m, n], DType::F16)?;
        let params = MatrixParams {
            m: to_u32(m, "m")?,
            n: to_u32(n, "n")?,
            k: to_u32(k, "k")?,
            padding: 0,
        };
        if m == 1 {
            let outputs_per_threadgroup = 32;
            let groups = n.div_ceil(outputs_per_threadgroup);
            self.dispatch(
                "matvec_f16",
                &[input, weight, &out],
                &params,
                size(checked_mul(groups, 256, "matvec grid")?, 1, 1),
                size(256, 1, 1),
            )?;
        } else {
            self.dispatch(
                "matmul_f16",
                &[input, weight, &out],
                &params,
                size(round_up(n, 16)?, round_up(m, 16)?, 1),
                size(16, 16, 1),
            )?;
        }
        Ok(out)
    }

    pub fn matmul2(
        &mut self,
        input: &Tensor,
        weight0: &Tensor,
        weight1: &Tensor,
    ) -> Result<(Tensor, Tensor), CoreError> {
        let [m, k] = matrix_shape(input)?;
        let [n0, k0] = matrix_shape(weight0)?;
        let [n1, k1] = matrix_shape(weight1)?;
        require_f16(input)?;
        require_f16(weight0)?;
        require_f16(weight1)?;
        if k != k0 || k != k1 {
            return Err(CoreError::Shape("matmul2 inner dimensions differ".into()));
        }
        if m != 1 {
            return Ok((self.matmul(input, weight0)?, self.matmul(input, weight1)?));
        }
        let out0 = self.empty(&[1, n0], DType::F16)?;
        let out1 = self.empty(&[1, n1], DType::F16)?;
        let params = MultiMatrixParams {
            n0: to_u32(n0, "n0")?,
            n1: to_u32(n1, "n1")?,
            n2: 0,
            k: to_u32(k, "k")?,
        };
        let outputs = n0.max(n1);
        let groups = outputs.div_ceil(32);
        self.dispatch(
            "matvec2_f16",
            &[input, weight0, weight1, &out0, &out1],
            &params,
            size(checked_mul(groups, 256, "matmul2 grid")?, 1, 1),
            size(256, 1, 1),
        )?;
        Ok((out0, out1))
    }

    pub fn matmul3(
        &mut self,
        input: &Tensor,
        weight0: &Tensor,
        weight1: &Tensor,
        weight2: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), CoreError> {
        let [m, k] = matrix_shape(input)?;
        let [n0, k0] = matrix_shape(weight0)?;
        let [n1, k1] = matrix_shape(weight1)?;
        let [n2, k2] = matrix_shape(weight2)?;
        require_f16(input)?;
        require_f16(weight0)?;
        require_f16(weight1)?;
        require_f16(weight2)?;
        if k != k0 || k != k1 || k != k2 {
            return Err(CoreError::Shape("matmul3 inner dimensions differ".into()));
        }
        if m != 1 {
            return Ok((
                self.matmul(input, weight0)?,
                self.matmul(input, weight1)?,
                self.matmul(input, weight2)?,
            ));
        }
        let out0 = self.empty(&[1, n0], DType::F16)?;
        let out1 = self.empty(&[1, n1], DType::F16)?;
        let out2 = self.empty(&[1, n2], DType::F16)?;
        let params = MultiMatrixParams {
            n0: to_u32(n0, "n0")?,
            n1: to_u32(n1, "n1")?,
            n2: to_u32(n2, "n2")?,
            k: to_u32(k, "k")?,
        };
        let outputs = n0.max(n1).max(n2);
        let groups = outputs.div_ceil(32);
        self.dispatch(
            "matvec3_f16",
            &[input, weight0, weight1, weight2, &out0, &out1, &out2],
            &params,
            size(checked_mul(groups, 256, "matmul3 grid")?, 1, 1),
            size(256, 1, 1),
        )?;
        Ok((out0, out1, out2))
    }

    pub fn rms_norm(
        &mut self,
        input: &Tensor,
        weight: &Tensor,
        epsilon: f32,
    ) -> Result<Tensor, CoreError> {
        require_f16(input)?;
        require_f16(weight)?;
        let width = *input
            .shape()
            .last()
            .ok_or_else(|| CoreError::Shape("rms_norm requires rank >= 1".into()))?;
        if weight.shape() != [width] {
            return Err(CoreError::Shape(format!(
                "rms_norm weight must have shape [{width}]"
            )));
        }
        let rows = input.len() / width;
        let out = self.empty(input.shape(), DType::F16)?;
        let params = NormParams {
            rows: to_u32(rows, "rows")?,
            width: to_u32(width, "width")?,
            epsilon,
            padding: 0,
        };
        self.dispatch(
            "rms_norm_f16",
            &[input, weight, &out],
            &params,
            size(checked_mul(rows.div_ceil(8), 256, "RMSNorm grid")?, 1, 1),
            size(256, 1, 1),
        )?;
        Ok(out)
    }

    pub fn add_rms_norm(
        &mut self,
        left: &Tensor,
        right: &Tensor,
        weight: &Tensor,
        epsilon: f32,
    ) -> Result<(Tensor, Tensor), CoreError> {
        require_f16(left)?;
        require_f16(right)?;
        require_f16(weight)?;
        require_same_shape(left, right)?;
        let width = *left
            .shape()
            .last()
            .ok_or_else(|| CoreError::Shape("add_rms_norm requires rank >= 1".into()))?;
        if weight.shape() != [width] {
            return Err(CoreError::Shape(format!(
                "RMSNorm weight must have shape [{width}]"
            )));
        }
        let rows = left.len() / width;
        let residual = self.empty(left.shape(), DType::F16)?;
        let normalized = self.empty(left.shape(), DType::F16)?;
        let params = NormParams {
            rows: to_u32(rows, "rows")?,
            width: to_u32(width, "width")?,
            epsilon,
            padding: 0,
        };
        self.dispatch(
            "add_rms_norm_f16",
            &[left, right, weight, &residual, &normalized],
            &params,
            size(
                checked_mul(rows.div_ceil(8), 256, "add RMSNorm grid")?,
                1,
                1,
            ),
            size(256, 1, 1),
        )?;
        Ok((residual, normalized))
    }

    pub fn swiglu(
        &mut self,
        gate: &Tensor,
        up: &Tensor,
    ) -> Result<Tensor, CoreError> {
        require_f16(gate)?;
        require_f16(up)?;
        require_same_shape(gate, up)?;
        let out = self.empty(gate.shape(), DType::F16)?;
        let count = to_u32(gate.len(), "element count")?;
        self.dispatch(
            "swiglu_f16",
            &[gate, up, &out],
            &count,
            size(gate.len(), 1, 1),
            size(gate.len().min(256), 1, 1),
        )?;
        Ok(out)
    }

    pub fn embedding(
        &mut self,
        tokens: &Tensor,
        table: &Tensor,
    ) -> Result<Tensor, CoreError> {
        if tokens.dtype() != DType::U32 || tokens.shape().len() != 1 {
            return Err(CoreError::Shape(
                "tokens must be a rank-1 u32 tensor".into(),
            ));
        }
        require_f16(table)?;
        let [_, width] = matrix_shape(table)?;
        let out = self.empty(&[tokens.len(), width], DType::F16)?;
        let params = [
            to_u32(tokens.len(), "token count")?,
            to_u32(width, "embedding width")?,
        ];
        self.dispatch(
            "embedding_f16",
            &[tokens, table, &out],
            &params,
            size(width, tokens.len(), 1),
            size(width.min(256), 1, 1),
        )?;
        Ok(out)
    }

    pub fn rope(
        &mut self,
        input: &Tensor,
        offset: usize,
        theta: f32,
    ) -> Result<Tensor, CoreError> {
        require_f16(input)?;
        let shape = input.shape();
        let [tokens, heads, head_dim] = shape else {
            return Err(CoreError::Shape(
                "RoPE expects [tokens, heads, head_dim]".into(),
            ));
        };
        if !head_dim.is_multiple_of(2) {
            return Err(CoreError::Shape("RoPE head_dim must be even".into()));
        }
        let out = self.empty(shape, DType::F16)?;
        let params = RopeParams {
            tokens: to_u32(*tokens, "tokens")?,
            heads: to_u32(*heads, "heads")?,
            head_dim: to_u32(*head_dim, "head_dim")?,
            offset: to_u32(offset, "offset")?,
            theta,
            padding: [0; 3],
        };
        self.dispatch(
            "rope_f16",
            &[input, &out],
            &params,
            size(*head_dim / 2, *heads, *tokens),
            size((*head_dim / 2).min(32), 1, 1),
        )?;
        Ok(out)
    }

    pub fn qk_norm_rope_cache(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        query_weight: &Tensor,
        key_weight: &Tensor,
        key_cache: &Tensor,
        config: QkNormRopeCacheConfig,
    ) -> Result<Tensor, CoreError> {
        require_f16(query)?;
        require_f16(key)?;
        require_f16(query_weight)?;
        require_f16(key_weight)?;
        require_f16(key_cache)?;
        let [tokens, query_heads, head_dim] = query.shape() else {
            return Err(CoreError::Shape("query must have rank 3".into()));
        };
        let [key_tokens, kv_heads, key_dim] = key.shape() else {
            return Err(CoreError::Shape("key must have rank 3".into()));
        };
        let [capacity, cache_heads, cache_dim] = key_cache.shape() else {
            return Err(CoreError::Shape("key cache must have rank 3".into()));
        };
        if tokens != key_tokens
            || head_dim != key_dim
            || kv_heads != cache_heads
            || head_dim != cache_dim
            || query_weight.shape() != [*head_dim]
            || key_weight.shape() != [*head_dim]
        {
            return Err(CoreError::Shape(
                "Q/K transform tensor shapes are incompatible".into(),
            ));
        }
        if !head_dim.is_multiple_of(2) || config.offset + *tokens > *capacity {
            return Err(CoreError::Shape(
                "Q/K transform has invalid head_dim or cache offset".into(),
            ));
        }
        let out = self.empty(query.shape(), DType::F16)?;
        let params = QkTransformParams {
            tokens: to_u32(*tokens, "tokens")?,
            query_heads: to_u32(*query_heads, "query heads")?,
            kv_heads: to_u32(*kv_heads, "KV heads")?,
            head_dim: to_u32(*head_dim, "head dim")?,
            offset: to_u32(config.offset, "offset")?,
            theta: config.theta,
            cache_capacity: to_u32(*capacity, "cache capacity")?,
            epsilon: config.epsilon,
        };
        let heads = checked_add(*query_heads, *kv_heads, "Q/K heads")?;
        let groups = checked_mul(*tokens, heads, "Q/K groups")?;
        self.dispatch(
            "qk_norm_rope_cache_f16",
            &[query, key, query_weight, key_weight, &out, key_cache],
            &params,
            size(checked_mul(groups, 32, "Q/K grid")?, 1, 1),
            size(32, 1, 1),
        )?;
        Ok(out)
    }

    pub fn attention(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        config: AttentionConfig,
        kind: AttentionKind,
    ) -> Result<Tensor, CoreError> {
        require_f16(query)?;
        require_f16(key)?;
        require_f16(value)?;
        let [tokens, query_heads, query_dim] = query.shape() else {
            return Err(CoreError::Shape("query must have rank 3".into()));
        };
        let [kv_length, key_heads, key_dim] = key.shape() else {
            return Err(CoreError::Shape("key must have rank 3".into()));
        };
        let [value_length, value_heads, value_dim] = value.shape() else {
            return Err(CoreError::Shape("value must have rank 3".into()));
        };
        if *query_heads != config.query_heads
            || *query_dim != config.head_dim
            || *key_heads != config.kv_heads
            || *key_dim != config.head_dim
            || value_length != kv_length
            || value_heads != key_heads
            || value_dim != key_dim
        {
            return Err(CoreError::Shape(
                "attention tensor shapes do not match config".into(),
            ));
        }
        if config.kv_heads == 0 || !config.query_heads.is_multiple_of(config.kv_heads) {
            return Err(CoreError::Shape(
                "query_heads must be divisible by kv_heads".into(),
            ));
        }
        if kind != AttentionKind::Reference
            && (!config.head_dim.is_power_of_two() || config.head_dim > 256)
        {
            return Err(CoreError::Shape(
                "tiled attention requires a power-of-two head_dim <= 256".into(),
            ));
        }
        let out = self.empty(query.shape(), DType::F16)?;
        let params = AttentionParams {
            tokens: to_u32(*tokens, "tokens")?,
            q_heads: to_u32(config.query_heads, "query_heads")?,
            kv_heads: to_u32(config.kv_heads, "kv_heads")?,
            head_dim: to_u32(config.head_dim, "head_dim")?,
            causal: u32::from(config.causal),
            query_offset: to_u32(config.query_offset, "query_offset")?,
            kv_length: to_u32(*kv_length, "kv_length")?,
            padding: 0,
        };
        if kind == AttentionKind::DecodeSplitKv && *tokens != 1 {
            return Err(CoreError::Shape(
                "split-KV decode attention requires exactly one query token".into(),
            ));
        }
        let kernel = match kind {
            AttentionKind::Reference => "attention_reference_f16",
            AttentionKind::Tiled => "attention_tiled_f16",
            AttentionKind::DecodeSplitKv => "attention_decode_f16",
        };
        let (grid, threadgroup) = match kind {
            AttentionKind::Reference => (
                size(config.head_dim, config.query_heads, *tokens),
                size(config.head_dim.min(256), 1, 1),
            ),
            AttentionKind::DecodeSplitKv => {
                let groups = checked_mul(config.query_heads, *tokens, "attention groups")?;
                (
                    size(checked_mul(groups, 256, "decode attention grid")?, 1, 1),
                    size(256, 1, 1),
                )
            }
            AttentionKind::Tiled => {
                let groups = checked_mul(config.query_heads, *tokens, "attention groups")?;
                (
                    size(checked_mul(groups, 32, "attention grid")?, 1, 1),
                    size(32, 1, 1),
                )
            }
        };
        self.dispatch(
            kernel,
            &[query, key, value, &out],
            &params,
            grid,
            threadgroup,
        )?;
        Ok(out)
    }

    pub fn copy_into_cache(
        &mut self,
        source: &Tensor,
        cache: &Tensor,
        offset: usize,
    ) -> Result<(), CoreError> {
        require_f16(source)?;
        require_f16(cache)?;
        let [source_tokens, source_heads, source_dim] = source.shape() else {
            return Err(CoreError::Shape("KV source must have rank 3".into()));
        };
        let [cache_tokens, cache_heads, cache_dim] = cache.shape() else {
            return Err(CoreError::Shape("KV cache must have rank 3".into()));
        };
        if source_heads != cache_heads || source_dim != cache_dim {
            return Err(CoreError::Shape(
                "KV source/cache shapes are incompatible".into(),
            ));
        }
        if offset + *source_tokens > *cache_tokens {
            return Err(CoreError::Shape("KV cache capacity exceeded".into()));
        }
        let stride = source_heads * source_dim;
        let params = [
            to_u32(*source_tokens, "tokens")?,
            to_u32(offset, "offset")?,
            to_u32(stride, "stride")?,
        ];
        self.dispatch(
            "copy_kv_f16",
            &[source, cache],
            &params,
            size(source.len(), 1, 1),
            size(source.len().min(256), 1, 1),
        )?;
        Ok(())
    }

    fn dispatch<T>(
        &mut self,
        kernel: &str,
        tensors: &[&Tensor],
        params: &T,
        grid: MTLSize,
        threadgroup: MTLSize,
    ) -> Result<(), CoreError> {
        let pipeline = self.context.pipeline(kernel)?;
        let encoder = self.encoder()?;
        encoder.setComputePipelineState(&pipeline);
        for (index, tensor) in tensors.iter().enumerate() {
            // SAFETY: tensor resources remain alive through command completion
            // and each kernel's binding order is fixed by its safe
            // wrapper above.
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(tensor.buffer.as_ref()),
                    tensor.offset_bytes,
                    index,
                )
            };
        }
        let (pointer, length): (NonNull<c_void>, usize) = unsafe { MetalContext::bytes(params) };
        // SAFETY: Metal copies `length` bytes from a valid repr(C)/scalar value
        // while encoding.
        unsafe { encoder.setBytes_length_atIndex(pointer, length, tensors.len()) };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
        Ok(())
    }
}

fn require_f16(tensor: &Tensor) -> Result<(), CoreError> {
    if tensor.dtype() == DType::F16 {
        Ok(())
    } else {
        Err(CoreError::DType {
            expected: "f16",
            actual: tensor.dtype().name(),
        })
    }
}

fn require_same_shape(
    left: &Tensor,
    right: &Tensor,
) -> Result<(), CoreError> {
    if left.shape() == right.shape() {
        Ok(())
    } else {
        Err(CoreError::Shape(format!(
            "shape mismatch: {:?} and {:?}",
            left.shape(),
            right.shape()
        )))
    }
}

fn matrix_shape(tensor: &Tensor) -> Result<[usize; 2], CoreError> {
    tensor
        .shape()
        .try_into()
        .map_err(|_| CoreError::Shape(format!("expected matrix, got {:?}", tensor.shape())))
}

fn to_u32(
    value: usize,
    label: &str,
) -> Result<u32, CoreError> {
    value
        .try_into()
        .map_err(|_| CoreError::Shape(format!("{label} does not fit in u32")))
}

fn checked_mul(
    left: usize,
    right: usize,
    label: &str,
) -> Result<usize, CoreError> {
    left.checked_mul(right)
        .ok_or_else(|| CoreError::Shape(format!("{label} overflow")))
}

fn checked_add(
    left: usize,
    right: usize,
    label: &str,
) -> Result<usize, CoreError> {
    left.checked_add(right)
        .ok_or_else(|| CoreError::Shape(format!("{label} overflow")))
}

fn round_up(
    value: usize,
    multiple: usize,
) -> Result<usize, CoreError> {
    value
        .checked_add(multiple - 1)
        .map(|rounded| rounded / multiple * multiple)
        .ok_or_else(|| CoreError::Shape("dispatch size overflow".into()))
}

const fn size(
    width: usize,
    height: usize,
    depth: usize,
) -> MTLSize {
    MTLSize {
        width,
        height,
        depth,
    }
}
