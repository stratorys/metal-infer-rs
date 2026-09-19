use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Instant;

use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize,
};

use crate::{CoreError, DType, DispatchStats, MetalContext, Tensor};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionKind {
    Reference,
    Tiled,
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionConfig {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub query_offset: usize,
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
    pub fn add(
        &self,
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
        &self,
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
            self.dispatch(
                "matvec_f16",
                &[input, weight, &out],
                &params,
                size(n, 1, 1),
                size(n.min(256), 1, 1),
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

    pub fn rms_norm(
        &self,
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
            size(rows, 1, 1),
            size(rows.min(256), 1, 1),
        )?;
        Ok(out)
    }

    pub fn swiglu(
        &self,
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
        &self,
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
        &self,
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

    pub fn attention(
        &self,
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
        if kind == AttentionKind::Tiled
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
        let kernel = match kind {
            AttentionKind::Reference => "attention_reference_f16",
            AttentionKind::Tiled => "attention_tiled_f16",
        };
        self.dispatch(
            kernel,
            &[query, key, value, &out],
            &params,
            size(config.head_dim, config.query_heads, *tokens),
            size(config.head_dim, 1, 1),
        )?;
        Ok(out)
    }

    pub fn copy_into_cache(
        &self,
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
        &self,
        kernel: &str,
        tensors: &[&Tensor],
        params: &T,
        grid: MTLSize,
        threadgroup: MTLSize,
    ) -> Result<DispatchStats, CoreError> {
        let pipeline = self.pipeline(kernel)?;
        let command_buffer = self.command_buffer()?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(CoreError::Resource("compute encoder"))?;
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
        let (pointer, length): (NonNull<c_void>, usize) = unsafe { Self::bytes(params) };
        // SAFETY: Metal copies `length` bytes from a valid repr(C)/scalar value
        // while encoding.
        unsafe { encoder.setBytes_length_atIndex(pointer, length, tensors.len()) };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
        encoder.endEncoding();
        let started = Instant::now();
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        let wall_time = started.elapsed();
        if command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(command_buffer
                .error()
                .map_or(CoreError::UnknownCommand, CoreError::Command));
        }
        let gpu_seconds = (command_buffer.GPUEndTime() - command_buffer.GPUStartTime()).max(0.0);
        Ok(DispatchStats {
            gpu_time: std::time::Duration::from_secs_f64(gpu_seconds),
            wall_time,
        })
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
