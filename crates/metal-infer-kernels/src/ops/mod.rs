mod attention;
mod elementwise;
mod fused;
mod gemm;
mod gemv;
mod kv;
mod norm;
mod rope;
mod sampling;

use metal_infer_runtime::{CoreError, DType, Tensor};
use objc2_metal::MTLSize;

pub use attention::{AttentionConfig, AttentionKind};
pub use rope::QkNormRopeCacheConfig;

use crate::{KernelBatch, Kernels};

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
struct NormMultiMatrixParams {
    n0: u32,
    n1: u32,
    n2: u32,
    k: u32,
    epsilon: f32,
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

impl Kernels {
    fn immediate<T>(
        &self,
        encode: impl FnOnce(&mut KernelBatch<'_>) -> Result<T, CoreError>,
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

    pub fn attention_flash_decode_with_block(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        config: AttentionConfig,
        block_keys: usize,
    ) -> Result<Tensor, CoreError> {
        self.attention_flash_decode_with_configuration(query, key, value, config, block_keys, 256)
    }

    pub fn attention_flash_decode_with_configuration(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        config: AttentionConfig,
        block_keys: usize,
        threads: usize,
    ) -> Result<Tensor, CoreError> {
        self.immediate(|batch| {
            batch.attention_flash_decode_with_configuration(
                query, key, value, config, block_keys, threads,
            )
        })
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
