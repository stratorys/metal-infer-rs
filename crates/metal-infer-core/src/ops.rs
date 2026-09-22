use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::AnyThread;
use objc2_metal::{MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};

use crate::{CommandBatch, CoreError, DType, MatmulBackend, MetalContext, Tensor};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionKind {
    Reference,
    Tiled,
    FlashPrefill,
    DecodeSplitKv,
    FlashDecode,
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
        let backend = self.context.matmul_backend();
        let auto_m4 = backend == MatmulBackend::Auto && self.context.is_m4_pro();
        let native = backend == MatmulBackend::NativeMsl || auto_m4;
        let simdgroup_compatible = m % 32 == 0 && n % 32 == 0 && k % 16 == 0;
        if backend == MatmulBackend::Mps {
            self.matmul_mps(input, weight, &out, m, n, k)?;
        } else if m == 1 {
            let splits = if auto_m4 && k.is_multiple_of(256) {
                self.context.auto_matvec_split_k()
            } else {
                1
            };
            if splits > 1 {
                let partial = self.empty(&[splits, n], DType::F32)?;
                let split_params = MatrixParams {
                    m: 1,
                    n: to_u32(n, "n")?,
                    k: to_u32(k, "k")?,
                    padding: to_u32(splits, "split-K count")?,
                };
                let groups_per_split = n.div_ceil(8);
                let groups = checked_mul(groups_per_split, splits, "split-K groups")?;
                self.dispatch(
                    "matvec_splitk_partial_f16",
                    &[input, weight, &partial],
                    &split_params,
                    size(checked_mul(groups, 256, "split-K grid")?, 1, 1),
                    size(256, 1, 1),
                )?;
                self.dispatch(
                    "matvec_splitk_reduce_f16",
                    &[&partial, &out],
                    &split_params,
                    size(round_up(n, 256)?, 1, 1),
                    size(256, 1, 1),
                )?;
                return Ok(out);
            }
            let vocabulary = n >= 65_536;
            let rows = if auto_m4 && k.is_multiple_of(256) {
                let selected = self.context.auto_matvec_rows();
                if vocabulary {
                    selected.vocab
                } else {
                    self.context.auto_single_matvec_rows(n, k)
                }
            } else if native && k.is_multiple_of(256) {
                if vocabulary { 2 } else { 4 }
            } else {
                0
            };
            let tuned = rows != 0;
            let threads = if rows == 1 || !tuned { 256 } else { 128 };
            let outputs_per_threadgroup = if rows == 1 {
                8
            } else if tuned {
                rows * 4
            } else {
                32
            };
            let groups = n.div_ceil(outputs_per_threadgroup);
            self.dispatch(
                match rows {
                    1 if self.context.auto_matvec_half8() => "matvec_one_row_half8_f16",
                    1 => "matvec_one_row_f16",
                    2 if vocabulary => "matvec_vocab_f16",
                    2 => "matvec_tuned2_f16",
                    4 => "matvec_tuned_f16",
                    8 => "matvec_tuned8_f16",
                    _ => "matvec_f16",
                },
                &[input, weight, &out],
                &params,
                size(checked_mul(groups, threads, "matvec grid")?, 1, 1),
                size(threads, 1, 1),
            )?;
        } else if native
            && m % 32 == 0
            && n % 32 == 0
            && k % 32 == 0
            && (!auto_m4 || (m >= 128 && n >= 256 && k >= 256))
        {
            self.dispatch(
                "matmul_simd_db_f16",
                &[input, weight, &out],
                &params,
                size(
                    checked_mul(
                        checked_mul(n / 32, m / 32, "simd matmul groups")?,
                        128,
                        "simd matmul grid",
                    )?,
                    1,
                    1,
                ),
                size(128, 1, 1),
            )?;
        } else if backend == MatmulBackend::NativeMsl && simdgroup_compatible {
            self.dispatch(
                "matmul_simd_f16",
                &[input, weight, &out],
                &params,
                size(checked_mul(n / 32, 128, "simd matmul grid")?, m / 32, 1),
                size(128, 1, 1),
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

    fn matmul_mps(
        &mut self,
        input: &Tensor,
        weight: &Tensor,
        output: &Tensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), CoreError> {
        let descriptor = |rows, columns, row_bytes| unsafe {
            MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
                rows,
                columns,
                row_bytes,
                MPSDataType::Float16,
            )
        };
        let input_descriptor = descriptor(m, k, k * DType::F16.size());
        let weight_descriptor = descriptor(n, k, k * DType::F16.size());
        let output_descriptor = descriptor(m, n, n * DType::F16.size());
        let input_matrix = unsafe {
            MPSMatrix::initWithBuffer_offset_descriptor(
                MPSMatrix::alloc(),
                &input.buffer,
                input.offset_bytes,
                &input_descriptor,
            )
        };
        let weight_matrix = unsafe {
            MPSMatrix::initWithBuffer_offset_descriptor(
                MPSMatrix::alloc(),
                &weight.buffer,
                weight.offset_bytes,
                &weight_descriptor,
            )
        };
        let output_matrix = unsafe {
            MPSMatrix::initWithBuffer_offset_descriptor(
                MPSMatrix::alloc(),
                &output.buffer,
                output.offset_bytes,
                &output_descriptor,
            )
        };
        let multiplication = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                &self.context.device,
                false,
                true,
                m,
                n,
                k,
                1.0,
                0.0,
            )
        };
        self.end_compute_encoding()?;
        unsafe {
            multiplication.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                self.command_buffer_ref(),
                &input_matrix,
                &weight_matrix,
                &output_matrix,
            )
        };
        self.resume_compute_encoding()
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
        let backend = self.context.matmul_backend();
        let rows = if k.is_multiple_of(256) {
            if backend == MatmulBackend::NativeMsl {
                2
            } else if backend == MatmulBackend::Auto && self.context.is_m4_pro() {
                self.context.auto_matvec_rows().fused2
            } else {
                0
            }
        } else {
            0
        };
        let tuned = rows != 0;
        let rows_per_group = if tuned { rows * 4 } else { 32 };
        let threads = if tuned { 128 } else { 256 };
        let groups = outputs.div_ceil(rows_per_group);
        self.dispatch(
            match rows {
                2 => "matvec2_tuned_f16",
                4 => "matvec2_tuned4_f16",
                8 => "matvec2_tuned8_f16",
                _ => "matvec2_f16",
            },
            &[input, weight0, weight1, &out0, &out1],
            &params,
            size(checked_mul(groups, threads, "matmul2 grid")?, 1, 1),
            size(threads, 1, 1),
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
        let backend = self.context.matmul_backend();
        let rows = if k.is_multiple_of(256) {
            if backend == MatmulBackend::NativeMsl {
                2
            } else if backend == MatmulBackend::Auto && self.context.is_m4_pro() {
                self.context.auto_matvec_rows().fused3
            } else {
                0
            }
        } else {
            0
        };
        let tuned = rows != 0;
        let rows_per_group = if tuned { rows * 4 } else { 32 };
        let threads = if tuned { 128 } else { 256 };
        let groups = outputs.div_ceil(rows_per_group);
        self.dispatch(
            match rows {
                2 => "matvec3_tuned_f16",
                4 => "matvec3_tuned4_f16",
                8 => "matvec3_tuned8_f16",
                _ => "matvec3_f16",
            },
            &[input, weight0, weight1, weight2, &out0, &out1, &out2],
            &params,
            size(checked_mul(groups, threads, "matmul3 grid")?, 1, 1),
            size(threads, 1, 1),
        )?;
        Ok((out0, out1, out2))
    }

    pub fn rms_norm_matmul3(
        &mut self,
        input: &Tensor,
        norm_weight: &Tensor,
        weight0: &Tensor,
        weight1: &Tensor,
        weight2: &Tensor,
        epsilon: f32,
    ) -> Result<(Tensor, Tensor, Tensor), CoreError> {
        require_f16(input)?;
        require_f16(norm_weight)?;
        require_f16(weight0)?;
        require_f16(weight1)?;
        require_f16(weight2)?;
        let [rows, width] = matrix_shape(input)?;
        let [n0, k0] = matrix_shape(weight0)?;
        let [n1, k1] = matrix_shape(weight1)?;
        let [n2, k2] = matrix_shape(weight2)?;
        if rows != 1
            || !width.is_multiple_of(256)
            || norm_weight.shape() != [width]
            || k0 != width
            || k1 != width
            || k2 != width
        {
            return Err(CoreError::Shape(
                "rms_norm_matmul3 requires one row and matching widths divisible by 256".into(),
            ));
        }
        let out0 = self.empty(&[1, n0], DType::F16)?;
        let out1 = self.empty(&[1, n1], DType::F16)?;
        let out2 = self.empty(&[1, n2], DType::F16)?;
        let params = NormMultiMatrixParams {
            n0: to_u32(n0, "n0")?,
            n1: to_u32(n1, "n1")?,
            n2: to_u32(n2, "n2")?,
            k: to_u32(width, "width")?,
            epsilon,
        };
        let groups = n0.max(n1).max(n2).div_ceil(8);
        self.dispatch(
            "matvec3_rms_f16",
            &[
                input,
                norm_weight,
                weight0,
                weight1,
                weight2,
                &out0,
                &out1,
                &out2,
            ],
            &params,
            size(checked_mul(groups, 128, "rms matmul3 grid")?, 1, 1),
            size(128, 1, 1),
        )?;
        Ok((out0, out1, out2))
    }

    pub fn add_rms_norm_matmul2(
        &mut self,
        left: &Tensor,
        right: &Tensor,
        norm_weight: &Tensor,
        weight0: &Tensor,
        weight1: &Tensor,
        epsilon: f32,
    ) -> Result<(Tensor, Tensor, Tensor), CoreError> {
        require_f16(left)?;
        require_f16(right)?;
        require_f16(norm_weight)?;
        require_f16(weight0)?;
        require_f16(weight1)?;
        require_same_shape(left, right)?;
        let [rows, width] = matrix_shape(left)?;
        let [n0, k0] = matrix_shape(weight0)?;
        let [n1, k1] = matrix_shape(weight1)?;
        if rows != 1
            || !width.is_multiple_of(256)
            || norm_weight.shape() != [width]
            || k0 != width
            || k1 != width
        {
            return Err(CoreError::Shape(
                "add_rms_norm_matmul2 requires one row and matching widths divisible by 256".into(),
            ));
        }
        let residual = self.empty(&[1, width], DType::F16)?;
        let out0 = self.empty(&[1, n0], DType::F16)?;
        let out1 = self.empty(&[1, n1], DType::F16)?;
        let params = NormMultiMatrixParams {
            n0: to_u32(n0, "n0")?,
            n1: to_u32(n1, "n1")?,
            n2: 0,
            k: to_u32(width, "width")?,
            epsilon,
        };
        let groups = n0.max(n1).div_ceil(8);
        self.dispatch(
            "matvec2_add_rms_f16",
            &[
                left,
                right,
                norm_weight,
                weight0,
                weight1,
                &residual,
                &out0,
                &out1,
            ],
            &params,
            size(checked_mul(groups, 128, "add rms matmul2 grid")?, 1, 1),
            size(128, 1, 1),
        )?;
        Ok((residual, out0, out1))
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
        self.attention_with_flash_configuration(query, key, value, config, kind, None)
    }

    pub fn attention_flash_decode_with_block(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        config: AttentionConfig,
        block_keys: usize,
    ) -> Result<Tensor, CoreError> {
        self.attention_flash_decode_with_configuration(query, key, value, config, block_keys, 256)
    }

    pub fn attention_flash_decode_with_configuration(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        config: AttentionConfig,
        block_keys: usize,
        threads: usize,
    ) -> Result<Tensor, CoreError> {
        self.attention_with_flash_configuration(
            query,
            key,
            value,
            config,
            AttentionKind::FlashDecode,
            Some((block_keys, threads)),
        )
    }

    fn attention_with_flash_configuration(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        config: AttentionConfig,
        kind: AttentionKind,
        flash_configuration: Option<(usize, usize)>,
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
        let (flash_block_keys, flash_threads) = if kind == AttentionKind::FlashDecode {
            let (block, threads) = flash_configuration.unwrap_or_else(|| {
                self.context
                    .flash_decode_configuration_for_length(*kv_length)
            });
            if !matches!(block, 32 | 64 | 128 | 256) {
                return Err(CoreError::Shape(
                    "flash decode block size must be 32, 64, 128, or 256".into(),
                ));
            }
            if !matches!(threads, 128 | 256) {
                return Err(CoreError::Shape(
                    "flash decode threadgroup size must be 128 or 256".into(),
                ));
            }
            (block, threads)
        } else {
            (0, 0)
        };
        let out = self.empty(query.shape(), DType::F16)?;
        let params = AttentionParams {
            tokens: to_u32(*tokens, "tokens")?,
            q_heads: to_u32(config.query_heads, "query_heads")?,
            kv_heads: to_u32(config.kv_heads, "kv_heads")?,
            head_dim: to_u32(config.head_dim, "head_dim")?,
            causal: u32::from(config.causal),
            query_offset: to_u32(config.query_offset, "query_offset")?,
            kv_length: to_u32(*kv_length, "kv_length")?,
            padding: to_u32(flash_block_keys, "flash decode block size")?,
        };
        if matches!(
            kind,
            AttentionKind::DecodeSplitKv | AttentionKind::FlashDecode
        ) && *tokens != 1
        {
            return Err(CoreError::Shape(
                "decode attention requires exactly one query token".into(),
            ));
        }
        if kind == AttentionKind::FlashDecode {
            if config.query_heads / config.kv_heads != 2 {
                return Err(CoreError::Shape(
                    "flash decode requires two query heads per KV head".into(),
                ));
            }
            let available = if config.causal {
                (*kv_length).min(config.query_offset.saturating_add(1))
            } else {
                *kv_length
            };
            if available == 0 {
                return Err(CoreError::Shape(
                    "flash decode requires at least one available key".into(),
                ));
            }
            let blocks = available.div_ceil(flash_block_keys);
            let partial_width = config
                .head_dim
                .checked_add(2)
                .ok_or_else(|| CoreError::Shape("flash decode partial width overflow".into()))?;
            let scratch = self.empty(&[config.kv_heads, blocks, 2, partial_width], DType::F32)?;
            let groups = checked_mul(config.kv_heads, blocks, "flash decode groups")?;
            self.dispatch(
                if flash_threads == 128 {
                    "attention_flash_decode_partial_128_f16"
                } else {
                    "attention_flash_decode_partial_f16"
                },
                &[query, key, value, &scratch],
                &params,
                size(
                    checked_mul(groups, flash_threads, "flash decode grid")?,
                    1,
                    1,
                ),
                size(flash_threads, 1, 1),
            )?;
            self.dispatch(
                "attention_flash_decode_reduce_f16",
                &[&scratch, &out],
                &params,
                size(
                    checked_mul(config.query_heads, 32, "flash decode reduction grid")?,
                    1,
                    1,
                ),
                size(32, 1, 1),
            )?;
            return Ok(out);
        }
        let kernel = match kind {
            AttentionKind::Reference => "attention_reference_f16",
            AttentionKind::Tiled => "attention_tiled_f16",
            AttentionKind::FlashPrefill => "attention_flash_prefill_f16",
            AttentionKind::DecodeSplitKv => "attention_decode_f16",
            AttentionKind::FlashDecode => unreachable!("handled above"),
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
            AttentionKind::FlashPrefill => (
                size(
                    checked_mul(tokens.div_ceil(32), 128, "flash prefill grid")?,
                    config.query_heads,
                    1,
                ),
                size(128, 1, 1),
            ),
            AttentionKind::FlashDecode => unreachable!("handled above"),
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
        let profile_index = self
            .profile
            .as_mut()
            .map(|profile| profile.reserve(kernel))
            .transpose()?;
        let profiled_encoder = profile_index
            .map(|index| self.profiled_encoder(index))
            .transpose()?;
        let encoder = if let Some(encoder) = &profiled_encoder {
            encoder.as_ref()
        } else {
            self.encoder()?
        };
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
        if let Some(encoder) = profiled_encoder {
            encoder.endEncoding();
        }
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
