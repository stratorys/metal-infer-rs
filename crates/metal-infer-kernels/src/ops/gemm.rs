use metal_infer_runtime::{CoreError, DType, Tensor};
use objc2::AnyThread;
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};

use super::{MatrixParams, checked_mul, matrix_shape, require_f16, round_up, size, to_u32};
use crate::{KernelBatch, MatmulBackend};

impl KernelBatch<'_> {
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
        let backend = self.kernels.matmul_backend();
        let auto_m4 = backend == MatmulBackend::Auto && self.kernels.is_m4_pro();
        let native = backend == MatmulBackend::NativeMsl || auto_m4;
        let simdgroup_compatible = m % 32 == 0 && n % 32 == 0 && k % 16 == 0;
        if backend == MatmulBackend::Mps {
            self.matmul_mps(input, weight, &out, m, n, k)?;
        } else if m == 1 {
            let splits = if auto_m4 && k.is_multiple_of(256) {
                self.kernels.auto_matvec_split_k()
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
                self.kernels.auto_matvec_rows_for_shape(n, k, vocabulary)
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
                    1 if self.kernels.auto_matvec_half8() => "matvec_one_row_half8_f16",
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
                input.buffer(),
                input.offset_bytes(),
                &input_descriptor,
            )
        };
        let weight_matrix = unsafe {
            MPSMatrix::initWithBuffer_offset_descriptor(
                MPSMatrix::alloc(),
                weight.buffer(),
                weight.offset_bytes(),
                &weight_descriptor,
            )
        };
        let output_matrix = unsafe {
            MPSMatrix::initWithBuffer_offset_descriptor(
                MPSMatrix::alloc(),
                output.buffer(),
                output.offset_bytes(),
                &output_descriptor,
            )
        };
        let multiplication = unsafe {
            MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                MPSMatrixMultiplication::alloc(),
                self.kernels.context().device(),
                false,
                true,
                m,
                n,
                k,
                1.0,
                0.0,
            )
        };
        self.batch.end_compute_encoding()?;
        unsafe {
            multiplication.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                self.batch.command_buffer_ref(),
                &input_matrix,
                &weight_matrix,
                &output_matrix,
            )
        };
        self.batch.resume_compute_encoding()
    }
}
