use metal_infer_runtime::{CoreError, DType, Tensor};

use super::{MatrixParams, checked_mul, matrix_shape, require_f16, round_up, size, to_u32};
use crate::KernelBatch;

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
        let m4 = self.kernels.is_m4_pro();
        if m == 1 {
            let vocabulary = n >= 65_536;
            let rows = if m4 && k.is_multiple_of(256) {
                self.kernels.matvec_rows_for_shape(n, k, vocabulary)
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
                    1 => "matvec_one_row_f16",
                    2 if vocabulary => "matvec_vocab_f16",
                    2 => "matvec_tuned2_f16",
                    4 => "matvec_tuned_f16",
                    _ => "matvec_f16",
                },
                &[input, weight, &out],
                &params,
                size(checked_mul(groups, threads, "matvec grid")?, 1, 1),
                size(threads, 1, 1),
            )?;
        } else if m4 && (2..128).contains(&m) && n % 32 == 0 && k % 32 == 0 && n >= 256 && k >= 256
        {
            self.dispatch(
                "matmul_skinny_f16",
                &[input, weight, &out],
                &params,
                size(
                    checked_mul(
                        checked_mul(n / 32, m.div_ceil(8), "skinny groups")?,
                        128,
                        "skinny grid",
                    )?,
                    1,
                    1,
                ),
                size(128, 1, 1),
            )?;
        } else if m4 && n % 32 == 0 && k % 32 == 0 && m >= 128 && n >= 256 && k >= 256 {
            self.dispatch(
                "matmul_simd_db_f16",
                &[input, weight, &out],
                &params,
                size(
                    checked_mul(
                        checked_mul(n / 32, m.div_ceil(32), "simd matmul groups")?,
                        128,
                        "simd matmul grid",
                    )?,
                    1,
                    1,
                ),
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
}
