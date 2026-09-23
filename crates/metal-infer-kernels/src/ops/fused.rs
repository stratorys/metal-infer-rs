use metal_infer_runtime::{CoreError, DType, Tensor};

use super::{
    NormMultiMatrixParams, checked_mul, matrix_shape, require_f16, require_same_shape, size, to_u32,
};
use crate::{KernelBatch, MatmulBackend};

impl KernelBatch<'_> {
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
        let rows =
            if self.kernels.matmul_backend() == MatmulBackend::Auto && self.kernels.is_m4_pro() {
                self.kernels
                    .fused_norm_matvec_rows_for_shape([n0, n1, n2], width)
            } else {
                2
            };
        let simdgroups = if rows == 1 { 8 } else { 4 };
        let groups = n0.max(n1).max(n2).div_ceil(rows * simdgroups);
        self.dispatch(
            match rows {
                1 => "matvec3_rms_r1_f16",
                4 => "matvec3_rms_r4_f16",
                8 => "matvec3_rms_r8_f16",
                _ => "matvec3_rms_f16",
            },
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
            size(
                checked_mul(groups, simdgroups * 32, "rms matmul3 grid")?,
                1,
                1,
            ),
            size(simdgroups * 32, 1, 1),
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
        let rows =
            if self.kernels.matmul_backend() == MatmulBackend::Auto && self.kernels.is_m4_pro() {
                self.kernels
                    .fused_norm_matvec_rows_for_shape([n0, n1, 0], width)
            } else {
                2
            };
        let simdgroups = if rows == 1 { 8 } else { 4 };
        let groups = n0.max(n1).div_ceil(rows * simdgroups);
        self.dispatch(
            match rows {
                1 => "matvec2_add_rms_r1_f16",
                4 => "matvec2_add_rms_r4_f16",
                8 => "matvec2_add_rms_r8_f16",
                _ => "matvec2_add_rms_f16",
            },
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
            size(
                checked_mul(groups, simdgroups * 32, "add rms matmul2 grid")?,
                1,
                1,
            ),
            size(simdgroups * 32, 1, 1),
        )?;
        Ok((residual, out0, out1))
    }
}
