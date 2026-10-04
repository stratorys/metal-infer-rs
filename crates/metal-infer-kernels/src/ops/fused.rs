use super::{
    NormMultiMatrixParams, checked_mul, matrix_shape, require_f16, require_same_shape, size, to_u32,
};
use crate::kernels::{dispatch, tuning};
use crate::{DType, KernelBatch, KernelError, Tensor};

impl KernelBatch<'_> {
    pub fn rms_norm_matmul3(
        &mut self,
        input: &Tensor,
        norm_weight: &Tensor,
        weight0: &Tensor,
        weight1: &Tensor,
        weight2: &Tensor,
        epsilon: f32,
    ) -> Result<(Tensor, Tensor, Tensor), KernelError> {
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
            return Err(KernelError::RmsNormMatmul3Shape);
        }
        let out0 = self.empty(&[1, n0], DType::F16)?;
        let out1 = self.empty(&[1, n1], DType::F16)?;
        let out2 = self.empty(&[1, n2], DType::F16)?;
        let params = NormMultiMatrixParams {
            n0: to_u32(n0)?,
            n1: to_u32(n1)?,
            n2: to_u32(n2)?,
            k: to_u32(width)?,
            epsilon,
        };
        let rows = if tuning(self).is_m4_pro() {
            tuning(self).fused_norm_matvec_rows_for_shape([n0, n1, n2], width)
        } else {
            2
        };
        let simdgroups = if rows == 1 { 8 } else { 4 };
        let groups = n0.max(n1).max(n2).div_ceil(rows * simdgroups);
        dispatch(
            self,
            match rows {
                1 => "matvec3_rms_r1_f16",
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
            size(checked_mul(groups, simdgroups * 32)?, 1, 1),
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
    ) -> Result<(Tensor, Tensor, Tensor), KernelError> {
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
            return Err(KernelError::AddRmsNormMatmul2Shape);
        }
        let residual = self.empty(&[1, width], DType::F16)?;
        let out0 = self.empty(&[1, n0], DType::F16)?;
        let out1 = self.empty(&[1, n1], DType::F16)?;
        let params = NormMultiMatrixParams {
            n0: to_u32(n0)?,
            n1: to_u32(n1)?,
            n2: 0,
            k: to_u32(width)?,
            epsilon,
        };
        let rows = if tuning(self).is_m4_pro() {
            tuning(self).fused_norm_matvec_rows_for_shape([n0, n1, 0], width)
        } else {
            2
        };
        let simdgroups = if rows == 1 { 8 } else { 4 };
        let groups = n0.max(n1).div_ceil(rows * simdgroups);
        dispatch(
            self,
            match rows {
                1 => "matvec2_add_rms_r1_f16",
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
            size(checked_mul(groups, simdgroups * 32)?, 1, 1),
            size(simdgroups * 32, 1, 1),
        )?;
        Ok((residual, out0, out1))
    }
}
