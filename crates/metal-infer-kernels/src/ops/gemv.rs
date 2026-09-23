use super::{MultiMatrixParams, checked_mul, matrix_shape, require_f16, size, to_u32};
use crate::kernels::{dispatch, tuning};
use crate::{DType, KernelBatch, KernelError, Tensor};

impl KernelBatch<'_> {
    pub fn matmul2(
        &mut self,
        input: &Tensor,
        weight0: &Tensor,
        weight1: &Tensor,
    ) -> Result<(Tensor, Tensor), KernelError> {
        let [m, k] = matrix_shape(input)?;
        let [n0, k0] = matrix_shape(weight0)?;
        let [n1, k1] = matrix_shape(weight1)?;
        require_f16(input)?;
        require_f16(weight0)?;
        require_f16(weight1)?;
        if k != k0 || k != k1 {
            return Err(KernelError::Matmul2InnerDimension);
        }
        if m != 1 {
            return Ok((self.matmul(input, weight0)?, self.matmul(input, weight1)?));
        }
        let out0 = self.empty(&[1, n0], DType::F16)?;
        let out1 = self.empty(&[1, n1], DType::F16)?;
        let params = MultiMatrixParams {
            n0: to_u32(n0)?,
            n1: to_u32(n1)?,
            n2: 0,
            k: to_u32(k)?,
        };
        let outputs = n0.max(n1);
        let rows = if k.is_multiple_of(256) && tuning(self).is_m4_pro() {
            2
        } else {
            0
        };
        let tuned = rows != 0;
        let rows_per_group = if tuned { rows * 4 } else { 32 };
        let threads = if tuned { 128 } else { 256 };
        let groups = outputs.div_ceil(rows_per_group);
        dispatch(
            self,
            match rows {
                2 => "matvec2_tuned_f16",
                _ => "matvec2_f16",
            },
            &[input, weight0, weight1, &out0, &out1],
            &params,
            size(checked_mul(groups, threads)?, 1, 1),
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
    ) -> Result<(Tensor, Tensor, Tensor), KernelError> {
        let [m, k] = matrix_shape(input)?;
        let [n0, k0] = matrix_shape(weight0)?;
        let [n1, k1] = matrix_shape(weight1)?;
        let [n2, k2] = matrix_shape(weight2)?;
        require_f16(input)?;
        require_f16(weight0)?;
        require_f16(weight1)?;
        require_f16(weight2)?;
        if k != k0 || k != k1 || k != k2 {
            return Err(KernelError::Matmul3InnerDimension);
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
            n0: to_u32(n0)?,
            n1: to_u32(n1)?,
            n2: to_u32(n2)?,
            k: to_u32(k)?,
        };
        let outputs = n0.max(n1).max(n2);
        let rows = if k.is_multiple_of(256) && tuning(self).is_m4_pro() {
            2
        } else {
            0
        };
        let tuned = rows != 0;
        let rows_per_group = if tuned { rows * 4 } else { 32 };
        let threads = if tuned { 128 } else { 256 };
        let groups = outputs.div_ceil(rows_per_group);
        dispatch(
            self,
            match rows {
                2 => "matvec3_tuned_f16",
                _ => "matvec3_f16",
            },
            &[input, weight0, weight1, weight2, &out0, &out1, &out2],
            &params,
            size(checked_mul(groups, threads)?, 1, 1),
            size(threads, 1, 1),
        )?;
        Ok((out0, out1, out2))
    }
}
