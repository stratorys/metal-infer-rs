use metal_infer_runtime::{CoreError, DType, Tensor};

use super::{NormParams, checked_mul, require_f16, require_same_shape, size, to_u32};
use crate::KernelBatch;

impl KernelBatch<'_> {
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
}
