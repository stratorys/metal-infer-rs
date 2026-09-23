use super::{require_f16, size, to_u32};
use crate::{DType, KernelBatch, KernelError, Tensor};

impl KernelBatch<'_> {
    pub fn argmax(
        &mut self,
        logits: &Tensor,
        output: &Tensor,
    ) -> Result<(), KernelError> {
        require_f16(logits)?;
        if logits.shape().len() != 1 || logits.is_empty() {
            return Err(KernelError::ArgmaxLogitsShape);
        }
        if output.dtype() != DType::U32 || output.shape() != [1] {
            return Err(KernelError::ArgmaxOutputShape);
        }
        let count = to_u32(logits.len())?;
        let groups = count.div_ceil(2048);
        let partial_values = self.empty(&[groups as usize], DType::F32)?;
        let partial_indices = self.empty(&[groups as usize], DType::U32)?;
        self.dispatch(
            "argmax_f16_partial",
            &[logits, &partial_values, &partial_indices],
            &count,
            size(groups as usize * 256, 1, 1),
            size(256, 1, 1),
        )?;
        self.dispatch(
            "argmax_f16_reduce",
            &[&partial_values, &partial_indices, output],
            &groups,
            size(256, 1, 1),
            size(256, 1, 1),
        )?;
        Ok(())
    }
}
