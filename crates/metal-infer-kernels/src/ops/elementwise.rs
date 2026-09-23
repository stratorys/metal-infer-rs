use metal_infer_runtime::{CoreError, DType, Tensor};

use super::{matrix_shape, require_f16, require_same_shape, size, to_u32};
use crate::KernelBatch;

impl KernelBatch<'_> {
    pub fn copy_row(
        &mut self,
        source: &Tensor,
        destination: &Tensor,
        row: usize,
    ) -> Result<(), CoreError> {
        require_f16(source)?;
        require_f16(destination)?;
        let [source_rows, width] = source.shape() else {
            return Err(CoreError::CopyRowSourceRank);
        };
        let [rows, destination_width] = destination.shape() else {
            return Err(CoreError::CopyRowDestinationRank);
        };
        if *source_rows != 1 || width != destination_width || row >= *rows {
            return Err(CoreError::CopyRowShape);
        }
        let params = [to_u32(*width)?, to_u32(row)?, 0];
        self.dispatch(
            "copy_row_f16",
            &[source, destination],
            &params,
            size(*width, 1, 1),
            size((*width).min(256), 1, 1),
        )
    }
    pub fn add(
        &mut self,
        left: &Tensor,
        right: &Tensor,
    ) -> Result<Tensor, CoreError> {
        require_f16(left)?;
        require_f16(right)?;
        require_same_shape(left, right)?;
        let out = self.empty(left.shape(), DType::F16)?;
        let count = to_u32(left.len())?;
        self.dispatch(
            "add_f16",
            &[left, right, &out],
            &count,
            size(left.len(), 1, 1),
            size(left.len().min(256), 1, 1),
        )?;
        Ok(out)
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
        let count = to_u32(gate.len())?;
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
            return Err(CoreError::TokensShape);
        }
        require_f16(table)?;
        let [_, width] = matrix_shape(table)?;
        let out = self.empty(&[tokens.len(), width], DType::F16)?;
        let params = [to_u32(tokens.len())?, to_u32(width)?];
        self.dispatch(
            "embedding_f16",
            &[tokens, table, &out],
            &params,
            size(width, tokens.len(), 1),
            size(width.min(256), 1, 1),
        )?;
        Ok(out)
    }
}
