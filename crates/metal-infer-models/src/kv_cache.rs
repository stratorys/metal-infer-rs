use metal_infer_kernels::{DType, MetalContext, Tensor};

use crate::{ModelError, Qwen3Config};

#[derive(Clone)]
pub(crate) struct LayerCache {
    pub(crate) key: Tensor,
    pub(crate) value: Tensor,
}

#[derive(Clone)]
pub struct KvCache {
    pub(crate) layers: Vec<LayerCache>,
    pub(crate) capacity: usize,
    pub(crate) filled: usize,
}

impl KvCache {
    pub fn new(
        context: &MetalContext,
        config: &Qwen3Config,
        capacity: usize,
    ) -> Result<Self, ModelError> {
        if capacity == 0 {
            return Err(ModelError::EmptyKvCache);
        }
        let shape = [capacity, config.num_key_value_heads, config.head_dim];
        let layers = (0..config.num_hidden_layers)
            .map(|_| {
                Ok(LayerCache {
                    key: context.empty(&shape, DType::F16)?,
                    value: context.empty(&shape, DType::F16)?,
                })
            })
            .collect::<Result<Vec<_>, ModelError>>()?;
        Ok(Self {
            layers,
            capacity,
            filled: 0,
        })
    }

    pub const fn len(&self) -> usize {
        self.filled
    }

    pub const fn is_empty(&self) -> bool {
        self.filled == 0
    }

    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn reset(&mut self) {
        self.filled = 0;
    }
}
