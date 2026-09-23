use metal_infer_kernels::{DType, MetalContext, Tensor};

use crate::{ModelError, Qwen3Config};

#[derive(Clone)]
pub struct LayerCache {
    pub key: Tensor,
    pub value: Tensor,
}

#[derive(Clone)]
pub struct KvCache {
    layers: Vec<LayerCache>,
    capacity: usize,
    filled: usize,
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

    pub fn layer(
        &self,
        index: usize,
    ) -> Result<&LayerCache, ModelError> {
        self.layers.get(index).ok_or(ModelError::MissingCacheLayer)
    }

    pub const fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn reserve(
        &self,
        tokens: usize,
    ) -> Result<usize, ModelError> {
        self.filled
            .checked_add(tokens)
            .filter(|length| *length <= self.capacity)
            .ok_or(ModelError::CacheCapacity)
    }

    pub fn advance(
        &mut self,
        tokens: usize,
    ) -> Result<(), ModelError> {
        self.filled = self.reserve(tokens)?;
        Ok(())
    }

    pub fn truncate(
        &mut self,
        length: usize,
    ) {
        self.filled = self.filled.min(length);
    }
}
