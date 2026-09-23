use metal_infer_runtime::{CoreError, Tensor};

use super::{require_f16, size, to_u32};
use crate::KernelBatch;

impl KernelBatch<'_> {
    pub fn copy_into_cache(
        &mut self,
        source: &Tensor,
        cache: &Tensor,
        offset: usize,
    ) -> Result<(), CoreError> {
        require_f16(source)?;
        require_f16(cache)?;
        let [source_tokens, source_heads, source_dim] = source.shape() else {
            return Err(CoreError::Shape("KV source must have rank 3".into()));
        };
        let [cache_tokens, cache_heads, cache_dim] = cache.shape() else {
            return Err(CoreError::Shape("KV cache must have rank 3".into()));
        };
        if source_heads != cache_heads || source_dim != cache_dim {
            return Err(CoreError::Shape(
                "KV source/cache shapes are incompatible".into(),
            ));
        }
        if offset + *source_tokens > *cache_tokens {
            return Err(CoreError::Shape("KV cache capacity exceeded".into()));
        }
        let stride = source_heads * source_dim;
        let params = [
            to_u32(*source_tokens, "tokens")?,
            to_u32(offset, "offset")?,
            to_u32(stride, "stride")?,
        ];
        self.dispatch(
            "copy_kv_f16",
            &[source, cache],
            &params,
            size(source.len(), 1, 1),
            size(source.len().min(256), 1, 1),
        )?;
        Ok(())
    }
}
