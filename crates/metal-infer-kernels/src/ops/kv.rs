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
            return Err(CoreError::KvSourceRank);
        };
        let [cache_tokens, cache_heads, cache_dim] = cache.shape() else {
            return Err(CoreError::KvCacheRank);
        };
        if source_heads != cache_heads || source_dim != cache_dim {
            return Err(CoreError::KvShape);
        }
        if offset + *source_tokens > *cache_tokens {
            return Err(CoreError::KvCapacity);
        }
        let stride = source_heads * source_dim;
        let params = [to_u32(*source_tokens)?, to_u32(offset)?, to_u32(stride)?];
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
