use super::{require_f16, size, to_u32};
use crate::kernels::dispatch;
use crate::{KernelBatch, KernelError, Tensor};

impl KernelBatch<'_> {
    pub fn copy_into_cache(
        &mut self,
        source: &Tensor,
        cache: &Tensor,
        offset: usize,
    ) -> Result<(), KernelError> {
        require_f16(source)?;
        require_f16(cache)?;
        let [source_tokens, source_heads, source_dim] = source.shape() else {
            return Err(KernelError::KvSourceRank);
        };
        let [cache_tokens, cache_heads, cache_dim] = cache.shape() else {
            return Err(KernelError::KvCacheRank);
        };
        if source_heads != cache_heads || source_dim != cache_dim {
            return Err(KernelError::KvShape);
        }
        if offset + *source_tokens > *cache_tokens {
            return Err(KernelError::KvCapacity);
        }
        let stride = source_heads * source_dim;
        let params = [to_u32(*source_tokens)?, to_u32(offset)?, to_u32(stride)?];
        dispatch(
            self,
            "copy_kv_f16",
            &[source, cache],
            &params,
            size(source.len(), 1, 1),
            size(source.len().min(256), 1, 1),
        )?;
        Ok(())
    }
}
