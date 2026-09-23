use super::{QkTransformParams, RopeParams, checked_add, checked_mul, require_f16, size, to_u32};
use crate::kernels::dispatch;
use crate::{DType, KernelBatch, KernelError, Tensor};

#[derive(Clone, Copy, Debug)]
pub struct QkNormRopeCacheConfig {
    pub offset: usize,
    pub theta: f32,
    pub epsilon: f32,
}

impl KernelBatch<'_> {
    pub fn rope(
        &mut self,
        input: &Tensor,
        offset: usize,
        theta: f32,
    ) -> Result<Tensor, KernelError> {
        require_f16(input)?;
        let shape = input.shape();
        let [tokens, heads, head_dim] = shape else {
            return Err(KernelError::RopeRank);
        };
        if !head_dim.is_multiple_of(2) {
            return Err(KernelError::RopeOddHeadDim);
        }
        let out = self.empty(shape, DType::F16)?;
        let params = RopeParams {
            tokens: to_u32(*tokens)?,
            heads: to_u32(*heads)?,
            head_dim: to_u32(*head_dim)?,
            offset: to_u32(offset)?,
            theta,
            padding: [0; 3],
        };
        dispatch(
            self,
            "rope_f16",
            &[input, &out],
            &params,
            size(*head_dim / 2, *heads, *tokens),
            size((*head_dim / 2).min(32), 1, 1),
        )?;
        Ok(out)
    }

    pub fn qk_norm_rope_cache(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        query_weight: &Tensor,
        key_weight: &Tensor,
        key_cache: &Tensor,
        config: QkNormRopeCacheConfig,
    ) -> Result<Tensor, KernelError> {
        require_f16(query)?;
        require_f16(key)?;
        require_f16(query_weight)?;
        require_f16(key_weight)?;
        require_f16(key_cache)?;
        let [tokens, query_heads, head_dim] = query.shape() else {
            return Err(KernelError::QueryRank);
        };
        let [key_tokens, kv_heads, key_dim] = key.shape() else {
            return Err(KernelError::KeyRank);
        };
        let [capacity, cache_heads, cache_dim] = key_cache.shape() else {
            return Err(KernelError::KeyCacheRank);
        };
        if tokens != key_tokens
            || head_dim != key_dim
            || kv_heads != cache_heads
            || head_dim != cache_dim
            || query_weight.shape() != [*head_dim]
            || key_weight.shape() != [*head_dim]
        {
            return Err(KernelError::QkTransformShape);
        }
        if !head_dim.is_multiple_of(2) || config.offset + *tokens > *capacity {
            return Err(KernelError::QkTransformOffset);
        }
        let out = self.empty(query.shape(), DType::F16)?;
        let params = QkTransformParams {
            tokens: to_u32(*tokens)?,
            query_heads: to_u32(*query_heads)?,
            kv_heads: to_u32(*kv_heads)?,
            head_dim: to_u32(*head_dim)?,
            offset: to_u32(config.offset)?,
            theta: config.theta,
            cache_capacity: to_u32(*capacity)?,
            epsilon: config.epsilon,
        };
        let heads = checked_add(*query_heads, *kv_heads)?;
        let groups = checked_mul(*tokens, heads)?;
        dispatch(
            self,
            "qk_norm_rope_cache_f16",
            &[query, key, query_weight, key_weight, &out, key_cache],
            &params,
            size(checked_mul(groups, 32)?, 1, 1),
            size(32, 1, 1),
        )?;
        Ok(out)
    }
}
