use metal_infer_runtime::{CoreError, DType, Tensor};

use super::{QkTransformParams, RopeParams, checked_add, checked_mul, require_f16, size, to_u32};
use crate::KernelBatch;

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
    ) -> Result<Tensor, CoreError> {
        require_f16(input)?;
        let shape = input.shape();
        let [tokens, heads, head_dim] = shape else {
            return Err(CoreError::Shape(
                "RoPE expects [tokens, heads, head_dim]".into(),
            ));
        };
        if !head_dim.is_multiple_of(2) {
            return Err(CoreError::Shape("RoPE head_dim must be even".into()));
        }
        let out = self.empty(shape, DType::F16)?;
        let params = RopeParams {
            tokens: to_u32(*tokens, "tokens")?,
            heads: to_u32(*heads, "heads")?,
            head_dim: to_u32(*head_dim, "head_dim")?,
            offset: to_u32(offset, "offset")?,
            theta,
            padding: [0; 3],
        };
        self.dispatch(
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
    ) -> Result<Tensor, CoreError> {
        require_f16(query)?;
        require_f16(key)?;
        require_f16(query_weight)?;
        require_f16(key_weight)?;
        require_f16(key_cache)?;
        let [tokens, query_heads, head_dim] = query.shape() else {
            return Err(CoreError::Shape("query must have rank 3".into()));
        };
        let [key_tokens, kv_heads, key_dim] = key.shape() else {
            return Err(CoreError::Shape("key must have rank 3".into()));
        };
        let [capacity, cache_heads, cache_dim] = key_cache.shape() else {
            return Err(CoreError::Shape("key cache must have rank 3".into()));
        };
        if tokens != key_tokens
            || head_dim != key_dim
            || kv_heads != cache_heads
            || head_dim != cache_dim
            || query_weight.shape() != [*head_dim]
            || key_weight.shape() != [*head_dim]
        {
            return Err(CoreError::Shape(
                "Q/K transform tensor shapes are incompatible".into(),
            ));
        }
        if !head_dim.is_multiple_of(2) || config.offset + *tokens > *capacity {
            return Err(CoreError::Shape(
                "Q/K transform has invalid head_dim or cache offset".into(),
            ));
        }
        let out = self.empty(query.shape(), DType::F16)?;
        let params = QkTransformParams {
            tokens: to_u32(*tokens, "tokens")?,
            query_heads: to_u32(*query_heads, "query heads")?,
            kv_heads: to_u32(*kv_heads, "KV heads")?,
            head_dim: to_u32(*head_dim, "head dim")?,
            offset: to_u32(config.offset, "offset")?,
            theta: config.theta,
            cache_capacity: to_u32(*capacity, "cache capacity")?,
            epsilon: config.epsilon,
        };
        let heads = checked_add(*query_heads, *kv_heads, "Q/K heads")?;
        let groups = checked_mul(*tokens, heads, "Q/K groups")?;
        self.dispatch(
            "qk_norm_rope_cache_f16",
            &[query, key, query_weight, key_weight, &out, key_cache],
            &params,
            size(checked_mul(groups, 32, "Q/K grid")?, 1, 1),
            size(32, 1, 1),
        )?;
        Ok(out)
    }
}
