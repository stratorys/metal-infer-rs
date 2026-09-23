use metal_infer_runtime::{CoreError, DType, Tensor};

use super::{AttentionParams, checked_mul, require_f16, size, to_u32};
use crate::KernelBatch;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttentionKind {
    Reference,
    Tiled,
    FlashPrefill,
    DecodeSplitKv,
    FlashDecode,
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionConfig {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub query_offset: usize,
}

impl KernelBatch<'_> {
    pub fn attention(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        config: AttentionConfig,
        kind: AttentionKind,
    ) -> Result<Tensor, CoreError> {
        self.attention_with_flash_block(query, key, value, config, kind, None)
    }

    pub fn attention_flash_decode_with_block(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        config: AttentionConfig,
        block_keys: usize,
    ) -> Result<Tensor, CoreError> {
        self.attention_with_flash_block(
            query,
            key,
            value,
            config,
            AttentionKind::FlashDecode,
            Some(block_keys),
        )
    }

    fn attention_with_flash_block(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        config: AttentionConfig,
        kind: AttentionKind,
        flash_block: Option<usize>,
    ) -> Result<Tensor, CoreError> {
        let kind = if kind == AttentionKind::FlashDecode && config.head_dim != 128 {
            AttentionKind::DecodeSplitKv
        } else {
            kind
        };
        require_f16(query)?;
        require_f16(key)?;
        require_f16(value)?;
        let [tokens, query_heads, query_dim] = query.shape() else {
            return Err(CoreError::Shape("query must have rank 3".into()));
        };
        let [kv_length, key_heads, key_dim] = key.shape() else {
            return Err(CoreError::Shape("key must have rank 3".into()));
        };
        let [value_length, value_heads, value_dim] = value.shape() else {
            return Err(CoreError::Shape("value must have rank 3".into()));
        };
        if *query_heads != config.query_heads
            || *query_dim != config.head_dim
            || *key_heads != config.kv_heads
            || *key_dim != config.head_dim
            || value_length != kv_length
            || value_heads != key_heads
            || value_dim != key_dim
        {
            return Err(CoreError::Shape(
                "attention tensor shapes do not match config".into(),
            ));
        }
        if config.kv_heads == 0 || !config.query_heads.is_multiple_of(config.kv_heads) {
            return Err(CoreError::Shape(
                "query_heads must be divisible by kv_heads".into(),
            ));
        }
        if kind != AttentionKind::Reference
            && (!config.head_dim.is_power_of_two() || config.head_dim > 256)
        {
            return Err(CoreError::Shape(
                "tiled attention requires a power-of-two head_dim <= 256".into(),
            ));
        }
        let flash_block_keys = if kind == AttentionKind::FlashDecode {
            let block = flash_block
                .unwrap_or_else(|| self.kernels.flash_decode_block_for_length(*kv_length));
            if !matches!(block, 32 | 64 | 128 | 256) {
                return Err(CoreError::Shape(
                    "flash decode block size must be 32, 64, 128, or 256".into(),
                ));
            }
            block
        } else {
            0
        };
        let out = self.empty(query.shape(), DType::F16)?;
        let params = AttentionParams {
            tokens: to_u32(*tokens, "tokens")?,
            q_heads: to_u32(config.query_heads, "query_heads")?,
            kv_heads: to_u32(config.kv_heads, "kv_heads")?,
            head_dim: to_u32(config.head_dim, "head_dim")?,
            causal: u32::from(config.causal),
            query_offset: to_u32(config.query_offset, "query_offset")?,
            kv_length: to_u32(*kv_length, "kv_length")?,
            padding: to_u32(flash_block_keys, "flash decode block size")?,
        };
        if matches!(
            kind,
            AttentionKind::DecodeSplitKv | AttentionKind::FlashDecode
        ) && *tokens != 1
        {
            return Err(CoreError::Shape(
                "decode attention requires exactly one query token".into(),
            ));
        }
        if kind == AttentionKind::FlashDecode {
            if config.query_heads / config.kv_heads != 2 {
                return Err(CoreError::Shape(
                    "flash decode requires two query heads per KV head".into(),
                ));
            }
            let available = if config.causal {
                (*kv_length).min(config.query_offset.saturating_add(1))
            } else {
                *kv_length
            };
            if available == 0 {
                return Err(CoreError::Shape(
                    "flash decode requires at least one available key".into(),
                ));
            }
            let blocks = available.div_ceil(flash_block_keys);
            let partial_width = config
                .head_dim
                .checked_add(2)
                .ok_or_else(|| CoreError::Shape("flash decode partial width overflow".into()))?;
            let scratch = self.empty(&[config.kv_heads, blocks, 2, partial_width], DType::F32)?;
            let groups = checked_mul(config.kv_heads, blocks, "flash decode groups")?;
            self.dispatch(
                "attention_flash_decode_partial_f16",
                &[query, key, value, &scratch],
                &params,
                size(checked_mul(groups, 128, "flash decode grid")?, 1, 1),
                size(128, 1, 1),
            )?;
            self.dispatch(
                "attention_flash_decode_reduce_f16",
                &[&scratch, &out],
                &params,
                size(
                    checked_mul(config.query_heads, 128, "flash decode reduction grid")?,
                    1,
                    1,
                ),
                size(128, 1, 1),
            )?;
            return Ok(out);
        }
        let kernel = match kind {
            AttentionKind::Reference => "attention_reference_f16",
            AttentionKind::Tiled => "attention_tiled_f16",
            AttentionKind::FlashPrefill => "attention_flash_prefill_f16",
            AttentionKind::DecodeSplitKv => "attention_decode_f16",
            AttentionKind::FlashDecode => unreachable!("handled above"),
        };
        let (grid, threadgroup) = match kind {
            AttentionKind::Reference => (
                size(config.head_dim, config.query_heads, *tokens),
                size(config.head_dim.min(256), 1, 1),
            ),
            AttentionKind::DecodeSplitKv => {
                let groups = checked_mul(config.query_heads, *tokens, "attention groups")?;
                (
                    size(checked_mul(groups, 256, "decode attention grid")?, 1, 1),
                    size(256, 1, 1),
                )
            }
            AttentionKind::Tiled => {
                let groups = checked_mul(config.query_heads, *tokens, "attention groups")?;
                (
                    size(checked_mul(groups, 32, "attention grid")?, 1, 1),
                    size(32, 1, 1),
                )
            }
            AttentionKind::FlashPrefill => (
                size(
                    checked_mul(tokens.div_ceil(32), 128, "flash prefill grid")?,
                    config.query_heads,
                    1,
                ),
                size(128, 1, 1),
            ),
            AttentionKind::FlashDecode => unreachable!("handled above"),
        };
        self.dispatch(
            kernel,
            &[query, key, value, &out],
            &params,
            grid,
            threadgroup,
        )?;
        Ok(out)
    }
}
