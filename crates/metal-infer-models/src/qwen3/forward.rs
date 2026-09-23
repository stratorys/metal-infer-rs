use metal_infer_kernels::{
    AttentionConfig, DType, DispatchStats, KernelBatch, PendingBatch, QkNormRopeCacheConfig, Tensor,
};

use crate::kv_cache::LayerCache;
use crate::qwen3::Qwen3Model;
use crate::qwen3::layers::LayerWeights;
use crate::{KvCache, ModelError};

impl Qwen3Model {
    pub fn decode_batch(
        &self,
        tokens: &[u32],
        caches: &mut [&mut KvCache],
    ) -> Result<Tensor, ModelError> {
        if tokens.is_empty() || tokens.len() != caches.len() {
            return Err(ModelError::DecodeBatchMismatch);
        }
        for cache in caches.iter() {
            if cache.filled == 0 || cache.filled >= cache.capacity {
                return Err(ModelError::CacheCapacity);
            }
        }
        let input = self.context.tensor_u32(tokens, &[tokens.len()])?;
        let mut batch = self.kernels.begin_batch()?;
        let mut hidden = batch.embedding(&input, &self.embedding)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let normalized =
                batch.rms_norm(&hidden, &layer.input_norm, self.config.rms_norm_eps)?;
            let (query, key, value) = if self.plan.fusions.qkv {
                batch.matmul3(
                    &normalized,
                    &layer.attention.query,
                    &layer.attention.key,
                    &layer.attention.value,
                )?
            } else {
                (
                    batch.matmul(&normalized, &layer.attention.query)?,
                    batch.matmul(&normalized, &layer.attention.key)?,
                    batch.matmul(&normalized, &layer.attention.value)?,
                )
            };
            let attention_rows =
                batch.empty(&[tokens.len(), self.config.query_width()], DType::F16)?;
            for (row, cache) in caches.iter().enumerate() {
                let layer_cache = cache
                    .layers
                    .get(layer_index)
                    .ok_or(ModelError::MissingCacheLayer)?;
                let offset = cache.filled;
                let query_row = query.row(row)?.reshape(&[
                    1,
                    self.config.num_attention_heads,
                    self.config.head_dim,
                ])?;
                let key_row = key.row(row)?.reshape(&[
                    1,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                ])?;
                let value_row = value.row(row)?.reshape(&[
                    1,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                ])?;
                let query_row = if self.plan.fusions.qk_rope_cache {
                    batch.qk_norm_rope_cache(
                        &query_row,
                        &key_row,
                        &layer.attention.query_norm,
                        &layer.attention.key_norm,
                        &layer_cache.key,
                        QkNormRopeCacheConfig {
                            offset,
                            theta: self.config.rope_theta,
                            epsilon: self.config.rms_norm_eps,
                        },
                    )?
                } else {
                    let q = batch.rms_norm(
                        &query_row,
                        &layer.attention.query_norm,
                        self.config.rms_norm_eps,
                    )?;
                    let q = batch.rope(&q, offset, self.config.rope_theta)?;
                    let k = batch.rms_norm(
                        &key_row,
                        &layer.attention.key_norm,
                        self.config.rms_norm_eps,
                    )?;
                    let k = batch.rope(&k, offset, self.config.rope_theta)?;
                    batch.copy_into_cache(&k, &layer_cache.key, offset)?;
                    q
                };
                batch.copy_into_cache(&value_row, &layer_cache.value, offset)?;
                let length = offset + 1;
                let kind = self.plan.attention_for_tokens(
                    1,
                    length,
                    self.config.num_attention_heads / self.config.num_key_value_heads,
                );
                let attention = batch.attention(
                    &query_row,
                    &layer_cache.key.prefix(length)?,
                    &layer_cache.value.prefix(length)?,
                    AttentionConfig {
                        query_heads: self.config.num_attention_heads,
                        kv_heads: self.config.num_key_value_heads,
                        head_dim: self.config.head_dim,
                        causal: true,
                        query_offset: offset,
                    },
                    kind,
                )?;
                batch.copy_row(
                    &attention.reshape(&[1, self.config.query_width()])?,
                    &attention_rows,
                    row,
                )?;
            }
            let attention = batch.matmul(&attention_rows, &layer.attention.output)?;
            let (residual, normalized) = if self.plan.fusions.add_rms_norm {
                batch.add_rms_norm(
                    &hidden,
                    &attention,
                    &layer.post_attention_norm,
                    self.config.rms_norm_eps,
                )?
            } else {
                let residual = batch.add(&hidden, &attention)?;
                let normalized = batch.rms_norm(
                    &residual,
                    &layer.post_attention_norm,
                    self.config.rms_norm_eps,
                )?;
                (residual, normalized)
            };
            let (gate, up) = if self.plan.fusions.gate_up {
                batch.matmul2(&normalized, &layer.mlp.gate, &layer.mlp.up)?
            } else {
                (
                    batch.matmul(&normalized, &layer.mlp.gate)?,
                    batch.matmul(&normalized, &layer.mlp.up)?,
                )
            };
            let activated = batch.swiglu(&gate, &up)?;
            let down = batch.matmul(&activated, &layer.mlp.down)?;
            hidden = batch.add(&residual, &down)?;
        }
        let normalized = batch.rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps)?;
        let logits = batch.matmul(&normalized, &self.lm_head)?;
        batch.finish()?;
        for cache in caches.iter_mut() {
            cache.filled += 1;
        }
        Ok(logits)
    }

    pub(super) fn forward_with_stats(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
    ) -> Result<(Tensor, DispatchStats), ModelError> {
        let token_tensor = self.context.tensor_u32(tokens, &[tokens.len()])?;
        let previous = cache.filled;
        let (logits, pending) = self.forward_async(&token_tensor, cache, None)?;
        match pending.wait() {
            Ok(stats) => Ok((logits, stats)),
            Err(error) => {
                cache.filled = previous;
                Err(error.into())
            }
        }
    }

    pub(super) fn forward_async<'model>(
        &'model self,
        tokens: &Tensor,
        cache: &mut KvCache,
        argmax_output: Option<&Tensor>,
    ) -> Result<(Tensor, PendingBatch<'model>), ModelError> {
        if tokens.dtype() != DType::U32 || tokens.shape().len() != 1 || tokens.is_empty() {
            return Err(ModelError::TokensShape);
        }
        if cache.layers.len() != self.layers.len() {
            return Err(ModelError::CacheLayerCount);
        }
        let requested = cache.filled + tokens.len();
        if requested > cache.capacity {
            return Err(ModelError::CacheCapacity);
        }
        let mut batch = self.kernels.begin_batch()?;
        let mut hidden = batch.embedding(tokens, &self.embedding)?;
        let offset = cache.filled;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache
                .layers
                .get(layer_index)
                .ok_or(ModelError::MissingCacheLayer)?;
            hidden =
                self.forward_layer(&mut batch, hidden, layer, layer_cache, offset, requested)?;
        }
        let normalized = batch.rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps)?;
        let last = normalized.row(tokens.len() - 1)?;
        let logits = batch.matmul(&last, &self.lm_head)?;
        if let Some(output) = argmax_output {
            batch.argmax(&logits.reshape(&[logits.len()])?, output)?;
        }
        let pending = batch.commit()?;
        cache.filled = requested;
        Ok((logits, pending))
    }

    pub(super) fn forward_layer(
        &self,
        batch: &mut KernelBatch<'_>,
        hidden: Tensor,
        layer: &LayerWeights,
        cache: &LayerCache,
        offset: usize,
        active_length: usize,
    ) -> Result<Tensor, ModelError> {
        let tokens = hidden
            .shape()
            .first()
            .copied()
            .ok_or(ModelError::HiddenStateRank)?;
        let fused_qkv = tokens == 1 && self.plan.fusions.decode_norm && self.plan.fusions.qkv;
        let (query, key, value) = if fused_qkv {
            batch.rms_norm_matmul3(
                &hidden,
                &layer.input_norm,
                &layer.attention.query,
                &layer.attention.key,
                &layer.attention.value,
                self.config.rms_norm_eps,
            )?
        } else {
            let normalized =
                batch.rms_norm(&hidden, &layer.input_norm, self.config.rms_norm_eps)?;
            if self.plan.fusions.qkv {
                batch.matmul3(
                    &normalized,
                    &layer.attention.query,
                    &layer.attention.key,
                    &layer.attention.value,
                )?
            } else {
                (
                    batch.matmul(&normalized, &layer.attention.query)?,
                    batch.matmul(&normalized, &layer.attention.key)?,
                    batch.matmul(&normalized, &layer.attention.value)?,
                )
            }
        };
        let query = query.reshape(&[
            tokens,
            self.config.num_attention_heads,
            self.config.head_dim,
        ])?;
        let key = key.reshape(&[
            tokens,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ])?;
        let value = value.reshape(&[
            tokens,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ])?;
        let query = if self.plan.fusions.qk_rope_cache {
            batch.qk_norm_rope_cache(
                &query,
                &key,
                &layer.attention.query_norm,
                &layer.attention.key_norm,
                &cache.key,
                QkNormRopeCacheConfig {
                    offset,
                    theta: self.config.rope_theta,
                    epsilon: self.config.rms_norm_eps,
                },
            )?
        } else {
            let query = batch.rms_norm(
                &query,
                &layer.attention.query_norm,
                self.config.rms_norm_eps,
            )?;
            let query = batch.rope(&query, offset, self.config.rope_theta)?;
            let key = batch.rms_norm(&key, &layer.attention.key_norm, self.config.rms_norm_eps)?;
            let key = batch.rope(&key, offset, self.config.rope_theta)?;
            batch.copy_into_cache(&key, &cache.key, offset)?;
            query
        };
        batch.copy_into_cache(&value, &cache.value, offset)?;
        let active_key = cache.key.prefix(active_length)?;
        let active_value = cache.value.prefix(active_length)?;
        let attention_kind = self.plan.attention_for_tokens(
            tokens,
            active_length,
            self.config.num_attention_heads / self.config.num_key_value_heads,
        );
        let attention = batch.attention(
            &query,
            &active_key,
            &active_value,
            AttentionConfig {
                query_heads: self.config.num_attention_heads,
                kv_heads: self.config.num_key_value_heads,
                head_dim: self.config.head_dim,
                causal: true,
                query_offset: offset,
            },
            attention_kind,
        )?;
        let attention = attention.reshape(&[tokens, self.config.query_width()])?;
        let attention = batch.matmul(&attention, &layer.attention.output)?;
        let fused_gate_up = tokens == 1
            && self.plan.fusions.decode_norm
            && self.plan.fusions.add_rms_norm
            && self.plan.fusions.gate_up;
        let (residual, gate, up) = if fused_gate_up {
            batch.add_rms_norm_matmul2(
                &hidden,
                &attention,
                &layer.post_attention_norm,
                &layer.mlp.gate,
                &layer.mlp.up,
                self.config.rms_norm_eps,
            )?
        } else {
            let (residual, normalized) = if self.plan.fusions.add_rms_norm {
                batch.add_rms_norm(
                    &hidden,
                    &attention,
                    &layer.post_attention_norm,
                    self.config.rms_norm_eps,
                )?
            } else {
                let residual = batch.add(&hidden, &attention)?;
                let normalized = batch.rms_norm(
                    &residual,
                    &layer.post_attention_norm,
                    self.config.rms_norm_eps,
                )?;
                (residual, normalized)
            };
            let (gate, up) = if self.plan.fusions.gate_up {
                batch.matmul2(&normalized, &layer.mlp.gate, &layer.mlp.up)?
            } else {
                (
                    batch.matmul(&normalized, &layer.mlp.gate)?,
                    batch.matmul(&normalized, &layer.mlp.up)?,
                )
            };
            (residual, gate, up)
        };
        let activated = batch.swiglu(&gate, &up)?;
        let down = batch.matmul(&activated, &layer.mlp.down)?;
        batch.add(&residual, &down).map_err(Into::into)
    }
}
