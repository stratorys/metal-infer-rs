use metal_infer_kernels::{
    AttentionConfig, DType, DispatchStats, KernelBatch, PendingBatch, QkNormRopeCacheConfig, Tensor,
};

use crate::kv_cache::LayerCache;
use crate::qwen3::Qwen3Model;
use crate::qwen3::layers::LayerWeights;
use crate::{KvCache, ModelError};

pub fn decode_batch(
    model: &Qwen3Model,
    tokens: &[u32],
    caches: &mut [&mut KvCache],
) -> Result<Tensor, ModelError> {
    if tokens.is_empty() || tokens.len() != caches.len() {
        return Err(ModelError::DecodeBatchMismatch);
    }
    for cache in caches.iter() {
        if cache.is_empty() {
            return Err(ModelError::CacheCapacity);
        }
        cache.reserve(1)?;
    }
    let input = model.context.tensor_u32(tokens, &[tokens.len()])?;
    let mut batch = model.kernels.begin_batch()?;
    let mut hidden = batch.embedding(&input, &model.embedding)?;
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let normalized = batch.rms_norm(&hidden, &layer.input_norm, model.config.rms_norm_eps)?;
        let (query, key, value) = if model.plan.fusions.qkv {
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
            batch.empty(&[tokens.len(), model.config.query_width()], DType::F16)?;
        for (row, cache) in caches.iter().enumerate() {
            let layer_cache = cache.layer(layer_index)?;
            let offset = cache.len();
            let query_row = query.row(row)?.reshape(&[
                1,
                model.config.num_attention_heads,
                model.config.head_dim,
            ])?;
            let key_row = key.row(row)?.reshape(&[
                1,
                model.config.num_key_value_heads,
                model.config.head_dim,
            ])?;
            let value_row = value.row(row)?.reshape(&[
                1,
                model.config.num_key_value_heads,
                model.config.head_dim,
            ])?;
            let query_row = if model.plan.fusions.qk_rope_cache {
                batch.qk_norm_rope_cache(
                    &query_row,
                    &key_row,
                    &layer.attention.query_norm,
                    &layer.attention.key_norm,
                    &layer_cache.key,
                    QkNormRopeCacheConfig {
                        offset,
                        theta: model.config.rope_theta,
                        epsilon: model.config.rms_norm_eps,
                    },
                )?
            } else {
                let q = batch.rms_norm(
                    &query_row,
                    &layer.attention.query_norm,
                    model.config.rms_norm_eps,
                )?;
                let q = batch.rope(&q, offset, model.config.rope_theta)?;
                let k = batch.rms_norm(
                    &key_row,
                    &layer.attention.key_norm,
                    model.config.rms_norm_eps,
                )?;
                let k = batch.rope(&k, offset, model.config.rope_theta)?;
                batch.copy_into_cache(&k, &layer_cache.key, offset)?;
                q
            };
            batch.copy_into_cache(&value_row, &layer_cache.value, offset)?;
            let length = offset + 1;
            let kind = model.plan.attention_for_tokens(
                1,
                length,
                model.config.num_attention_heads / model.config.num_key_value_heads,
            );
            let attention = batch.attention(
                &query_row,
                &layer_cache.key.prefix(length)?,
                &layer_cache.value.prefix(length)?,
                AttentionConfig {
                    query_heads: model.config.num_attention_heads,
                    kv_heads: model.config.num_key_value_heads,
                    head_dim: model.config.head_dim,
                    causal: true,
                    query_offset: offset,
                },
                kind,
            )?;
            batch.copy_row(
                &attention.reshape(&[1, model.config.query_width()])?,
                &attention_rows,
                row,
            )?;
        }
        let attention = batch.matmul(&attention_rows, &layer.attention.output)?;
        let (residual, normalized) = if model.plan.fusions.add_rms_norm {
            batch.add_rms_norm(
                &hidden,
                &attention,
                &layer.post_attention_norm,
                model.config.rms_norm_eps,
            )?
        } else {
            let residual = batch.add(&hidden, &attention)?;
            let normalized = batch.rms_norm(
                &residual,
                &layer.post_attention_norm,
                model.config.rms_norm_eps,
            )?;
            (residual, normalized)
        };
        let (gate, up) = if model.plan.fusions.gate_up {
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
    let normalized = batch.rms_norm(&hidden, &model.final_norm, model.config.rms_norm_eps)?;
    let logits = batch.matmul(&normalized, &model.lm_head)?;
    batch.finish()?;
    for cache in caches.iter_mut() {
        cache.advance(1)?;
    }
    Ok(logits)
}

pub fn forward_with_stats(
    model: &Qwen3Model,
    tokens: &[u32],
    cache: &mut KvCache,
) -> Result<(Tensor, DispatchStats), ModelError> {
    let token_tensor = model.context.tensor_u32(tokens, &[tokens.len()])?;
    let previous = cache.len();
    let (logits, pending) = forward_async(model, &token_tensor, cache, None)?;
    match pending.wait() {
        Ok(stats) => Ok((logits, stats)),
        Err(error) => {
            cache.truncate(previous);
            Err(error.into())
        }
    }
}

pub fn forward_async<'model>(
    model: &'model Qwen3Model,
    tokens: &Tensor,
    cache: &mut KvCache,
    argmax_output: Option<&Tensor>,
) -> Result<(Tensor, PendingBatch<'model>), ModelError> {
    if tokens.dtype() != DType::U32 || tokens.shape().len() != 1 || tokens.is_empty() {
        return Err(ModelError::TokensShape);
    }
    if cache.layer_count() != model.layers.len() {
        return Err(ModelError::CacheLayerCount);
    }
    let requested = cache.reserve(tokens.len())?;
    let mut batch = model.kernels.begin_batch()?;
    let mut hidden = batch.embedding(tokens, &model.embedding)?;
    let offset = cache.len();
    for (layer_index, layer) in model.layers.iter().enumerate() {
        let layer_cache = cache.layer(layer_index)?;
        hidden = forward_layer(
            model,
            &mut batch,
            hidden,
            layer,
            layer_cache,
            offset,
            requested,
        )?;
    }
    let normalized = batch.rms_norm(&hidden, &model.final_norm, model.config.rms_norm_eps)?;
    let last = normalized.row(tokens.len() - 1)?;
    let logits = batch.matmul(&last, &model.lm_head)?;
    if let Some(output) = argmax_output {
        batch.argmax(&logits.reshape(&[logits.len()])?, output)?;
    }
    let pending = batch.commit()?;
    cache.advance(tokens.len())?;
    Ok((logits, pending))
}

pub fn forward_layer(
    model: &Qwen3Model,
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
    let fused_qkv = tokens == 1 && model.plan.fusions.decode_norm && model.plan.fusions.qkv;
    let (query, key, value) = if fused_qkv {
        batch.rms_norm_matmul3(
            &hidden,
            &layer.input_norm,
            &layer.attention.query,
            &layer.attention.key,
            &layer.attention.value,
            model.config.rms_norm_eps,
        )?
    } else {
        let normalized = batch.rms_norm(&hidden, &layer.input_norm, model.config.rms_norm_eps)?;
        if model.plan.fusions.qkv {
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
        model.config.num_attention_heads,
        model.config.head_dim,
    ])?;
    let key = key.reshape(&[
        tokens,
        model.config.num_key_value_heads,
        model.config.head_dim,
    ])?;
    let value = value.reshape(&[
        tokens,
        model.config.num_key_value_heads,
        model.config.head_dim,
    ])?;
    let query = if model.plan.fusions.qk_rope_cache {
        batch.qk_norm_rope_cache(
            &query,
            &key,
            &layer.attention.query_norm,
            &layer.attention.key_norm,
            &cache.key,
            QkNormRopeCacheConfig {
                offset,
                theta: model.config.rope_theta,
                epsilon: model.config.rms_norm_eps,
            },
        )?
    } else {
        let query = batch.rms_norm(
            &query,
            &layer.attention.query_norm,
            model.config.rms_norm_eps,
        )?;
        let query = batch.rope(&query, offset, model.config.rope_theta)?;
        let key = batch.rms_norm(&key, &layer.attention.key_norm, model.config.rms_norm_eps)?;
        let key = batch.rope(&key, offset, model.config.rope_theta)?;
        batch.copy_into_cache(&key, &cache.key, offset)?;
        query
    };
    batch.copy_into_cache(&value, &cache.value, offset)?;
    let active_key = cache.key.prefix(active_length)?;
    let active_value = cache.value.prefix(active_length)?;
    let attention_kind = model.plan.attention_for_tokens(
        tokens,
        active_length,
        model.config.num_attention_heads / model.config.num_key_value_heads,
    );
    let attention = batch.attention(
        &query,
        &active_key,
        &active_value,
        AttentionConfig {
            query_heads: model.config.num_attention_heads,
            kv_heads: model.config.num_key_value_heads,
            head_dim: model.config.head_dim,
            causal: true,
            query_offset: offset,
        },
        attention_kind,
    )?;
    let attention = attention.reshape(&[tokens, model.config.query_width()])?;
    let attention = batch.matmul(&attention, &layer.attention.output)?;
    let fused_gate_up = tokens == 1
        && model.plan.fusions.decode_norm
        && model.plan.fusions.add_rms_norm
        && model.plan.fusions.gate_up;
    let (residual, gate, up) = if fused_gate_up {
        batch.add_rms_norm_matmul2(
            &hidden,
            &attention,
            &layer.post_attention_norm,
            &layer.mlp.gate,
            &layer.mlp.up,
            model.config.rms_norm_eps,
        )?
    } else {
        let (residual, normalized) = if model.plan.fusions.add_rms_norm {
            batch.add_rms_norm(
                &hidden,
                &attention,
                &layer.post_attention_norm,
                model.config.rms_norm_eps,
            )?
        } else {
            let residual = batch.add(&hidden, &attention)?;
            let normalized = batch.rms_norm(
                &residual,
                &layer.post_attention_norm,
                model.config.rms_norm_eps,
            )?;
            (residual, normalized)
        };
        let (gate, up) = if model.plan.fusions.gate_up {
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
