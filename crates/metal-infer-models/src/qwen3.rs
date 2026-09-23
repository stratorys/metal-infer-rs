use std::fs;
use std::path::Path;

use half::f16;

use metal_infer_kernels::{
    AttentionConfig, AttentionKind, KernelBatch, Kernels, QkNormRopeCacheConfig,
};
use metal_infer_planner::{Fusions, Plan};
use metal_infer_runtime::{DType, DispatchStats, MetalContext, PendingBatch, Tensor};

use crate::weights::WeightMap;
use crate::{ModelError, Qwen3Config};

struct AttentionWeights {
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    query_norm: Tensor,
    key_norm: Tensor,
}

struct MlpWeights {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

struct LayerWeights {
    input_norm: Tensor,
    post_attention_norm: Tensor,
    attention: AttentionWeights,
    mlp: MlpWeights,
}

#[derive(Clone)]
struct LayerCache {
    key: Tensor,
    value: Tensor,
}

const INITIAL_FUSIONS: Fusions = Fusions {
    qkv: true,
    gate_up: true,
    add_rms_norm: true,
    qk_rope_cache: true,
    decode_norm: false,
};

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
            return Err(ModelError::Config(
                "KV cache capacity must be non-zero".into(),
            ));
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

pub struct Qwen3Model {
    context: MetalContext,
    kernels: Kernels,
    config: Qwen3Config,
    embedding: Tensor,
    layers: Vec<LayerWeights>,
    final_norm: Tensor,
    lm_head: Tensor,
    fusions: Fusions,
    attention: AttentionKind,
}

#[derive(Clone, Debug)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub stop_token_ids: Vec<u32>,
}

pub struct TokenSampler {
    options: GenerationOptions,
    random: XorShift64,
}

impl TokenSampler {
    pub fn new(options: GenerationOptions) -> Result<Self, ModelError> {
        validate_generation_options(&options)?;
        let random = XorShift64::new(options.seed);
        Ok(Self { options, random })
    }

    pub fn sample(
        &mut self,
        logits: &Tensor,
    ) -> Result<u32, ModelError> {
        logits.with_f16_bits(|bits| {
            if self.options.temperature == 0.0 {
                bits.iter()
                    .enumerate()
                    .filter_map(|(index, bits)| {
                        let value = f16::from_bits(*bits).to_f32();
                        value.is_finite().then_some((index, value))
                    })
                    .max_by(|left, right| {
                        left.1
                            .total_cmp(&right.1)
                            .then_with(|| right.0.cmp(&left.0))
                    })
                    .map(|(index, _)| index as u32)
                    .ok_or_else(|| ModelError::Config("logits contain no finite value".into()))
            } else {
                sample_token_f16(bits, &self.options, &mut self.random)
            }
        })?
    }
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 32,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            stop_token_ids: Vec::new(),
        }
    }
}

impl Qwen3Model {
    pub const fn context(&self) -> &MetalContext {
        &self.context
    }
    pub fn decode_batch(
        &self,
        tokens: &[u32],
        caches: &mut [&mut KvCache],
    ) -> Result<Tensor, ModelError> {
        if tokens.is_empty() || tokens.len() != caches.len() {
            return Err(ModelError::Config(
                "decode batch needs one cache per token".into(),
            ));
        }
        for cache in caches.iter() {
            if cache.filled == 0 || cache.filled >= cache.capacity {
                return Err(ModelError::CacheCapacity {
                    capacity: cache.capacity,
                    requested: cache.filled.saturating_add(1),
                });
            }
        }
        let input = self.context.tensor_u32(tokens, &[tokens.len()])?;
        let mut batch = self.kernels.begin_batch()?;
        let mut hidden = batch.embedding(&input, &self.embedding)?;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let normalized =
                batch.rms_norm(&hidden, &layer.input_norm, self.config.rms_norm_eps)?;
            let (query, key, value) = if self.fusions.qkv {
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
                    .ok_or_else(|| ModelError::Config("missing KV cache layer".into()))?;
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
                let query_row = if self.fusions.qk_rope_cache {
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
                let kind = attention_kind_for_tokens(
                    self.attention,
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
            let (residual, normalized) = if self.fusions.add_rms_norm {
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
            let (gate, up) = if self.fusions.gate_up {
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
    pub fn load(
        directory: &Path,
        context: &MetalContext,
    ) -> Result<Self, ModelError> {
        Self::load_with(directory, Kernels::new(context)?, &[])
    }

    pub fn load_with(
        directory: &Path,
        kernels: Kernels,
        overrides: &[String],
    ) -> Result<Self, ModelError> {
        let context = kernels.context();
        let mut initial = Plan {
            kernels: kernels.selection(),
            fusions: INITIAL_FUSIONS,
            attention: AttentionKind::Tiled,
        };
        for assignment in overrides {
            initial.apply_override(assignment)?;
        }
        kernels.select(&initial.kernels)?;
        let config: Qwen3Config =
            serde_json::from_slice(&fs::read(directory.join("config.json"))?)?;
        config.validate()?;
        let mut weights = WeightMap::load(directory, context)?;
        let embedding = weights.take("model.embed_tokens.weight")?;
        expect_shape(
            &embedding,
            &[config.vocab_size, config.hidden_size],
            "embedding",
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            let prefix = format!("model.layers.{index}");
            let attention = AttentionWeights {
                query: weights.take(&format!("{prefix}.self_attn.q_proj.weight"))?,
                key: weights.take(&format!("{prefix}.self_attn.k_proj.weight"))?,
                value: weights.take(&format!("{prefix}.self_attn.v_proj.weight"))?,
                output: weights.take(&format!("{prefix}.self_attn.o_proj.weight"))?,
                query_norm: weights.take(&format!("{prefix}.self_attn.q_norm.weight"))?,
                key_norm: weights.take(&format!("{prefix}.self_attn.k_norm.weight"))?,
            };
            let mlp = MlpWeights {
                gate: weights.take(&format!("{prefix}.mlp.gate_proj.weight"))?,
                up: weights.take(&format!("{prefix}.mlp.up_proj.weight"))?,
                down: weights.take(&format!("{prefix}.mlp.down_proj.weight"))?,
            };
            validate_layer(&config, &attention, &mlp)?;
            layers.push(LayerWeights {
                input_norm: weights.take(&format!("{prefix}.input_layernorm.weight"))?,
                post_attention_norm: weights
                    .take(&format!("{prefix}.post_attention_layernorm.weight"))?,
                attention,
                mlp,
            });
        }
        let final_norm = weights.take("model.norm.weight")?;
        let lm_head = if config.tie_word_embeddings {
            embedding.clone()
        } else {
            weights.take("lm_head.weight")?
        };
        expect_shape(&final_norm, &[config.hidden_size], "final norm")?;
        expect_shape(
            &lm_head,
            &[config.vocab_size, config.hidden_size],
            "LM head",
        )?;
        kernels.select(
            &kernels.device_selection(config.num_attention_heads, config.num_key_value_heads),
        )?;
        let decode_norm =
            context.device_name() == "Apple M4 Pro" && config.hidden_size.is_multiple_of(256);
        let mut model = Self {
            context: context.clone(),
            kernels: kernels.clone(),
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            fusions: Fusions {
                decode_norm,
                ..initial.fusions
            },
            attention: initial.attention,
        };
        let mut plan = model.plan();
        for assignment in overrides {
            plan.apply_override(assignment)?;
        }
        model.apply_plan(&plan)?;
        Ok(model)
    }

    pub const fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub fn plan(&self) -> Plan {
        Plan {
            kernels: self.kernels.selection(),
            fusions: self.fusions,
            attention: self.attention,
        }
    }

    fn apply_plan(
        &mut self,
        plan: &Plan,
    ) -> Result<(), ModelError> {
        self.kernels.select(&plan.kernels)?;
        self.fusions = plan.fusions;
        self.attention = plan.attention;
        Ok(())
    }

    pub fn prefill(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
    ) -> Result<Tensor, ModelError> {
        self.prefill_with_stats(tokens, cache)
            .map(|(tensor, _)| tensor)
    }

    pub fn prefill_with_stats(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
    ) -> Result<(Tensor, DispatchStats), ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::Config(
                "prefill requires at least one token".into(),
            ));
        }
        cache.reset();
        self.forward_with_stats(tokens, cache)
    }

    pub fn decode(
        &self,
        token: u32,
        cache: &mut KvCache,
    ) -> Result<Tensor, ModelError> {
        self.decode_with_stats(token, cache)
            .map(|(tensor, _)| tensor)
    }

    pub fn decode_with_stats(
        &self,
        token: u32,
        cache: &mut KvCache,
    ) -> Result<(Tensor, DispatchStats), ModelError> {
        self.forward_with_stats(&[token], cache)
    }

    pub fn prefill_argmax(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
        output: &Tensor,
    ) -> Result<(Tensor, DispatchStats), ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::Config(
                "prefill requires at least one token".into(),
            ));
        }
        cache.reset();
        let token_tensor = self.context.tensor_u32(tokens, &[tokens.len()])?;
        let previous = cache.filled;
        let (logits, pending) = self.forward_async(&token_tensor, cache, Some(output))?;
        match pending.wait() {
            Ok(stats) => Ok((logits, stats)),
            Err(error) => {
                cache.filled = previous;
                Err(error.into())
            }
        }
    }

    pub fn decode_argmax<'model>(
        &'model self,
        token: &Tensor,
        cache: &mut KvCache,
        output: &Tensor,
    ) -> Result<(Tensor, PendingBatch<'model>), ModelError> {
        if token.shape() != [1] || token.dtype() != DType::U32 {
            return Err(ModelError::Config("decode token must be one u32".into()));
        }
        self.forward_async(token, cache, Some(output))
    }

    pub fn generate(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        cache: &mut KvCache,
    ) -> Result<Vec<u32>, ModelError> {
        self.generate_with(
            prompt,
            &GenerationOptions {
                max_tokens,
                ..GenerationOptions::default()
            },
            cache,
            |_| true,
        )
    }

    pub fn generate_with(
        &self,
        prompt: &[u32],
        options: &GenerationOptions,
        cache: &mut KvCache,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, ModelError> {
        validate_generation_options(options)?;
        if options.max_tokens == 0 {
            return Ok(Vec::new());
        }
        if options.temperature == 0.0 {
            return self.generate_greedy(prompt, options, cache, on_token);
        }
        let mut logits = self.prefill(prompt, cache)?;
        let mut generated = Vec::with_capacity(options.max_tokens);
        let mut random = XorShift64::new(options.seed);
        for step in 0..options.max_tokens {
            let token =
                logits.with_f16_bits(|bits| sample_token_f16(bits, options, &mut random))??;
            if options.stop_token_ids.contains(&token) {
                break;
            }
            generated.push(token);
            if !on_token(token) {
                break;
            }
            if step + 1 < options.max_tokens {
                logits = self.decode(token, cache)?;
            }
        }
        Ok(generated)
    }

    fn generate_greedy(
        &self,
        prompt: &[u32],
        options: &GenerationOptions,
        cache: &mut KvCache,
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>, ModelError> {
        let slots = self
            .context
            .tensor_u32(&vec![u32::MAX; options.max_tokens], &[options.max_tokens])?;
        let first = slots.slice_1d(0, 1)?;
        let _ = self.prefill_argmax(prompt, cache, &first)?;
        let mut generated = Vec::with_capacity(options.max_tokens);
        let mut prefetched: Option<(PendingBatch<'_>, usize)> = None;
        for step in 0..options.max_tokens {
            let input = slots.slice_1d(step, 1)?;
            let token = input
                .to_u32_vec()?
                .first()
                .copied()
                .ok_or_else(|| ModelError::Config("missing argmax token".into()))?;
            if token == u32::MAX {
                if let Some((pending, previous)) = prefetched.take() {
                    let result = pending.wait();
                    cache.filled = previous;
                    result?;
                }
                return Err(ModelError::Config("logits contain no finite value".into()));
            }
            let pending = if step + 1 < options.max_tokens {
                if let Some(pending) = prefetched.take() {
                    Some(pending)
                } else {
                    let output = slots.slice_1d(step + 1, 1)?;
                    let previous = cache.filled;
                    Some((self.decode_argmax(&input, cache, &output)?.1, previous))
                }
            } else {
                None
            };
            let stopped = options.stop_token_ids.contains(&token);
            let continued = if stopped {
                false
            } else {
                generated.push(token);
                on_token(token)
            };
            if !continued {
                if let Some((pending, previous)) = pending {
                    let result = pending.wait();
                    cache.filled = previous;
                    result?;
                }
                break;
            }
            if let Some((pending, previous)) = pending {
                let next = if step + 2 < options.max_tokens {
                    let next_input = slots.slice_1d(step + 1, 1)?;
                    let next_output = slots.slice_1d(step + 2, 1)?;
                    let next_previous = cache.filled;
                    match self.decode_argmax(&next_input, cache, &next_output) {
                        Ok((_, batch)) => Some((batch, next_previous)),
                        Err(error) => {
                            let _ = pending.wait();
                            cache.filled = previous;
                            return Err(error);
                        }
                    }
                } else {
                    None
                };
                if let Err(error) = pending.wait() {
                    drop(next);
                    cache.filled = previous;
                    return Err(error.into());
                }
                prefetched = next;
            }
        }
        Ok(generated)
    }

    pub fn run_first_block(
        &self,
        tokens: &[u32],
    ) -> Result<Tensor, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::Config("block benchmark requires tokens".into()));
        }
        let token_tensor = self.context.tensor_u32(tokens, &[tokens.len()])?;
        let mut batch = self.kernels.begin_batch()?;
        let hidden = batch.embedding(&token_tensor, &self.embedding)?;
        let shape = [
            tokens.len(),
            self.config.num_key_value_heads,
            self.config.head_dim,
        ];
        let cache = LayerCache {
            key: self.context.empty(&shape, DType::F16)?,
            value: self.context.empty(&shape, DType::F16)?,
        };
        let layer = self
            .layers
            .first()
            .ok_or_else(|| ModelError::Config("model has no transformer layer".into()))?;
        let output = self.forward_layer(&mut batch, hidden, layer, &cache, 0, tokens.len())?;
        batch.finish()?;
        Ok(output)
    }

    fn forward_with_stats(
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

    fn forward_async<'model>(
        &'model self,
        tokens: &Tensor,
        cache: &mut KvCache,
        argmax_output: Option<&Tensor>,
    ) -> Result<(Tensor, PendingBatch<'model>), ModelError> {
        if tokens.dtype() != DType::U32 || tokens.shape().len() != 1 || tokens.is_empty() {
            return Err(ModelError::Config(
                "tokens must be a nonempty u32 vector".into(),
            ));
        }
        if cache.layers.len() != self.layers.len() {
            return Err(ModelError::Config(
                "KV cache layer count differs from model".into(),
            ));
        }
        let requested = cache.filled + tokens.len();
        if requested > cache.capacity {
            return Err(ModelError::CacheCapacity {
                capacity: cache.capacity,
                requested,
            });
        }
        let mut batch = self.kernels.begin_batch()?;
        let mut hidden = batch.embedding(tokens, &self.embedding)?;
        let offset = cache.filled;
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_cache = cache
                .layers
                .get(layer_index)
                .ok_or_else(|| ModelError::Config("missing KV cache layer".into()))?;
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

    fn forward_layer(
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
            .ok_or_else(|| ModelError::Config("hidden state has no token dimension".into()))?;
        let fused_qkv = tokens == 1 && self.fusions.decode_norm && self.fusions.qkv;
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
            if self.fusions.qkv {
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
        let query = if self.fusions.qk_rope_cache {
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
        let attention_kind = attention_kind_for_tokens(
            self.attention,
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
            && self.fusions.decode_norm
            && self.fusions.add_rms_norm
            && self.fusions.gate_up;
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
            let (residual, normalized) = if self.fusions.add_rms_norm {
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
            let (gate, up) = if self.fusions.gate_up {
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

const fn attention_kind_for_tokens(
    configured: AttentionKind,
    tokens: usize,
    active_length: usize,
    query_heads_per_kv: usize,
) -> AttentionKind {
    match (configured, tokens) {
        (AttentionKind::Tiled, 32..) => AttentionKind::FlashPrefill,
        (AttentionKind::Tiled, 1) if active_length >= 256 && query_heads_per_kv == 2 => {
            AttentionKind::FlashDecode
        }
        (AttentionKind::Tiled, 1) => AttentionKind::DecodeSplitKv,
        (AttentionKind::DecodeSplitKv, 1) => AttentionKind::DecodeSplitKv,
        (AttentionKind::DecodeSplitKv, _) => AttentionKind::Tiled,
        (AttentionKind::FlashDecode, 1) if query_heads_per_kv == 2 => AttentionKind::FlashDecode,
        (AttentionKind::FlashDecode, 1) => AttentionKind::DecodeSplitKv,
        (AttentionKind::FlashDecode, _) => AttentionKind::Tiled,
        (AttentionKind::FlashPrefill, 1) if active_length >= 256 && query_heads_per_kv == 2 => {
            AttentionKind::FlashDecode
        }
        (AttentionKind::FlashPrefill, 1) => AttentionKind::DecodeSplitKv,
        (kind, _) => kind,
    }
}

fn validate_layer(
    config: &Qwen3Config,
    attention: &AttentionWeights,
    mlp: &MlpWeights,
) -> Result<(), ModelError> {
    expect_shape(
        &attention.query,
        &[config.query_width(), config.hidden_size],
        "q_proj",
    )?;
    expect_shape(
        &attention.key,
        &[config.kv_width(), config.hidden_size],
        "k_proj",
    )?;
    expect_shape(
        &attention.value,
        &[config.kv_width(), config.hidden_size],
        "v_proj",
    )?;
    expect_shape(
        &attention.output,
        &[config.hidden_size, config.query_width()],
        "o_proj",
    )?;
    expect_shape(&attention.query_norm, &[config.head_dim], "q_norm")?;
    expect_shape(&attention.key_norm, &[config.head_dim], "k_norm")?;
    expect_shape(
        &mlp.gate,
        &[config.intermediate_size, config.hidden_size],
        "gate_proj",
    )?;
    expect_shape(
        &mlp.up,
        &[config.intermediate_size, config.hidden_size],
        "up_proj",
    )?;
    expect_shape(
        &mlp.down,
        &[config.hidden_size, config.intermediate_size],
        "down_proj",
    )?;
    Ok(())
}

fn expect_shape(
    tensor: &Tensor,
    expected: &[usize],
    name: &str,
) -> Result<(), ModelError> {
    if tensor.shape() == expected {
        Ok(())
    } else {
        Err(ModelError::Config(format!(
            "{name} has shape {:?}, expected {expected:?}",
            tensor.shape()
        )))
    }
}

#[cfg(test)]
fn argmax(values: &[f32]) -> Result<u32, ModelError> {
    let (index, _) = values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or_else(|| ModelError::Config("logits contain no finite value".into()))?;
    index
        .try_into()
        .map_err(|_| ModelError::Config("token id does not fit in u32".into()))
}

fn validate_generation_options(options: &GenerationOptions) -> Result<(), ModelError> {
    if !options.temperature.is_finite() || options.temperature < 0.0 {
        return Err(ModelError::Config(
            "temperature must be finite and non-negative".into(),
        ));
    }
    if !options.top_p.is_finite() || !(0.0..=1.0).contains(&options.top_p) {
        return Err(ModelError::Config("top_p must be between 0 and 1".into()));
    }
    Ok(())
}

#[cfg(test)]
fn sample_token(
    values: &[f32],
    options: &GenerationOptions,
    random: &mut XorShift64,
) -> Result<u32, ModelError> {
    if options.temperature == 0.0 {
        return argmax(values);
    }
    let candidates: Vec<(usize, f32)> = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .collect();
    sample_candidates(candidates, options, random)
}

fn sample_token_f16(
    bits: &[u16],
    options: &GenerationOptions,
    random: &mut XorShift64,
) -> Result<u32, ModelError> {
    let candidates: Vec<(usize, f32)> = bits
        .iter()
        .enumerate()
        .map(|(index, bits)| (index, f16::from_bits(*bits).to_f32()))
        .filter(|(_, value)| value.is_finite())
        .collect();
    sample_candidates(candidates, options, random)
}

fn sample_candidates(
    mut candidates: Vec<(usize, f32)>,
    options: &GenerationOptions,
    random: &mut XorShift64,
) -> Result<u32, ModelError> {
    if candidates.is_empty() {
        return Err(ModelError::Config("logits contain no finite value".into()));
    }
    if options.top_k > 0 && candidates.len() > options.top_k {
        let mut original = candidates.clone();
        candidates.select_nth_unstable_by(options.top_k, |left, right| right.1.total_cmp(&left.1));
        let (top, rest) = candidates.split_at_mut(options.top_k);
        top.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        let has_tie = top.windows(2).any(|pair| {
            pair.first()
                .zip(pair.get(1))
                .is_some_and(|(left, right)| left.1 == right.1)
        }) || top
            .last()
            .zip(rest.first())
            .is_some_and(|(last, next)| last.1 == next.1);
        if has_tie {
            original.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
            original.truncate(options.top_k);
            candidates = original;
        } else {
            candidates.truncate(options.top_k);
        }
    } else {
        candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    }
    let max_logit = candidates
        .first()
        .ok_or_else(|| ModelError::Config("logits contain no finite value".into()))?
        .1;
    let inverse_temperature = options.temperature.recip();
    let mut total = 0.0f64;
    for (_, value) in &mut candidates {
        *value = ((*value - max_logit) * inverse_temperature).exp();
        total += f64::from(*value);
    }
    if options.top_p < 1.0 {
        let threshold = total * f64::from(options.top_p);
        let mut cumulative = 0.0f64;
        let mut keep = 0usize;
        for (_, probability) in &candidates {
            cumulative += f64::from(*probability);
            keep += 1;
            if cumulative >= threshold {
                break;
            }
        }
        candidates.truncate(keep.max(1));
        total = candidates
            .iter()
            .map(|(_, probability)| f64::from(*probability))
            .sum();
    }
    let mut target = random.next_f64() * total;
    for (index, probability) in candidates {
        target -= f64::from(probability);
        if target <= 0.0 {
            return index
                .try_into()
                .map_err(|_| ModelError::Config("token id does not fit in u32".into()));
        }
    }
    Err(ModelError::Config("failed to sample a token".into()))
}

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_f64(&mut self) -> f64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        (value as f64) / (u64::MAX as f64 + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionKind, GenerationOptions, KvCache, Qwen3Model, XorShift64, argmax,
        attention_kind_for_tokens, sample_token,
    };
    use metal_infer_runtime::MetalContext;

    #[test]
    fn tiled_attention_selects_flash_decode_for_long_gqa_decode() {
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Tiled, 1, 640, 2),
            AttentionKind::FlashDecode,
            "long single-token decode should select flash decode"
        );
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Tiled, 1, 255, 2),
            AttentionKind::DecodeSplitKv,
            "short single-token decode should select split-KV attention"
        );
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Tiled, 1, 640, 4),
            AttentionKind::DecodeSplitKv,
            "other GQA ratios should select split-KV attention"
        );
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Tiled, 31, 31, 2),
            AttentionKind::Tiled,
            "short prefill should keep tiled attention"
        );
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Tiled, 32, 32, 2),
            AttentionKind::FlashPrefill,
            "32-token prefill should select flash prefill"
        );
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Tiled, 512, 512, 2),
            AttentionKind::FlashPrefill,
            "long prefill should select flash prefill"
        );
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Reference, 1, 640, 2),
            AttentionKind::Reference,
            "reference attention should remain explicitly selectable"
        );
    }

    #[test]
    fn zero_temperature_is_greedy() {
        let options = GenerationOptions::default();
        let token = sample_token(&[1.0, 4.0, 2.0], &options, &mut XorShift64::new(1));
        assert_eq!(token.expect("sampling should succeed"), 1);
    }

    #[test]
    fn top_k_one_is_greedy_even_with_temperature() {
        let options = GenerationOptions {
            temperature: 1.0,
            top_k: 1,
            ..GenerationOptions::default()
        };
        let token = sample_token(&[1.0, 4.0, 2.0], &options, &mut XorShift64::new(1));
        assert_eq!(token.expect("sampling should succeed"), 1);
    }

    #[test]
    #[ignore = "requires QWEN3_MODEL and direct access to an Apple Metal device"]
    fn pipelined_greedy_matches_synchronous_bits() {
        let model_path = std::env::var("QWEN3_MODEL").expect("set QWEN3_MODEL");
        let context = MetalContext::new().expect("Metal device");
        let model = Qwen3Model::load(std::path::Path::new(&model_path), &context).expect("model");
        let prompt = vec![1; 512];
        let mut synchronous_cache = KvCache::new(&context, model.config(), 576).expect("cache");
        let mut pipelined_cache = KvCache::new(&context, model.config(), 576).expect("cache");
        let mut synchronous = model
            .prefill(&prompt, &mut synchronous_cache)
            .expect("prefill");
        let slots = context
            .tensor_u32(&vec![u32::MAX; 64], &[64])
            .expect("slots");
        let (mut pipelined, _) = model
            .prefill_argmax(
                &prompt,
                &mut pipelined_cache,
                &slots.slice_1d(0, 1).expect("slot"),
            )
            .expect("prefill argmax");
        let mut tokens = Vec::new();
        let mut queued = None;
        for step in 0..64 {
            let expected_bits = synchronous
                .with_f16_bits(|bits| bits.to_vec())
                .expect("synchronous bits");
            let actual_bits = pipelined
                .with_f16_bits(|bits| bits.to_vec())
                .expect("pipelined bits");
            assert_eq!(actual_bits, expected_bits, "logits at step {step}");
            let expected = argmax(&synchronous.to_f32_vec().expect("logits")).expect("CPU argmax");
            let input = slots.slice_1d(step, 1).expect("slot");
            let actual = input
                .to_u32_vec()
                .expect("GPU argmax")
                .first()
                .copied()
                .expect("one token");
            assert_eq!(actual, expected, "token at step {step}");
            tokens.push(actual);
            if step < 63 {
                let (next, current) = if let Some(batch) = queued.take() {
                    batch
                } else {
                    let output = slots.slice_1d(step + 1, 1).expect("next slot");
                    model
                        .decode_argmax(&input, &mut pipelined_cache, &output)
                        .expect("pipelined decode")
                };
                let following = if step < 62 {
                    let following_input = slots.slice_1d(step + 1, 1).expect("following slot");
                    let following_output = slots.slice_1d(step + 2, 1).expect("following output");
                    Some(
                        model
                            .decode_argmax(
                                &following_input,
                                &mut pipelined_cache,
                                &following_output,
                            )
                            .expect("overlapped decode"),
                    )
                } else {
                    None
                };
                synchronous = model
                    .decode(expected, &mut synchronous_cache)
                    .expect("synchronous decode");
                current.wait().expect("decode completion");
                pipelined = next;
                queued = following;
            }
        }
        let mut generation_cache = KvCache::new(&context, model.config(), 576).expect("cache");
        let generated = model
            .generate_with(
                &prompt,
                &GenerationOptions {
                    max_tokens: 64,
                    ..GenerationOptions::default()
                },
                &mut generation_cache,
                |_| true,
            )
            .expect("generation");
        assert_eq!(generated, tokens, "generate_with tokens");
        let mut early_stop_cache = KvCache::new(&context, model.config(), 576).expect("cache");
        let mut seen = 0;
        let early = model
            .generate_with(
                &prompt,
                &GenerationOptions {
                    max_tokens: 64,
                    ..GenerationOptions::default()
                },
                &mut early_stop_cache,
                |_| {
                    seen += 1;
                    seen < 2
                },
            )
            .expect("early stop");
        assert_eq!(
            early,
            tokens.get(..2).expect("two tokens"),
            "callback keeps emitted tokens"
        );
        assert_eq!(early_stop_cache.len(), 513, "speculative cache rolls back");
        let mut stop_token_cache = KvCache::new(&context, model.config(), 576).expect("cache");
        let stopped = model
            .generate_with(
                &prompt,
                &GenerationOptions {
                    max_tokens: 64,
                    stop_token_ids: vec![*tokens.first().expect("first token")],
                    ..GenerationOptions::default()
                },
                &mut stop_token_cache,
                |_| true,
            )
            .expect("stop token");
        assert!(stopped.is_empty(), "stop token is not emitted");
        assert_eq!(stop_token_cache.len(), 512, "prefill length restored");
    }
}
