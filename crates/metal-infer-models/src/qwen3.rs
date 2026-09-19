use std::fs;
use std::path::Path;

use metal_infer_core::{AttentionConfig, AttentionKind, CommandBatch, DType, MetalContext, Tensor};

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

struct LayerCache {
    key: Tensor,
    value: Tensor,
}

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
    config: Qwen3Config,
    embedding: Tensor,
    layers: Vec<LayerWeights>,
    final_norm: Tensor,
    lm_head: Tensor,
    attention_kind: AttentionKind,
}

impl Qwen3Model {
    pub fn load(
        directory: &Path,
        context: &MetalContext,
    ) -> Result<Self, ModelError> {
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
        Ok(Self {
            context: context.clone(),
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            attention_kind: AttentionKind::Tiled,
        })
    }

    pub const fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub fn set_attention_kind(
        &mut self,
        kind: AttentionKind,
    ) {
        self.attention_kind = kind;
    }

    pub fn prefill(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
    ) -> Result<Tensor, ModelError> {
        if tokens.is_empty() {
            return Err(ModelError::Config(
                "prefill requires at least one token".into(),
            ));
        }
        cache.reset();
        self.forward(tokens, cache)
    }

    pub fn decode(
        &self,
        token: u32,
        cache: &mut KvCache,
    ) -> Result<Tensor, ModelError> {
        self.forward(&[token], cache)
    }

    pub fn generate(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        cache: &mut KvCache,
    ) -> Result<Vec<u32>, ModelError> {
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        let mut logits = self.prefill(prompt, cache)?;
        let mut generated = Vec::with_capacity(max_tokens);
        for step in 0..max_tokens {
            let token = argmax(&logits.to_f32_vec()?)?;
            generated.push(token);
            if step + 1 < max_tokens {
                logits = self.decode(token, cache)?;
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
        let mut batch = self.context.begin_batch()?;
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

    fn forward(
        &self,
        tokens: &[u32],
        cache: &mut KvCache,
    ) -> Result<Tensor, ModelError> {
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
        let token_tensor = self.context.tensor_u32(tokens, &[tokens.len()])?;
        let mut batch = self.context.begin_batch()?;
        let mut hidden = batch.embedding(&token_tensor, &self.embedding)?;
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
        batch.finish()?;
        cache.filled = requested;
        Ok(logits)
    }

    fn forward_layer(
        &self,
        batch: &mut CommandBatch<'_>,
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
        let normalized = batch.rms_norm(&hidden, &layer.input_norm, self.config.rms_norm_eps)?;
        let query = batch
            .matmul(&normalized, &layer.attention.query)?
            .reshape(&[
                tokens,
                self.config.num_attention_heads,
                self.config.head_dim,
            ])?;
        let key = batch.matmul(&normalized, &layer.attention.key)?.reshape(&[
            tokens,
            self.config.num_key_value_heads,
            self.config.head_dim,
        ])?;
        let value = batch
            .matmul(&normalized, &layer.attention.value)?
            .reshape(&[
                tokens,
                self.config.num_key_value_heads,
                self.config.head_dim,
            ])?;
        let query = batch.rms_norm(
            &query,
            &layer.attention.query_norm,
            self.config.rms_norm_eps,
        )?;
        let key = batch.rms_norm(&key, &layer.attention.key_norm, self.config.rms_norm_eps)?;
        let query = batch.rope(&query, offset, self.config.rope_theta)?;
        let key = batch.rope(&key, offset, self.config.rope_theta)?;
        batch.copy_into_cache(&key, &cache.key, offset)?;
        batch.copy_into_cache(&value, &cache.value, offset)?;
        let active_key = cache.key.prefix(active_length)?;
        let active_value = cache.value.prefix(active_length)?;
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
            self.attention_kind,
        )?;
        let attention = attention.reshape(&[tokens, self.config.query_width()])?;
        let attention = batch.matmul(&attention, &layer.attention.output)?;
        let residual = batch.add(&hidden, &attention)?;
        let normalized = batch.rms_norm(
            &residual,
            &layer.post_attention_norm,
            self.config.rms_norm_eps,
        )?;
        let gate = batch.matmul(&normalized, &layer.mlp.gate)?;
        let up = batch.matmul(&normalized, &layer.mlp.up)?;
        let activated = batch.swiglu(&gate, &up)?;
        let down = batch.matmul(&activated, &layer.mlp.down)?;
        batch.add(&residual, &down).map_err(Into::into)
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
