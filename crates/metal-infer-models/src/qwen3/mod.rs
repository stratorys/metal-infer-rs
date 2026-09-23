mod forward;
mod layers;

use std::fs;
use std::path::Path;

use metal_infer_kernels::{DType, DispatchStats, Kernels, MetalContext, PendingBatch, Tensor};
use metal_infer_planner::{Plan, PlanInputs};

use crate::kv_cache::LayerCache;
use crate::qwen3::layers::{
    AttentionWeights, LayerWeights, MlpWeights, expect_shape, validate_layer,
};
use crate::sampling::{XorShift64, sample_token_f16, validate_generation_options};
use crate::weights::WeightMap;
use crate::{GenerationOptions, KvCache, ModelError, Qwen3Config};

pub struct Qwen3Model {
    context: MetalContext,
    kernels: Kernels,
    config: Qwen3Config,
    embedding: Tensor,
    layers: Vec<LayerWeights>,
    final_norm: Tensor,
    lm_head: Tensor,
    plan: Plan,
}

impl Qwen3Model {
    pub const fn context(&self) -> &MetalContext {
        &self.context
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
        let config: Qwen3Config =
            serde_json::from_slice(&fs::read(directory.join("config.json"))?)?;
        config.validate()?;
        let plan = Plan::resolve(
            PlanInputs {
                device: kernels.device(),
                query_heads: config.num_attention_heads,
                kv_heads: config.num_key_value_heads,
                hidden_size: config.hidden_size,
            },
            overrides,
        )?;
        let kernels = kernels.with_selection(plan.kernels.clone())?;
        let context = kernels.context().clone();
        let mut weights = WeightMap::load(directory, &context)?;
        let embedding = weights.take("model.embed_tokens.weight")?;
        expect_shape(&embedding, &[config.vocab_size, config.hidden_size])?;
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
        expect_shape(&final_norm, &[config.hidden_size])?;
        expect_shape(&lm_head, &[config.vocab_size, config.hidden_size])?;
        Ok(Self {
            context,
            kernels,
            config,
            embedding,
            layers,
            final_norm,
            lm_head,
            plan,
        })
    }

    pub const fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub const fn plan(&self) -> &Plan {
        &self.plan
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
            return Err(ModelError::EmptyPrefill);
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
            return Err(ModelError::EmptyPrefill);
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
            return Err(ModelError::DecodeTokenShape);
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
                .ok_or(ModelError::MissingArgmaxToken)?;
            if token == u32::MAX {
                if let Some((pending, previous)) = prefetched.take() {
                    let result = pending.wait();
                    cache.filled = previous;
                    result?;
                }
                return Err(ModelError::NoFiniteLogit);
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
            return Err(ModelError::EmptyBlock);
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
        let layer = self.layers.first().ok_or(ModelError::NoLayer)?;
        let output = self.forward_layer(&mut batch, hidden, layer, &cache, 0, tokens.len())?;
        batch.finish()?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use metal_infer_kernels::MetalContext;

    use crate::sampling::argmax;
    use crate::{GenerationOptions, KvCache, Qwen3Model};

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
