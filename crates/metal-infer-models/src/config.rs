use serde::Deserialize;

use crate::ModelError;

#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3Config {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Qwen3Config {
    pub fn validate(&self) -> Result<(), ModelError> {
        if self.model_type != "qwen3" {
            return Err(ModelError::Unsupported(format!(
                "expected model_type qwen3, got {}",
                self.model_type
            )));
        }
        if self.attention_bias {
            return Err(ModelError::Unsupported(
                "attention biases are not implemented".into(),
            ));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
            || self.vocab_size == 0
            || !self.head_dim.is_multiple_of(2)
        {
            return Err(ModelError::Config(
                "model dimensions must be non-zero and head_dim even".into(),
            ));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(ModelError::Config(
                "num_attention_heads must be divisible by num_key_value_heads".into(),
            ));
        }
        Ok(())
    }

    pub const fn query_width(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub const fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::Qwen3Config;

    fn config() -> Qwen3Config {
        Qwen3Config {
            model_type: "qwen3".into(),
            hidden_size: 1024,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1.0e-6,
            rope_theta: 1_000_000.0,
            vocab_size: 151_936,
            attention_bias: false,
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn qwen3_dimensions_are_accepted() {
        let config = config();
        assert!(config.validate().is_ok(), "Qwen3 config should be valid");
        assert_eq!(config.query_width(), 2048, "query width mismatch");
        assert_eq!(config.kv_width(), 1024, "KV width mismatch");
    }

    #[test]
    fn invalid_gqa_ratio_is_rejected() {
        let mut config = config();
        config.num_key_value_heads = 3;
        assert!(
            config.validate().is_err(),
            "invalid GQA ratio must be rejected"
        );
    }
}
