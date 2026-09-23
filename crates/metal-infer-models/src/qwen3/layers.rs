use metal_infer_kernels::Tensor;

use crate::{ModelError, Qwen3Config};

pub struct AttentionWeights {
    pub query: Tensor,
    pub key: Tensor,
    pub value: Tensor,
    pub output: Tensor,
    pub query_norm: Tensor,
    pub key_norm: Tensor,
}

pub struct MlpWeights {
    pub gate: Tensor,
    pub up: Tensor,
    pub down: Tensor,
}

pub struct LayerWeights {
    pub input_norm: Tensor,
    pub post_attention_norm: Tensor,
    pub attention: AttentionWeights,
    pub mlp: MlpWeights,
}

pub fn validate_layer(
    config: &Qwen3Config,
    attention: &AttentionWeights,
    mlp: &MlpWeights,
) -> Result<(), ModelError> {
    expect_shape(
        &attention.query,
        &[config.query_width(), config.hidden_size],
    )?;
    expect_shape(&attention.key, &[config.kv_width(), config.hidden_size])?;
    expect_shape(&attention.value, &[config.kv_width(), config.hidden_size])?;
    expect_shape(
        &attention.output,
        &[config.hidden_size, config.query_width()],
    )?;
    expect_shape(&attention.query_norm, &[config.head_dim])?;
    expect_shape(&attention.key_norm, &[config.head_dim])?;
    expect_shape(&mlp.gate, &[config.intermediate_size, config.hidden_size])?;
    expect_shape(&mlp.up, &[config.intermediate_size, config.hidden_size])?;
    expect_shape(&mlp.down, &[config.hidden_size, config.intermediate_size])?;
    Ok(())
}

pub fn expect_shape(
    tensor: &Tensor,
    expected: &[usize],
) -> Result<(), ModelError> {
    if tensor.shape() == expected {
        Ok(())
    } else {
        Err(ModelError::WeightShape)
    }
}
