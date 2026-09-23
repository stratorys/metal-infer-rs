use metal_infer_kernels::Tensor;

use crate::{ModelError, Qwen3Config};

pub(super) struct AttentionWeights {
    pub(super) query: Tensor,
    pub(super) key: Tensor,
    pub(super) value: Tensor,
    pub(super) output: Tensor,
    pub(super) query_norm: Tensor,
    pub(super) key_norm: Tensor,
}

pub(super) struct MlpWeights {
    pub(super) gate: Tensor,
    pub(super) up: Tensor,
    pub(super) down: Tensor,
}

pub(super) struct LayerWeights {
    pub(super) input_norm: Tensor,
    pub(super) post_attention_norm: Tensor,
    pub(super) attention: AttentionWeights,
    pub(super) mlp: MlpWeights,
}

pub(super) fn validate_layer(
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

pub(super) fn expect_shape(
    tensor: &Tensor,
    expected: &[usize],
) -> Result<(), ModelError> {
    if tensor.shape() == expected {
        Ok(())
    } else {
        Err(ModelError::WeightShape)
    }
}
