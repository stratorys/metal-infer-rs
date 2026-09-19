use metal_infer_core::{AttentionConfig, AttentionKind, CoreError, MetalContext};

const TOLERANCE: f32 = 0.02;

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn matmul_matches_cpu() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let input = context.tensor_f16(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let weight = context.tensor_f16(&[5.0, 6.0, 7.0, 8.0], &[2, 2])?;
    let actual = context.matmul(&input, &weight)?.to_f32_vec()?;
    assert_close(&actual, &[17.0, 23.0, 39.0, 53.0]);
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn matvec_matches_cpu() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let input = context.tensor_f16(&[1.0, 2.0, 3.0], &[1, 3])?;
    let weight = context.tensor_f16(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[2, 3])?;
    let actual = context.matmul(&input, &weight)?.to_f32_vec()?;
    assert_close(&actual, &[32.0, 50.0]);
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn tiled_attention_matches_reference() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let query_values: Vec<f32> = (0..32).map(|value| value as f32 / 31.0 - 0.5).collect();
    let key_values: Vec<f32> = (0..16).map(|value| value as f32 / 15.0 - 0.25).collect();
    let value_values: Vec<f32> = (0..16).map(|value| value as f32 / 10.0).collect();
    let query = context.tensor_f16(&query_values, &[2, 2, 8])?;
    let key = context.tensor_f16(&key_values, &[2, 1, 8])?;
    let value = context.tensor_f16(&value_values, &[2, 1, 8])?;
    let config = AttentionConfig {
        query_heads: 2,
        kv_heads: 1,
        head_dim: 8,
        causal: true,
        query_offset: 0,
    };
    let reference = context
        .attention(&query, &key, &value, config, AttentionKind::Reference)?
        .to_f32_vec()?;
    let tiled = context
        .attention(&query, &key, &value, config, AttentionKind::Tiled)?
        .to_f32_vec()?;
    assert_close(&tiled, &reference);
    Ok(())
}

fn assert_close(
    actual: &[f32],
    expected: &[f32],
) {
    assert_eq!(actual.len(), expected.len(), "length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= TOLERANCE,
            "element {index}: actual={actual}, expected={expected}"
        );
    }
}
