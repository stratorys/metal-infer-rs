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
fn vectorized_matvec_matches_cpu() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let input_values: Vec<f32> = (0..128).map(|index| index as f32 / 128.0 - 0.5).collect();
    let weight_values: Vec<f32> = (0..257 * 128)
        .map(|index| (index % 31) as f32 / 31.0 - 0.5)
        .collect();
    let input = context.tensor_f16(&input_values, &[1, 128])?;
    let weight = context.tensor_f16(&weight_values, &[257, 128])?;
    let actual = context.matmul(&input, &weight)?.to_f32_vec()?;
    let expected: Vec<f32> = weight_values
        .chunks_exact(128)
        .map(|row| {
            row.iter()
                .zip(&input_values)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect();
    assert_close(&actual, &expected);
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn batched_dependent_kernels_match_eager_execution() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let left = context.tensor_f16(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let right = context.tensor_f16(&[0.5, 1.0, 1.5, 2.0], &[2, 2])?;

    let eager = context.add(&context.add(&left, &right)?, &right)?;

    let mut batch = context.begin_batch()?;
    let intermediate = batch.add(&left, &right)?;
    let batched = batch.add(&intermediate, &right)?;
    drop(intermediate);
    let batched = batch.add(&batched, &right)?;
    batch.finish()?;

    let expected = context.add(&eager, &right)?;
    assert_close(&batched.to_f32_vec()?, &expected.to_f32_vec()?);
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn invalid_batch_can_be_abandoned() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let input = context.tensor_f16(&[1.0, 2.0], &[1, 2])?;
    let incompatible_weight = context.tensor_f16(&[1.0, 2.0, 3.0], &[1, 3])?;
    let mut batch = context.begin_batch()?;

    let result = batch.matmul(&input, &incompatible_weight);
    assert!(result.is_err(), "incompatible matmul should fail");
    drop(batch);
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn tiled_attention_matches_reference() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let query_values: Vec<f32> = (0..2 * 2 * 128)
        .map(|value| (value % 41) as f32 / 41.0 - 0.5)
        .collect();
    let key_values: Vec<f32> = (0..2 * 128)
        .map(|value| (value % 37) as f32 / 37.0 - 0.25)
        .collect();
    let value_values: Vec<f32> = (0..2 * 128)
        .map(|value| (value % 29) as f32 / 29.0)
        .collect();
    let query = context.tensor_f16(&query_values, &[2, 2, 128])?;
    let key = context.tensor_f16(&key_values, &[2, 1, 128])?;
    let value = context.tensor_f16(&value_values, &[2, 1, 128])?;
    let config = AttentionConfig {
        query_heads: 2,
        kv_heads: 1,
        head_dim: 128,
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
