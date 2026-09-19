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
fn simd_rms_norm_matches_cpu() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let input_values: Vec<f32> = (0..3 * 65)
        .map(|index| (index % 23) as f32 / 11.0 - 1.0)
        .collect();
    let weight_values: Vec<f32> = (0..65).map(|index| 0.5 + index as f32 / 130.0).collect();
    let input = context.tensor_f16(&input_values, &[3, 65])?;
    let weight = context.tensor_f16(&weight_values, &[65])?;
    let actual = context.rms_norm(&input, &weight, 1.0e-6)?.to_f32_vec()?;
    let mut expected = Vec::with_capacity(input_values.len());
    for row in input_values.chunks_exact(65) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() / 65.0;
        let scale = (mean_square + 1.0e-6).sqrt().recip();
        expected.extend(
            row.iter()
                .zip(&weight_values)
                .map(|(value, weight)| value * scale * weight),
        );
    }
    assert_close(&actual, &expected);
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn fused_projections_match_individual_matvecs() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    for k in [128, 127] {
        let input_values: Vec<f32> = (0..k)
            .map(|index| (index % 17) as f32 / 17.0 - 0.5)
            .collect();
        let weights = |rows: usize, modulus: usize| {
            (0..rows * k)
                .map(|index| (index % modulus) as f32 / modulus as f32 - 0.5)
                .collect::<Vec<_>>()
        };
        let input = context.tensor_f16(&input_values, &[1, k])?;
        let weight0 = context.tensor_f16(&weights(35, 19), &[35, k])?;
        let weight1 = context.tensor_f16(&weights(17, 23), &[17, k])?;
        let weight2 = context.tensor_f16(&weights(6, 29), &[6, k])?;
        let expected0 = context.matmul(&input, &weight0)?.to_f32_vec()?;
        let expected1 = context.matmul(&input, &weight1)?.to_f32_vec()?;
        let expected2 = context.matmul(&input, &weight2)?.to_f32_vec()?;

        let mut batch = context.begin_batch()?;
        let (actual0, actual1, actual2) = batch.matmul3(&input, &weight0, &weight1, &weight2)?;
        batch.finish()?;
        assert_close(&actual0.to_f32_vec()?, &expected0);
        assert_close(&actual1.to_f32_vec()?, &expected1);
        assert_close(&actual2.to_f32_vec()?, &expected2);

        let mut batch = context.begin_batch()?;
        let (actual0, actual1) = batch.matmul2(&input, &weight0, &weight1)?;
        batch.finish()?;
        assert_close(&actual0.to_f32_vec()?, &expected0);
        assert_close(&actual1.to_f32_vec()?, &expected1);
    }
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn fused_add_rms_norm_matches_individual_ops() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let left = context.tensor_f16(&[1.0, -2.0, 3.0, 0.5, 0.25, -0.75], &[2, 3])?;
    let right = context.tensor_f16(&[0.5, 1.0, -1.0, 1.5, -0.25, 0.5], &[2, 3])?;
    let weight = context.tensor_f16(&[0.5, 1.0, 1.5], &[3])?;
    let expected_residual = context.add(&left, &right)?;
    let expected_normalized = context.rms_norm(&expected_residual, &weight, 1.0e-6)?;

    let mut batch = context.begin_batch()?;
    let (residual, normalized) = batch.add_rms_norm(&left, &right, &weight, 1.0e-6)?;
    batch.finish()?;
    assert_close(&residual.to_f32_vec()?, &expected_residual.to_f32_vec()?);
    assert_close(
        &normalized.to_f32_vec()?,
        &expected_normalized.to_f32_vec()?,
    );
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn fused_qk_transform_matches_individual_ops() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let query_values: Vec<f32> = (0..32).map(|index| index as f32 / 16.0 - 1.0).collect();
    let key_values: Vec<f32> = (0..16).map(|index| index as f32 / 12.0 - 0.5).collect();
    let norm_values: Vec<f32> = (0..8).map(|index| 0.75 + index as f32 / 16.0).collect();
    let query = context.tensor_f16(&query_values, &[2, 2, 8])?;
    let key = context.tensor_f16(&key_values, &[2, 1, 8])?;
    let query_weight = context.tensor_f16(&norm_values, &[8])?;
    let key_weight = context.tensor_f16(&norm_values, &[8])?;
    let expected_cache = context.tensor_f16(&[0.0; 4 * 8], &[4, 1, 8])?;
    let actual_cache = context.tensor_f16(&[0.0; 4 * 8], &[4, 1, 8])?;
    let expected_query = context.rms_norm(&query, &query_weight, 1.0e-6)?;
    let expected_query = context.rope(&expected_query, 1, 10_000.0)?;
    let expected_key = context.rms_norm(&key, &key_weight, 1.0e-6)?;
    let expected_key = context.rope(&expected_key, 1, 10_000.0)?;
    context.copy_into_cache(&expected_key, &expected_cache, 1)?;

    let mut batch = context.begin_batch()?;
    let actual_query = batch.qk_norm_rope_cache(
        &query,
        &key,
        &query_weight,
        &key_weight,
        &actual_cache,
        1,
        10_000.0,
        1.0e-6,
    )?;
    batch.finish()?;
    assert_close(&actual_query.to_f32_vec()?, &expected_query.to_f32_vec()?);
    assert_close(&actual_cache.to_f32_vec()?, &expected_cache.to_f32_vec()?);
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
