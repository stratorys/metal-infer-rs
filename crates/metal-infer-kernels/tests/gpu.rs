use half::f16;
use metal_infer_kernels::{
    AttentionConfig, AttentionKind, DecodeGemvConfig, KernelSelection, Kernels,
    QkNormRopeCacheConfig,
};
use metal_infer_runtime::{CoreError, MetalContext};

const TOLERANCE: f32 = 0.02;

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn argmax_f16_matches_cpu() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    for size in [1, 2, 31, 32, 33, 2048, 2049, 151_936] {
        let mut bits: Vec<u16> = (0..size)
            .map(|index| ((index as u32 * 1103 + 17) & 0xffff) as u16)
            .collect();
        *bits.first_mut().expect("nonempty logits") = 0x7c00; // infinity is ignored
        if size > 1 {
            *bits.get_mut(1).expect("second logit") = 0x7e00; // NaN is ignored
            *bits.get_mut(size / 2).expect("middle logit") = 0x7bff;
            *bits.last_mut().expect("last logit") = 0x7bff; // the last equal maximum wins
        } else {
            *bits.first_mut().expect("only logit") = 0;
        }
        let expected = bits
            .iter()
            .enumerate()
            .map(|(index, bits)| (index, f16::from_bits(*bits).to_f32()))
            .filter(|(_, value)| value.is_finite())
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .map(|(index, _)| index as u32)
            .unwrap_or(u32::MAX);
        let logits = context.tensor_f16_bits(&bits, &[size])?;
        let output = context.tensor_u32(&[0], &[1])?;
        let mut batch = kernels.begin_batch()?;
        batch.argmax(&logits, &output)?;
        batch.finish()?;
        assert_eq!(output.to_u32_vec()?, vec![expected], "size {size}");
    }
    for bits in [[0x8000, 0], [0x7c00, 0x7e00]] {
        let logits = context.tensor_f16_bits(&bits, &[2])?;
        let output = context.tensor_u32(&[0], &[1])?;
        let mut batch = kernels.begin_batch()?;
        batch.argmax(&logits, &output)?;
        batch.finish()?;
        assert_eq!(
            *output.to_u32_vec()?.first().expect("argmax output"),
            if bits.first().copied() == Some(0x8000) {
                1
            } else {
                u32::MAX
            }
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn speculative_embedding_handles_missing_argmax() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let logits = context.tensor_f16_bits(&[0x7c00, 0x7e00], &[2])?;
    let token = context.tensor_u32(&[0], &[1])?;
    let table = context.tensor_f16(&[1.0, 2.0], &[1, 2])?;
    let mut first = kernels.begin_batch()?;
    first.argmax(&logits, &token)?;
    let first = first.commit()?;
    let mut second = kernels.begin_batch()?;
    let embedded = second.embedding(&token, &table)?;
    let second = second.commit()?;
    first.wait()?;
    second.wait()?;
    assert_eq!(token.to_u32_vec()?, vec![u32::MAX]);
    assert_eq!(embedded.to_f32_vec()?, vec![0.0, 0.0]);
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn matmul_matches_cpu() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let input = context.tensor_f16(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let weight = context.tensor_f16(&[5.0, 6.0, 7.0, 8.0], &[2, 2])?;
    let actual = kernels.matmul(&input, &weight)?.to_f32_vec()?;
    assert_close(&actual, &[17.0, 23.0, 39.0, 53.0]);
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn matmul_variants_match_cpu_reference() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    for (m, n, k) in [
        (128, 256, 256),
        (129, 256, 256),
        (511, 256, 256),
        (32, 32, 32),
        (31, 35, 37),
    ] {
        let input_values: Vec<f32> = (0..m * k)
            .map(|index| (index % 19) as f32 / 19.0 - 0.5)
            .collect();
        let weight_values: Vec<f32> = (0..n * k)
            .map(|index| (index % 23) as f32 / 23.0 - 0.5)
            .collect();
        let input = context.tensor_f16(&input_values, &[m, k])?;
        let weight = context.tensor_f16(&weight_values, &[n, k])?;
        let actual = kernels.matmul(&input, &weight)?.to_f32_vec()?;
        assert_close(&actual, &cpu_matmul(&input_values, &weight_values, k));
    }
    Ok(())
}
#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn matvec_matches_cpu() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let input = context.tensor_f16(&[1.0, 2.0, 3.0], &[1, 3])?;
    let weight = context.tensor_f16(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[2, 3])?;
    let actual = kernels.matmul(&input, &weight)?.to_f32_vec()?;
    assert_close(&actual, &[32.0, 50.0]);
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn kernel_profiler_records_only_enabled_dispatches() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let input = context.tensor_f16(&[1.0, 2.0, 3.0], &[1, 3])?;
    let weight = context.tensor_f16(&[4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[2, 3])?;
    context.set_kernel_profiling(true)?;
    let actual = kernels.matmul(&input, &weight)?.to_f32_vec()?;
    assert_close(&actual, &[32.0, 50.0]);
    let profiles = context.take_kernel_profiles();
    assert_eq!(profiles.len(), 1);
    let profile = profiles.first().expect("one profiled dispatch");
    assert_eq!(profile.kernel, "matvec_f16");
    assert!(profile.gpu_time > std::time::Duration::ZERO);
    context.set_kernel_profiling(false)?;
    let _ = kernels.matmul(&input, &weight)?;
    assert!(context.take_kernel_profiles().is_empty());
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn vectorized_matvec_matches_cpu() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let input_values: Vec<f32> = (0..128).map(|index| index as f32 / 128.0 - 0.5).collect();
    let weight_values: Vec<f32> = (0..257 * 128)
        .map(|index| (index % 31) as f32 / 31.0 - 0.5)
        .collect();
    let input = context.tensor_f16(&input_values, &[1, 128])?;
    let weight = context.tensor_f16(&weight_values, &[257, 128])?;
    let actual = kernels.matmul(&input, &weight)?.to_f32_vec()?;
    let expected: Vec<f32> = weight_values
        .as_chunks::<128>()
        .0
        .iter()
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
fn matvec_variants_match_cpu_reference_at_qwen_shapes() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    for (n, k) in [(1024, 1024), (1024, 2048), (1024, 3072), (65_537, 256)] {
        let input_values: Vec<f32> = (0..k)
            .map(|index| (index % 17) as f32 / 34.0 - 0.25)
            .collect();
        let weight_values: Vec<f32> = (0..n * k)
            .map(|index| (index % 29) as f32 / 58.0 - 0.25)
            .collect();
        let input = context.tensor_f16(&input_values, &[1, k])?;
        let weight = context.tensor_f16(&weight_values, &[n, k])?;
        let expected = cpu_matmul(&input_values, &weight_values, k);
        for config in [
            None,
            Some(DecodeGemvConfig::Baseline),
            Some(DecodeGemvConfig::Tuned),
        ] {
            select(&kernels, |selection| selection.decode_gemv = config)?;
            let actual = kernels.matmul(&input, &weight)?.to_f32_vec()?;
            assert_close(&actual, &expected);
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn simd_rms_norm_matches_cpu() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let input_values: Vec<f32> = (0..3 * 65)
        .map(|index| (index % 23) as f32 / 11.0 - 1.0)
        .collect();
    let weight_values: Vec<f32> = (0..65).map(|index| 0.5 + index as f32 / 130.0).collect();
    let input = context.tensor_f16(&input_values, &[3, 65])?;
    let weight = context.tensor_f16(&weight_values, &[65])?;
    let actual = kernels.rms_norm(&input, &weight, 1.0e-6)?.to_f32_vec()?;
    let mut expected = Vec::with_capacity(input_values.len());
    for row in input_values.as_chunks::<65>().0 {
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
fn fused_projections_match_cpu_reference() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    for k in [1024, 128, 127] {
        let input_values: Vec<f32> = (0..k)
            .map(|index| (index % 17) as f32 / 17.0 - 0.5)
            .collect();
        let weights = |rows: usize, modulus: usize| {
            (0..rows * k)
                .map(|index| (index % modulus) as f32 / modulus as f32 - 0.5)
                .collect::<Vec<_>>()
        };
        let (values0, values1, values2) = (weights(35, 19), weights(17, 23), weights(6, 29));
        let input = context.tensor_f16(&input_values, &[1, k])?;
        let weight0 = context.tensor_f16(&values0, &[35, k])?;
        let weight1 = context.tensor_f16(&values1, &[17, k])?;
        let weight2 = context.tensor_f16(&values2, &[6, k])?;
        let expected0 = cpu_matmul(&input_values, &values0, k);
        let expected1 = cpu_matmul(&input_values, &values1, k);
        let expected2 = cpu_matmul(&input_values, &values2, k);

        let mut batch = kernels.begin_batch()?;
        let (actual0, actual1, actual2) = batch.matmul3(&input, &weight0, &weight1, &weight2)?;
        batch.finish()?;
        assert_close(&actual0.to_f32_vec()?, &expected0);
        assert_close(&actual1.to_f32_vec()?, &expected1);
        assert_close(&actual2.to_f32_vec()?, &expected2);

        let mut batch = kernels.begin_batch()?;
        let (actual0, actual1) = batch.matmul2(&input, &weight0, &weight1)?;
        batch.finish()?;
        assert_close(&actual0.to_f32_vec()?, &expected0);
        assert_close(&actual1.to_f32_vec()?, &expected1);
    }
    Ok(())
}
#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn fused_projections_match_cpu_reference_at_qwen_dimensions() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let input_values: Vec<f32> = (0..1024)
        .map(|index| (index % 31) as f32 / 31.0 - 0.5)
        .collect();
    let input = context.tensor_f16(&input_values, &[1, 1024])?;
    let weight_values = |rows: usize, modulus: usize| {
        (0..rows * 1024)
            .map(|index| (index % modulus) as f32 / modulus as f32 - 0.5)
            .collect::<Vec<_>>()
    };
    let query_values = weight_values(2048, 37);
    let key_values = weight_values(1024, 41);
    let value_values = weight_values(1024, 43);
    let gate_values = weight_values(3072, 47);
    let up_values = weight_values(3072, 53);
    let query = context.tensor_f16(&query_values, &[2048, 1024])?;
    let key = context.tensor_f16(&key_values, &[1024, 1024])?;
    let value = context.tensor_f16(&value_values, &[1024, 1024])?;
    let gate = context.tensor_f16(&gate_values, &[3072, 1024])?;
    let up = context.tensor_f16(&up_values, &[3072, 1024])?;

    let mut batch = kernels.begin_batch()?;
    let (actual_query, actual_key, actual_value) = batch.matmul3(&input, &query, &key, &value)?;
    let (actual_gate, actual_up) = batch.matmul2(&input, &gate, &up)?;
    batch.finish()?;
    assert_close(
        &actual_query.to_f32_vec()?,
        &cpu_matmul(&input_values, &query_values, 1024),
    );
    assert_close(
        &actual_key.to_f32_vec()?,
        &cpu_matmul(&input_values, &key_values, 1024),
    );
    assert_close(
        &actual_value.to_f32_vec()?,
        &cpu_matmul(&input_values, &value_values, 1024),
    );
    assert_close(
        &actual_gate.to_f32_vec()?,
        &cpu_matmul(&input_values, &gate_values, 1024),
    );
    assert_close(
        &actual_up.to_f32_vec()?,
        &cpu_matmul(&input_values, &up_values, 1024),
    );
    Ok(())
}
#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn fused_add_rms_norm_matches_individual_ops() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let left = context.tensor_f16(&[1.0, -2.0, 3.0, 0.5, 0.25, -0.75], &[2, 3])?;
    let right = context.tensor_f16(&[0.5, 1.0, -1.0, 1.5, -0.25, 0.5], &[2, 3])?;
    let weight = context.tensor_f16(&[0.5, 1.0, 1.5], &[3])?;
    let expected_residual = kernels.add(&left, &right)?;
    let expected_normalized = kernels.rms_norm(&expected_residual, &weight, 1.0e-6)?;

    let mut batch = kernels.begin_batch()?;
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
fn fused_decode_norm_projections_match_separate_ops() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let input_values: Vec<f32> = (0..256).map(|i| (i % 29) as f32 / 29.0 - 0.5).collect();
    let right_values: Vec<f32> = (0..256).map(|i| (i % 19) as f32 / 38.0 - 0.25).collect();
    let norm_values: Vec<f32> = (0..256).map(|i| 0.5 + (i % 17) as f32 / 34.0).collect();
    let make_weight = |rows: usize, modulus: usize| {
        let values: Vec<f32> = (0..rows * 256)
            .map(|i| (i % modulus) as f32 / modulus as f32 - 0.5)
            .collect();
        context.tensor_f16(&values, &[rows, 256])
    };
    let input = context.tensor_f16(&input_values, &[1, 256])?;
    let right = context.tensor_f16(&right_values, &[1, 256])?;
    let norm_weight = context.tensor_f16(&norm_values, &[256])?;
    let weight0 = make_weight(40, 31)?;
    let weight1 = make_weight(24, 37)?;
    let weight2 = make_weight(16, 41)?;
    let normalized = kernels.rms_norm(&input, &norm_weight, 1.0e-6)?;
    let mut batch = kernels.begin_batch()?;
    let (expected0, expected1, expected2) =
        batch.matmul3(&normalized, &weight0, &weight1, &weight2)?;
    let (actual0, actual1, actual2) =
        batch.rms_norm_matmul3(&input, &norm_weight, &weight0, &weight1, &weight2, 1.0e-6)?;
    batch.finish()?;
    assert_close(&actual0.to_f32_vec()?, &expected0.to_f32_vec()?);
    assert_close(&actual1.to_f32_vec()?, &expected1.to_f32_vec()?);
    assert_close(&actual2.to_f32_vec()?, &expected2.to_f32_vec()?);
    let mut batch = kernels.begin_batch()?;
    let (expected_residual, expected_normalized) =
        batch.add_rms_norm(&input, &right, &norm_weight, 1.0e-6)?;
    let (expected_gate, expected_up) = batch.matmul2(&expected_normalized, &weight0, &weight1)?;
    let (actual_residual, actual_gate, actual_up) =
        batch.add_rms_norm_matmul2(&input, &right, &norm_weight, &weight0, &weight1, 1.0e-6)?;
    batch.finish()?;
    assert_close(
        &actual_residual.to_f32_vec()?,
        &expected_residual.to_f32_vec()?,
    );
    assert_close(&actual_gate.to_f32_vec()?, &expected_gate.to_f32_vec()?);
    assert_close(&actual_up.to_f32_vec()?, &expected_up.to_f32_vec()?);
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn fused_decode_norm_projections_match_separate_ops_at_qwen_dimensions() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let values = |count: usize, modulus: usize, scale: f32| {
        (0..count)
            .map(|i| (i % modulus) as f32 / modulus as f32 * scale - scale / 2.0)
            .collect::<Vec<_>>()
    };
    let input = context.tensor_f16(&values(1024, 29, 1.0), &[1, 1024])?;
    let right = context.tensor_f16(&values(1024, 19, 0.5), &[1, 1024])?;
    let norm_values: Vec<f32> = (0..1024).map(|i| 0.5 + (i % 17) as f32 / 34.0).collect();
    let norm_weight = context.tensor_f16(&norm_values, &[1024])?;
    let query = context.tensor_f16(&values(2048 * 1024, 31, 0.1), &[2048, 1024])?;
    let key = context.tensor_f16(&values(1024 * 1024, 37, 0.1), &[1024, 1024])?;
    let value = context.tensor_f16(&values(1024 * 1024, 41, 0.1), &[1024, 1024])?;
    let gate = context.tensor_f16(&values(3072 * 1024, 43, 0.1), &[3072, 1024])?;
    let up = context.tensor_f16(&values(3072 * 1024, 47, 0.1), &[3072, 1024])?;
    for config in [
        None,
        Some(DecodeGemvConfig::Baseline),
        Some(DecodeGemvConfig::Tuned),
    ] {
        select(&kernels, |selection| selection.decode_gemv = config)?;
        let normalized = kernels.rms_norm(&input, &norm_weight, 1.0e-6)?;
        let mut batch = kernels.begin_batch()?;
        let (expected_query, expected_key, expected_value) =
            batch.matmul3(&normalized, &query, &key, &value)?;
        let (actual_query, actual_key, actual_value) =
            batch.rms_norm_matmul3(&input, &norm_weight, &query, &key, &value, 1.0e-6)?;
        let (expected_residual, expected_normalized) =
            batch.add_rms_norm(&input, &right, &norm_weight, 1.0e-6)?;
        let (expected_gate, expected_up) = batch.matmul2(&expected_normalized, &gate, &up)?;
        let (actual_residual, actual_gate, actual_up) =
            batch.add_rms_norm_matmul2(&input, &right, &norm_weight, &gate, &up, 1.0e-6)?;
        batch.finish()?;
        assert_close(&actual_query.to_f32_vec()?, &expected_query.to_f32_vec()?);
        assert_close(&actual_key.to_f32_vec()?, &expected_key.to_f32_vec()?);
        assert_close(&actual_value.to_f32_vec()?, &expected_value.to_f32_vec()?);
        assert_close(
            &actual_residual.to_f32_vec()?,
            &expected_residual.to_f32_vec()?,
        );
        assert_close(&actual_gate.to_f32_vec()?, &expected_gate.to_f32_vec()?);
        assert_close(&actual_up.to_f32_vec()?, &expected_up.to_f32_vec()?);
    }
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn fused_qk_transform_matches_individual_ops() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let query_values: Vec<f32> = (0..32).map(|index| index as f32 / 16.0 - 1.0).collect();
    let key_values: Vec<f32> = (0..16).map(|index| index as f32 / 12.0 - 0.5).collect();
    let norm_values: Vec<f32> = (0..8).map(|index| 0.75 + index as f32 / 16.0).collect();
    let query = context.tensor_f16(&query_values, &[2, 2, 8])?;
    let key = context.tensor_f16(&key_values, &[2, 1, 8])?;
    let query_weight = context.tensor_f16(&norm_values, &[8])?;
    let key_weight = context.tensor_f16(&norm_values, &[8])?;
    let expected_cache = context.tensor_f16(&[0.0; 4 * 8], &[4, 1, 8])?;
    let actual_cache = context.tensor_f16(&[0.0; 4 * 8], &[4, 1, 8])?;
    let expected_query = kernels.rms_norm(&query, &query_weight, 1.0e-6)?;
    let expected_query = kernels.rope(&expected_query, 1, 10_000.0)?;
    let expected_key = kernels.rms_norm(&key, &key_weight, 1.0e-6)?;
    let expected_key = kernels.rope(&expected_key, 1, 10_000.0)?;
    kernels.copy_into_cache(&expected_key, &expected_cache, 1)?;

    let mut batch = kernels.begin_batch()?;
    let actual_query = batch.qk_norm_rope_cache(
        &query,
        &key,
        &query_weight,
        &key_weight,
        &actual_cache,
        QkNormRopeCacheConfig {
            offset: 1,
            theta: 10_000.0,
            epsilon: 1.0e-6,
        },
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
    let kernels = Kernels::new(&context)?;
    let left = context.tensor_f16(&[1.0, 2.0, 3.0, 4.0], &[2, 2])?;
    let right = context.tensor_f16(&[0.5, 1.0, 1.5, 2.0], &[2, 2])?;

    let eager = kernels.add(&kernels.add(&left, &right)?, &right)?;

    let mut batch = kernels.begin_batch()?;
    let intermediate = batch.add(&left, &right)?;
    let batched = batch.add(&intermediate, &right)?;
    drop(intermediate);
    let batched = batch.add(&batched, &right)?;
    batch.finish()?;

    let expected = kernels.add(&eager, &right)?;
    assert_close(&batched.to_f32_vec()?, &expected.to_f32_vec()?);
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn invalid_batch_can_be_abandoned() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let input = context.tensor_f16(&[1.0, 2.0], &[1, 2])?;
    let incompatible_weight = context.tensor_f16(&[1.0, 2.0, 3.0], &[1, 3])?;
    let mut batch = kernels.begin_batch()?;

    let result = batch.matmul(&input, &incompatible_weight);
    assert!(result.is_err(), "incompatible matmul should fail");
    drop(batch);
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn tiled_attention_matches_reference() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
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
    let reference = kernels
        .attention(&query, &key, &value, config, AttentionKind::Reference)?
        .to_f32_vec()?;
    let tiled = kernels
        .attention(&query, &key, &value, config, AttentionKind::Tiled)?
        .to_f32_vec()?;
    assert_close(&tiled, &reference);

    let query_values: Vec<f32> = (0..2 * 128)
        .map(|value| (value % 43) as f32 / 43.0 - 0.5)
        .collect();
    let key_values: Vec<f32> = (0..17 * 128)
        .map(|value| (value % 37) as f32 / 37.0 - 0.25)
        .collect();
    let value_values: Vec<f32> = (0..17 * 128)
        .map(|value| (value % 29) as f32 / 29.0)
        .collect();
    let query = context.tensor_f16(&query_values, &[1, 2, 128])?;
    let key = context.tensor_f16(&key_values, &[17, 1, 128])?;
    let value = context.tensor_f16(&value_values, &[17, 1, 128])?;
    let config = AttentionConfig {
        query_heads: 2,
        kv_heads: 1,
        head_dim: 128,
        causal: true,
        query_offset: 16,
    };
    let reference = kernels
        .attention(&query, &key, &value, config, AttentionKind::Reference)?
        .to_f32_vec()?;
    let tiled = kernels
        .attention(&query, &key, &value, config, AttentionKind::Tiled)?
        .to_f32_vec()?;
    assert_close(&tiled, &reference);
    let decode = kernels
        .attention(&query, &key, &value, config, AttentionKind::DecodeSplitKv)?
        .to_f32_vec()?;
    assert_close(&decode, &reference);
    let flash = kernels
        .attention(&query, &key, &value, config, AttentionKind::FlashDecode)?
        .to_f32_vec()?;
    assert_close(&flash, &reference);
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn flash_prefill_matches_reference_at_tile_edges() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    for head_dim in [1, 2, 4, 8, 16, 32, 64, 128, 256] {
        for (tokens, length) in [
            (1, 1),
            (31, 31),
            (32, 32),
            (33, 33),
            (63, 63),
            (64, 64),
            (65, 65),
            (33, 65),
        ] {
            let query_heads = 4;
            let kv_heads = 2;
            let query_values: Vec<f32> = (0..tokens * query_heads * head_dim)
                .map(|index| (index % 43) as f32 / 43.0 - 0.5)
                .collect();
            let key_values: Vec<f32> = (0..length * kv_heads * head_dim)
                .map(|index| (index % 37) as f32 / 37.0 - 0.25)
                .collect();
            let value_values: Vec<f32> = (0..length * kv_heads * head_dim)
                .map(|index| (index % 29) as f32 / 29.0 - 0.5)
                .collect();
            let query = context.tensor_f16(&query_values, &[tokens, query_heads, head_dim])?;
            let key = context.tensor_f16(&key_values, &[length, kv_heads, head_dim])?;
            let value = context.tensor_f16(&value_values, &[length, kv_heads, head_dim])?;
            let config = AttentionConfig {
                query_heads,
                kv_heads,
                head_dim,
                causal: true,
                query_offset: length - tokens,
            };
            let expected = kernels
                .attention(&query, &key, &value, config, AttentionKind::Reference)?
                .to_f32_vec()?;
            let actual = kernels
                .attention(&query, &key, &value, config, AttentionKind::FlashPrefill)?
                .to_f32_vec()?;
            assert_close(&actual, &expected);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn flash_decode_matches_reference_across_block_boundaries() -> Result<(), CoreError> {
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let query_values: Vec<f32> = (0..2 * 128)
        .map(|index| if index % 3 == 0 { 2.0 } else { -2.0 })
        .collect();
    let query = context.tensor_f16(&query_values, &[1, 2, 128])?;
    for length in [1, 7, 17, 33, 63, 64, 65, 256, 640, 2049, 8192] {
        let key_values: Vec<f32> = (0..length * 128)
            .map(|index| if index % 7 < 3 { 2.0 } else { -2.0 })
            .collect();
        let value_values: Vec<f32> = (0..length * 128)
            .map(|index| (index % 37) as f32 / 37.0 - 0.5)
            .collect();
        let key = context.tensor_f16(&key_values, &[length, 1, 128])?;
        let value = context.tensor_f16(&value_values, &[length, 1, 128])?;
        let config = AttentionConfig {
            query_heads: 2,
            kv_heads: 1,
            head_dim: 128,
            causal: true,
            query_offset: length - 1,
        };
        let reference = kernels
            .attention(&query, &key, &value, config, AttentionKind::Reference)?
            .to_f32_vec()?;
        for block in [32, 64, 128, 256] {
            let flash = kernels
                .attention_flash_decode_with_block(&query, &key, &value, config, block)?
                .to_f32_vec()?;
            assert_close(&flash, &reference);
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires direct access to an Apple Metal device"]
fn flash_decode_matches_reference_for_multiple_gqa_groups_and_causal_limit() -> Result<(), CoreError>
{
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let query_values: Vec<f32> = (0..16 * 128)
        .map(|index| (index % 41) as f32 / 41.0 - 0.5)
        .collect();
    let key_values: Vec<f32> = (0..65 * 8 * 128)
        .map(|index| (index % 37) as f32 / 37.0 - 0.5)
        .collect();
    let value_values: Vec<f32> = (0..65 * 8 * 128)
        .map(|index| (index % 29) as f32 / 29.0)
        .collect();
    let query = context.tensor_f16(&query_values, &[1, 16, 128])?;
    let key = context.tensor_f16(&key_values, &[65, 8, 128])?;
    let value = context.tensor_f16(&value_values, &[65, 8, 128])?;
    for offset in [62, 64] {
        let config = AttentionConfig {
            query_heads: 16,
            kv_heads: 8,
            head_dim: 128,
            causal: true,
            query_offset: offset,
        };
        let reference = kernels
            .attention(&query, &key, &value, config, AttentionKind::Reference)?
            .to_f32_vec()?;
        for block in [32, 64, 128, 256] {
            let flash = kernels
                .attention_flash_decode_with_block(&query, &key, &value, config, block)?
                .to_f32_vec()?;
            assert_close(&flash, &reference);
        }
    }
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

fn cpu_matmul(
    input: &[f32],
    weight: &[f32],
    k: usize,
) -> Vec<f32> {
    let round = |values: &[f32]| {
        values
            .iter()
            .map(|value| f16::from_f32(*value).to_f32())
            .collect::<Vec<_>>()
    };
    let input = round(input);
    let weight = round(weight);
    input
        .chunks_exact(k)
        .flat_map(|row| {
            weight.chunks_exact(k).map(move |column| {
                row.iter()
                    .zip(column)
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
            })
        })
        .collect()
}

fn select(
    kernels: &Kernels,
    change: impl FnOnce(&mut KernelSelection),
) -> Result<(), CoreError> {
    let mut selection = kernels.selection();
    change(&mut selection);
    kernels.select(&selection)
}
