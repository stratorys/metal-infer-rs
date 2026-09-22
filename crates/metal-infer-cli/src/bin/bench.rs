use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};
use metal_infer_cli::CliError;
use metal_infer_core::{
    AttentionConfig, AttentionKind, DecodeGemvConfig, DispatchStats, KernelDispatchProfile,
    MatmulBackend, MetalContext, QkNormRopeCacheConfig, Tensor,
};
use metal_infer_models::{FusionOptions, KvCache, Qwen3Model};
use serde::Serialize;

#[derive(Parser)]
#[command(name = "metal-infer-bench", about = "Metal transformer benchmarks")]
struct Arguments {
    #[command(subcommand)]
    command: Command,
    #[arg(long, value_enum, default_value_t = Format::Table, global = true)]
    format: Format,
}

#[derive(Subcommand)]
enum Command {
    Kernel {
        #[arg(long, default_value_t = 512)]
        m: usize,
        #[arg(long, default_value_t = 1024)]
        n: usize,
        #[arg(long, default_value_t = 1024)]
        k: usize,
        #[arg(long, default_value_t = 10)]
        iterations: usize,
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        /// Distinct weight matrices dispatched in one command buffer.
        #[arg(long)]
        rotate: Option<usize>,
        /// Force rows per SIMD group for the single-matrix auto GEMV path.
        #[arg(long)]
        rows: Option<usize>,
        /// Divide K into this many partial GEMV dispatches, then reduce.
        #[arg(long)]
        split_k: Option<usize>,
        /// Compare 16-byte vector loads in the one-row GEMV kernel.
        #[arg(long)]
        half8: bool,
        #[arg(long, value_enum, default_value_t = MatmulBackendArgument::Auto)]
        matmul_backend: MatmulBackendArgument,
    },
    Fusion {
        #[arg(long, value_enum)]
        kind: FusionKind,
        #[arg(long, default_value_t = 1024)]
        k: usize,
        #[arg(long, default_value_t = 3072)]
        intermediate: usize,
        #[arg(long, default_value_t = 16)]
        query_heads: usize,
        #[arg(long, default_value_t = 8)]
        kv_heads: usize,
        #[arg(long, default_value_t = 128)]
        head_dim: usize,
        #[arg(long, default_value_t = 1024)]
        cache_capacity: usize,
        #[arg(long, default_value_t = 10)]
        iterations: usize,
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        /// Distinct sets of projection weights dispatched in one command buffer.
        #[arg(long)]
        rotate: Option<usize>,
        /// Force rows per SIMD group for the fused auto GEMV path.
        #[arg(long)]
        rows: Option<usize>,
        #[arg(long, value_enum, default_value_t = MatmulBackendArgument::Auto)]
        matmul_backend: MatmulBackendArgument,
    },
    Attention {
        #[arg(long, value_enum, default_value_t = AttentionBenchmarkKind::Compare)]
        kind: AttentionBenchmarkKind,
        #[arg(long, default_value_t = 1)]
        tokens: usize,
        #[arg(long, default_value_t = 640)]
        length: usize,
        #[arg(long, default_value_t = 16)]
        query_heads: usize,
        #[arg(long, default_value_t = 8)]
        kv_heads: usize,
        #[arg(long, default_value_t = 128)]
        head_dim: usize,
        #[arg(long, default_value_t = 50)]
        iterations: usize,
        #[arg(long, default_value_t = 10)]
        warmup: usize,
    },
    Block {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 128)]
        tokens: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
    },
    Model {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 512)]
        prompt: usize,
        #[arg(long, default_value_t = 128)]
        generate: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long)]
        fuse_qkv: bool,
        #[arg(long)]
        fuse_gate_up: bool,
        #[arg(long)]
        fuse_add_rms_norm: bool,
        #[arg(long)]
        fuse_qk_rope_cache: bool,
        /// Rows per SIMD group in the fused decode QKV plus RMSNorm kernel.
        #[arg(long)]
        qkv_rms_rows: Option<usize>,
        /// Rows per SIMD group in the fused decode gate/up plus add/RMSNorm kernel.
        #[arg(long)]
        gate_up_add_rms_rows: Option<usize>,
        /// Fixed decode GEMV configuration for a same-binary A/B comparison.
        #[arg(long, value_enum)]
        gemv_config: Option<DecodeGemvConfigArgument>,
        #[arg(long, value_enum, default_value_t = MatmulBackendArgument::Auto)]
        matmul_backend: MatmulBackendArgument,
        #[arg(long)]
        profile_kernels: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Table,
    Json,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum MatmulBackendArgument {
    Auto,
    ReferenceMsl,
    NativeMsl,
    Mps,
}

impl From<MatmulBackendArgument> for MatmulBackend {
    fn from(value: MatmulBackendArgument) -> Self {
        match value {
            MatmulBackendArgument::Auto => Self::Auto,
            MatmulBackendArgument::ReferenceMsl => Self::ReferenceMsl,
            MatmulBackendArgument::NativeMsl => Self::NativeMsl,
            MatmulBackendArgument::Mps => Self::Mps,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DecodeGemvConfigArgument {
    Baseline,
    Tuned,
}

impl From<DecodeGemvConfigArgument> for DecodeGemvConfig {
    fn from(value: DecodeGemvConfigArgument) -> Self {
        match value {
            DecodeGemvConfigArgument::Baseline => Self::Baseline,
            DecodeGemvConfigArgument::Tuned => Self::Tuned,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FusionKind {
    Qkv,
    GateUp,
    QkvRms,
    GateUpAddRms,
    AddRmsNorm,
    QkRopeCache,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum AttentionBenchmarkKind {
    Compare,
    Reference,
    Tiled,
    FlashPrefill,
    DecodeSplitKv,
    FlashDecode,
}

#[derive(Serialize)]
struct Report {
    benchmark: String,
    device: String,
    iterations: usize,
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_mean_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_median_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_p95_ms: Option<f64>,
    throughput: Option<f64>,
    throughput_unit: Option<&'static str>,
    allocated_bytes: usize,
    matmul_backend: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_gemv_config: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    allocation_growth_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fusions: Option<FusionSelection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparison: Option<FusionComparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attention_comparison: Option<AttentionComparison>,
    prefill: Option<PhaseReport>,
    decode: Option<PhaseReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kernel_profile: Option<KernelProfileReport>,
}

#[derive(Serialize)]
struct KernelProfileReport {
    diagnostic_only: bool,
    prefill: KernelPhaseProfile,
    decode: KernelPhaseProfile,
}

#[derive(Serialize)]
struct KernelPhaseProfile {
    gpu_ms: f64,
    attributed_ms: f64,
    unattributed_ms: f64,
    kernels: Vec<KernelProfileRow>,
}

#[derive(Serialize)]
struct KernelProfileRow {
    kernel: String,
    calls: usize,
    total_gpu_ms: f64,
    mean_gpu_ms: f64,
}

#[derive(Serialize)]
struct PhaseReport {
    tokens: usize,
    wall_mean_ms: f64,
    wall_median_ms: f64,
    wall_p95_ms: f64,
    gpu_mean_ms: f64,
    gpu_median_ms: f64,
    gpu_p95_ms: f64,
    tokens_per_second: f64,
}

#[derive(Clone, Copy, Serialize)]
struct FusionSelection {
    qkv: bool,
    gate_up: bool,
    add_rms_norm: bool,
    qk_rope_cache: bool,
}

impl From<FusionOptions> for FusionSelection {
    fn from(options: FusionOptions) -> Self {
        Self {
            qkv: options.qkv,
            gate_up: options.gate_up,
            add_rms_norm: options.add_rms_norm,
            qk_rope_cache: options.qk_rope_cache,
        }
    }
}

#[derive(Serialize)]
struct FusionComparison {
    unfused: TimingReport,
    fused: TimingReport,
    gpu_speedup: f64,
    wall_speedup: f64,
}

#[derive(Serialize)]
struct TimingReport {
    wall_mean_ms: f64,
    wall_median_ms: f64,
    wall_p95_ms: f64,
    gpu_mean_ms: f64,
    gpu_median_ms: f64,
    gpu_p95_ms: f64,
}

#[derive(Serialize)]
struct AttentionComparison {
    reference: AttentionVariantReport,
    tiled: AttentionVariantReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    flash_prefill: Option<AttentionVariantReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_split_kv: Option<AttentionVariantReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    flash_decode: Option<AttentionVariantReport>,
}

#[derive(Serialize)]
struct AttentionVariantReport {
    kind: &'static str,
    #[serde(flatten)]
    timing: TimingReport,
    max_abs_error: f64,
}

#[derive(Clone, Copy)]
struct PhaseSample {
    wall: Duration,
    gpu: Duration,
}

#[derive(Clone, Copy)]
struct DurationStats {
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
}

#[derive(Clone, Copy)]
struct FusionDimensions {
    k: usize,
    intermediate: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    cache_capacity: usize,
}

#[derive(Clone, Copy)]
struct AttentionDimensions {
    tokens: usize,
    length: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

#[derive(Clone, Copy)]
struct AttentionCase<'tensor> {
    context: &'tensor MetalContext,
    query: &'tensor Tensor,
    key: &'tensor Tensor,
    value: &'tensor Tensor,
    config: AttentionConfig,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), CliError> {
    let arguments = Arguments::parse();
    let context = MetalContext::new()?;
    let report = match arguments.command {
        Command::Kernel {
            m,
            n,
            k,
            iterations,
            warmup,
            rotate,
            rows,
            split_k,
            half8,
            matmul_backend,
        } => {
            require_iterations(iterations)?;
            context.set_matmul_backend(matmul_backend.into());
            if let Some(rows) = rows {
                if matmul_backend != MatmulBackendArgument::Auto || m != 1 || split_k.is_some() {
                    return Err(CliError::InvalidArguments(
                        "--rows requires auto GEMV and cannot be combined with --split-k".into(),
                    ));
                }
                if n >= 65_536 && rows == 1 {
                    return Err(CliError::InvalidArguments(
                        "--rows 1 is unavailable for vocabulary GEMV".into(),
                    ));
                }
                context.set_auto_matvec_rows(rows, 2, 2, if n >= 65_536 { rows } else { 0 })?;
            }
            if let Some(splits) = split_k {
                if matmul_backend != MatmulBackendArgument::Auto || m != 1 {
                    return Err(CliError::InvalidArguments(
                        "--split-k requires --matmul-backend auto and m=1".into(),
                    ));
                }
                context.set_auto_matvec_split_k(splits)?;
            }
            if half8 {
                if matmul_backend != MatmulBackendArgument::Auto || m != 1 || rows != Some(1) {
                    return Err(CliError::InvalidArguments(
                        "--half8 requires auto GEMV, m=1, and --rows 1".into(),
                    ));
                }
                context.set_auto_matvec_half8(true);
            }
            if let Some(copies) = rotate {
                run_rotated_matvec(
                    &context, m, n, k, copies, iterations, warmup, rows, split_k, half8,
                )?
            } else {
                let input = context.tensor_f16(&vec![0.01; m * k], &[m, k])?;
                let weight = context.tensor_f16(&vec![0.02; n * k], &[n, k])?;
                for _ in 0..warmup {
                    let _ = dispatch_matmul(&context, &input, &weight)?;
                }
                let (samples, gpu_samples) =
                    measure_dispatch(iterations, || dispatch_matmul(&context, &input, &weight))?;
                let operations = 2.0 * m as f64 * n as f64 * k as f64;
                let throughput = (operations / 1.0e12) / mean_seconds(&samples);
                report(
                    format!("matmul_f16[{m},{n},{k}]"),
                    &context,
                    samples,
                    Some(gpu_samples),
                    Some(throughput),
                    Some("TFLOP/s"),
                )
            }
        }
        Command::Fusion {
            kind,
            k,
            intermediate,
            query_heads,
            kv_heads,
            head_dim,
            cache_capacity,
            iterations,
            warmup,
            rotate,
            rows,
            matmul_backend,
        } => {
            require_iterations(iterations)?;
            context.set_matmul_backend(matmul_backend.into());
            if let Some(rows) = rows {
                if matmul_backend != MatmulBackendArgument::Auto
                    || !matches!(
                        kind,
                        FusionKind::Qkv
                            | FusionKind::GateUp
                            | FusionKind::QkvRms
                            | FusionKind::GateUpAddRms
                    )
                {
                    return Err(CliError::InvalidArguments(
                        "fusion --rows requires auto QKV or gate-up GEMV".into(),
                    ));
                }
                if matches!(kind, FusionKind::QkvRms | FusionKind::GateUpAddRms) {
                    context.set_fused_norm_matvec_rows(rows, rows)?;
                } else {
                    context.set_auto_matvec_rows(4, rows, rows, 0)?;
                }
            }
            run_fusion_benchmark(
                &context,
                kind,
                FusionDimensions {
                    k,
                    intermediate,
                    query_heads,
                    kv_heads,
                    head_dim,
                    cache_capacity,
                },
                iterations,
                warmup,
                rotate,
                rows,
            )?
        }
        Command::Attention {
            kind,
            tokens,
            length,
            query_heads,
            kv_heads,
            head_dim,
            iterations,
            warmup,
        } => run_attention_benchmark(
            &context,
            kind,
            AttentionDimensions {
                tokens,
                length,
                query_heads,
                kv_heads,
                head_dim,
            },
            iterations,
            warmup,
        )?,
        Command::Block {
            model,
            tokens,
            iterations,
            warmup,
        } => {
            require_iterations(iterations)?;
            let mut model = Qwen3Model::load(&model, &context)?;
            model.set_attention_kind(AttentionKind::Tiled);
            let token_ids = vec![1; tokens];
            for _ in 0..warmup {
                let _ = model.run_first_block(&token_ids)?;
            }
            let samples = measure(iterations, || model.run_first_block(&token_ids).map(|_| ()))?;
            report("qwen3_block".into(), &context, samples, None, None, None)
        }
        Command::Model {
            model,
            prompt,
            generate,
            iterations,
            warmup,
            fuse_qkv,
            fuse_gate_up,
            fuse_add_rms_norm,
            fuse_qk_rope_cache,
            qkv_rms_rows,
            gate_up_add_rms_rows,
            gemv_config,
            matmul_backend,
            profile_kernels,
        } => {
            if prompt == 0 || generate == 0 || iterations == 0 {
                return Err(CliError::InvalidArguments(
                    "model prompt, generate, and iterations must be greater than zero".into(),
                ));
            }
            if gemv_config.is_some() && (qkv_rms_rows.is_some() || gate_up_add_rms_rows.is_some()) {
                return Err(CliError::InvalidArguments(
                    "--gemv-config cannot be combined with individual fused norm row overrides"
                        .into(),
                ));
            }
            if gemv_config.is_some()
                && (matmul_backend != MatmulBackendArgument::Auto
                    || context.device_name() != "Apple M4 Pro")
            {
                return Err(CliError::InvalidArguments(
                    "--gemv-config requires auto matmul on Apple M4 Pro".into(),
                ));
            }
            context.set_matmul_backend(matmul_backend.into());
            let mut model = Qwen3Model::load(&model, &context)?;
            if let Some(config) = gemv_config {
                context.set_decode_gemv_config(config.into());
            }
            if qkv_rms_rows.is_some() || gate_up_add_rms_rows.is_some() {
                if matmul_backend != MatmulBackendArgument::Auto {
                    return Err(CliError::InvalidArguments(
                        "fused norm row overrides require --matmul-backend auto".into(),
                    ));
                }
                context.set_fused_norm_matvec_rows(
                    qkv_rms_rows.unwrap_or(2),
                    gate_up_add_rms_rows.unwrap_or(2),
                )?;
            }
            let fusion_options = FusionOptions {
                qkv: fuse_qkv,
                gate_up: fuse_gate_up,
                add_rms_norm: fuse_add_rms_norm,
                qk_rope_cache: fuse_qk_rope_cache,
            };
            model.set_fusion_options(fusion_options);
            let mut cache = KvCache::new(&context, model.config(), prompt + generate)?;
            let token_ids = vec![1; prompt];
            for _ in 0..warmup {
                let _ =
                    run_model_iteration(&context, &model, &token_ids, generate, &mut cache, false)?;
            }
            if profile_kernels {
                context.set_kernel_profiling(true)?;
            }
            let allocated_before_measurement = context.allocated_bytes();
            let mut prefill_samples = Vec::with_capacity(iterations);
            let mut decode_samples = Vec::with_capacity(iterations);
            let mut prefill_kernels = Vec::new();
            let mut decode_kernels = Vec::new();
            for _ in 0..iterations {
                let (prefill, decode, prefill_profile, decode_profile) = run_model_iteration(
                    &context,
                    &model,
                    &token_ids,
                    generate,
                    &mut cache,
                    profile_kernels,
                )?;
                prefill_samples.push(prefill);
                decode_samples.push(decode);
                prefill_kernels.extend(prefill_profile);
                decode_kernels.extend(decode_profile);
            }
            context.set_kernel_profiling(false)?;
            let kernel_profile = profile_kernels.then(|| KernelProfileReport {
                diagnostic_only: true,
                prefill: kernel_phase_profile(&prefill_samples, prefill_kernels),
                decode: kernel_phase_profile(&decode_samples, decode_kernels),
            });
            let mut report = model_report(
                &context,
                fusion_options,
                allocated_before_measurement,
                prompt,
                generate,
                prefill_samples,
                decode_samples,
                kernel_profile,
            );
            report.decode_gemv_config = context.decode_gemv_config().map(DecodeGemvConfig::name);
            report
        }
    };
    match arguments.format {
        Format::Json => println!("{}", serde_json::to_string_pretty(&report)?),
        Format::Table => {
            println!("benchmark: {}", report.benchmark);
            println!("device: {}", report.device);
            println!("matmul backend: {}", report.matmul_backend);
            println!(
                "mean: {:.3} ms, median: {:.3} ms, p95: {:.3} ms",
                report.mean_ms, report.median_ms, report.p95_ms
            );
            if let (Some(mean), Some(median), Some(p95)) =
                (report.gpu_mean_ms, report.gpu_median_ms, report.gpu_p95_ms)
            {
                println!("GPU: {mean:.3} ms mean, {median:.3} ms median, {p95:.3} ms p95");
            }
            if let (Some(value), Some(unit)) = (report.throughput, report.throughput_unit) {
                println!("throughput: {value:.3} {unit}");
            }
            if let Some(prefill) = &report.prefill {
                println!(
                    "prefill: {:.3} ms wall, {:.3} ms GPU, {:.3} tokens/s",
                    prefill.wall_mean_ms, prefill.gpu_mean_ms, prefill.tokens_per_second
                );
            }
            if let Some(decode) = &report.decode {
                println!(
                    "decode: {:.3} ms wall, {:.3} ms GPU, {:.3} tokens/s",
                    decode.wall_mean_ms, decode.gpu_mean_ms, decode.tokens_per_second
                );
            }
            if let Some(profile) = &report.kernel_profile {
                println!("kernel profiling uses separate compute passes; timings are diagnostic");
                print_kernel_phase("prefill", &profile.prefill);
                print_kernel_phase("decode", &profile.decode);
            }
            if let Some(comparison) = &report.comparison {
                println!(
                    "fusion speedup: {:.3}x GPU, {:.3}x wall",
                    comparison.gpu_speedup, comparison.wall_speedup
                );
            }
            if let Some(comparison) = &report.attention_comparison {
                print_attention_variant(&comparison.reference);
                print_attention_variant(&comparison.tiled);
                if let Some(prefill) = &comparison.flash_prefill {
                    print_attention_variant(prefill);
                }
                if let Some(decode) = &comparison.decode_split_kv {
                    print_attention_variant(decode);
                }
                if let Some(decode) = &comparison.flash_decode {
                    print_attention_variant(decode);
                }
            }
            println!("Metal allocated: {} bytes", report.allocated_bytes);
        }
    }
    Ok(())
}

fn measure<E>(
    iterations: usize,
    mut operation: impl FnMut() -> Result<(), E>,
) -> Result<Vec<Duration>, E> {
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        operation()?;
        samples.push(started.elapsed());
    }
    Ok(samples)
}

fn measure_dispatch<E>(
    iterations: usize,
    mut operation: impl FnMut() -> Result<DispatchStats, E>,
) -> Result<(Vec<Duration>, Vec<Duration>), E> {
    let mut wall_samples = Vec::with_capacity(iterations);
    let mut gpu_samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        let stats = operation()?;
        wall_samples.push(started.elapsed());
        gpu_samples.push(stats.gpu_time);
    }
    Ok((wall_samples, gpu_samples))
}

fn dispatch_matmul(
    context: &MetalContext,
    input: &metal_infer_core::Tensor,
    weight: &metal_infer_core::Tensor,
) -> Result<DispatchStats, metal_infer_core::CoreError> {
    let mut batch = context.begin_batch()?;
    let _output = batch.matmul(input, weight)?;
    batch.finish()
}

#[allow(clippy::too_many_arguments)]
fn run_rotated_matvec(
    context: &MetalContext,
    m: usize,
    n: usize,
    k: usize,
    copies: usize,
    iterations: usize,
    warmup: usize,
    rows: Option<usize>,
    split_k: Option<usize>,
    half8: bool,
) -> Result<Report, CliError> {
    if m != 1 || n == 0 || k == 0 || copies == 0 {
        return Err(CliError::InvalidArguments(
            "--rotate requires m=1 and positive n, k, and copy count".into(),
        ));
    }
    let elements = n
        .checked_mul(k)
        .ok_or_else(|| CliError::InvalidArguments("matrix element count overflow".into()))?;
    let weight_bytes = elements
        .checked_mul(2)
        .ok_or_else(|| CliError::InvalidArguments("matrix byte count overflow".into()))?;
    let working_set = weight_bytes
        .checked_mul(copies)
        .ok_or_else(|| CliError::InvalidArguments("rotated working set overflow".into()))?;
    let input = context.tensor_f16(&vec![0.01; k], &[1, k])?;
    let values = vec![0.02; elements];
    let weights = (0..copies)
        .map(|_| context.tensor_f16(&values, &[n, k]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut dispatch = || {
        let mut batch = context.begin_batch()?;
        let outputs = weights
            .iter()
            .map(|weight| batch.matmul(&input, weight))
            .collect::<Result<Vec<_>, _>>()?;
        let stats = batch.finish()?;
        drop(outputs);
        Ok::<DispatchStats, metal_infer_core::CoreError>(stats)
    };
    for _ in 0..warmup {
        dispatch()?;
    }
    let (mut wall, mut gpu) = measure_dispatch(iterations, &mut dispatch)?;
    let divisor = copies as f64;
    for sample in &mut wall {
        *sample = Duration::from_secs_f64(sample.as_secs_f64() / divisor);
    }
    for sample in &mut gpu {
        *sample = Duration::from_secs_f64(sample.as_secs_f64() / divisor);
    }
    let bandwidth = weight_bytes as f64 / 1.0e9 / mean_seconds(&gpu);
    Ok(report(
        format!(
            "matvec_f16[1,{n},{k},rotate={copies},rows={rows:?},split_k={split_k:?},half8={half8},working_set={working_set}]"
        ),
        context,
        wall,
        Some(gpu),
        Some(bandwidth),
        Some("GB/s GPU weight reads"),
    ))
}

fn dispatch_attention(
    context: &MetalContext,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    config: AttentionConfig,
    kind: AttentionKind,
) -> Result<DispatchStats, metal_infer_core::CoreError> {
    let mut batch = context.begin_batch()?;
    let _output = batch.attention(query, key, value, config, kind)?;
    batch.finish()
}

fn run_attention_benchmark(
    context: &MetalContext,
    selection: AttentionBenchmarkKind,
    dimensions: AttentionDimensions,
    iterations: usize,
    warmup: usize,
) -> Result<Report, CliError> {
    require_iterations(iterations)?;
    let AttentionDimensions {
        tokens,
        length,
        query_heads,
        kv_heads,
        head_dim,
    } = dimensions;
    if tokens == 0
        || length == 0
        || tokens > length
        || query_heads == 0
        || kv_heads == 0
        || head_dim == 0
    {
        return Err(CliError::InvalidArguments(
            "attention dimensions must be non-zero and tokens must not exceed length".into(),
        ));
    }
    if !query_heads.is_multiple_of(kv_heads) {
        return Err(CliError::InvalidArguments(
            "attention query-heads must be divisible by kv-heads".into(),
        ));
    }
    if matches!(
        selection,
        AttentionBenchmarkKind::DecodeSplitKv | AttentionBenchmarkKind::FlashDecode
    ) && tokens != 1
    {
        return Err(CliError::InvalidArguments(
            "decode attention requires --tokens 1".into(),
        ));
    }
    if matches!(selection, AttentionBenchmarkKind::FlashDecode) && query_heads / kv_heads != 2 {
        return Err(CliError::InvalidArguments(
            "flash decode requires two query heads per KV head".into(),
        ));
    }

    let query_values: Vec<f32> = (0..tokens * query_heads * head_dim)
        .map(|index| (index % 43) as f32 / 43.0 - 0.5)
        .collect();
    let key_values: Vec<f32> = (0..length * kv_heads * head_dim)
        .map(|index| (index % 37) as f32 / 37.0 - 0.25)
        .collect();
    let value_values: Vec<f32> = (0..length * kv_heads * head_dim)
        .map(|index| (index % 29) as f32 / 29.0)
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
    let case = AttentionCase {
        context,
        query: &query,
        key: &key,
        value: &value,
        config,
    };
    let benchmark = format!("attention_f16[tokens={tokens},length={length}]");

    if !matches!(selection, AttentionBenchmarkKind::Compare) {
        let kind = attention_kind(selection);
        let (samples, gpu_samples) = measure_attention_variant(case, kind, iterations, warmup)?;
        return Ok(report(
            format!("{benchmark}[{}]", attention_kind_name(kind)),
            context,
            samples,
            Some(gpu_samples),
            None,
            None,
        ));
    }

    let reference_output = attention_output(case, AttentionKind::Reference)?;
    let reference = attention_variant_report(
        case,
        AttentionKind::Reference,
        &reference_output,
        iterations,
        warmup,
    )?;
    let tiled = attention_variant_report(
        case,
        AttentionKind::Tiled,
        &reference_output,
        iterations,
        warmup,
    )?;
    let flash_prefill = if tokens > 1 {
        Some(attention_variant_report(
            case,
            AttentionKind::FlashPrefill,
            &reference_output,
            iterations,
            warmup,
        )?)
    } else {
        None
    };
    let decode_split_kv = if tokens == 1 {
        Some(attention_variant_report(
            case,
            AttentionKind::DecodeSplitKv,
            &reference_output,
            iterations,
            warmup,
        )?)
    } else {
        None
    };
    let flash_decode = if tokens == 1 && query_heads / kv_heads == 2 {
        Some(attention_variant_report(
            case,
            AttentionKind::FlashDecode,
            &reference_output,
            iterations,
            warmup,
        )?)
    } else {
        None
    };
    let selected = if length >= 256 {
        flash_decode.as_ref().or(decode_split_kv.as_ref())
    } else {
        decode_split_kv.as_ref()
    }
    .unwrap_or_else(|| {
        if tokens >= 32 {
            flash_prefill.as_ref().unwrap_or(&tiled)
        } else {
            &tiled
        }
    });
    Ok(Report {
        benchmark,
        device: context.device_name(),
        iterations,
        mean_ms: selected.timing.wall_mean_ms,
        median_ms: selected.timing.wall_median_ms,
        p95_ms: selected.timing.wall_p95_ms,
        gpu_mean_ms: Some(selected.timing.gpu_mean_ms),
        gpu_median_ms: Some(selected.timing.gpu_median_ms),
        gpu_p95_ms: Some(selected.timing.gpu_p95_ms),
        throughput: None,
        throughput_unit: None,
        allocated_bytes: context.allocated_bytes(),
        matmul_backend: context.matmul_backend().name(),
        decode_gemv_config: None,
        allocation_growth_bytes: None,
        fusions: None,
        comparison: None,
        attention_comparison: Some(AttentionComparison {
            reference,
            tiled,
            flash_prefill,
            decode_split_kv,
            flash_decode,
        }),
        prefill: None,
        decode: None,
        kernel_profile: None,
    })
}

fn attention_variant_report(
    case: AttentionCase<'_>,
    kind: AttentionKind,
    reference: &[f32],
    iterations: usize,
    warmup: usize,
) -> Result<AttentionVariantReport, CliError> {
    let output = attention_output(case, kind)?;
    let (wall, gpu) = measure_attention_variant(case, kind, iterations, warmup)?;
    Ok(AttentionVariantReport {
        kind: attention_kind_name(kind),
        timing: timing_report(wall, gpu),
        max_abs_error: max_abs_error(&output, reference),
    })
}

fn measure_attention_variant(
    case: AttentionCase<'_>,
    kind: AttentionKind,
    iterations: usize,
    warmup: usize,
) -> Result<(Vec<Duration>, Vec<Duration>), metal_infer_core::CoreError> {
    for _ in 0..warmup {
        let _ = dispatch_attention(
            case.context,
            case.query,
            case.key,
            case.value,
            case.config,
            kind,
        )?;
    }
    measure_dispatch(iterations, || {
        dispatch_attention(
            case.context,
            case.query,
            case.key,
            case.value,
            case.config,
            kind,
        )
    })
}

fn attention_output(
    case: AttentionCase<'_>,
    kind: AttentionKind,
) -> Result<Vec<f32>, metal_infer_core::CoreError> {
    case.context
        .attention(case.query, case.key, case.value, case.config, kind)?
        .to_f32_vec()
}

const fn attention_kind(selection: AttentionBenchmarkKind) -> AttentionKind {
    match selection {
        AttentionBenchmarkKind::Reference => AttentionKind::Reference,
        AttentionBenchmarkKind::Tiled => AttentionKind::Tiled,
        AttentionBenchmarkKind::FlashPrefill => AttentionKind::FlashPrefill,
        AttentionBenchmarkKind::DecodeSplitKv => AttentionKind::DecodeSplitKv,
        AttentionBenchmarkKind::FlashDecode => AttentionKind::FlashDecode,
        AttentionBenchmarkKind::Compare => AttentionKind::Reference,
    }
}

const fn attention_kind_name(kind: AttentionKind) -> &'static str {
    match kind {
        AttentionKind::Reference => "reference",
        AttentionKind::Tiled => "tiled",
        AttentionKind::FlashPrefill => "flash-prefill",
        AttentionKind::DecodeSplitKv => "decode-split-kv",
        AttentionKind::FlashDecode => "flash-decode",
    }
}

fn max_abs_error(
    actual: &[f32],
    expected: &[f32],
) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| f64::from((actual - expected).abs()))
        .fold(0.0, f64::max)
}

fn print_attention_variant(variant: &AttentionVariantReport) {
    println!(
        "{}: {:.3} ms wall, {:.3} ms GPU, max abs error {:.6}",
        variant.kind,
        variant.timing.wall_mean_ms,
        variant.timing.gpu_mean_ms,
        variant.max_abs_error
    );
}

fn run_fusion_benchmark(
    context: &MetalContext,
    kind: FusionKind,
    dimensions: FusionDimensions,
    iterations: usize,
    warmup: usize,
    rotate: Option<usize>,
    rows: Option<usize>,
) -> Result<Report, CliError> {
    let FusionDimensions {
        k,
        intermediate,
        query_heads,
        kv_heads,
        head_dim,
        cache_capacity,
    } = dimensions;
    if k == 0
        || intermediate == 0
        || query_heads == 0
        || kv_heads == 0
        || head_dim == 0
        || cache_capacity == 0
    {
        return Err(CliError::InvalidArguments(
            "fusion benchmark dimensions must be greater than zero".into(),
        ));
    }
    if let Some(copies) = rotate {
        if copies == 0
            || !matches!(
                kind,
                FusionKind::Qkv
                    | FusionKind::GateUp
                    | FusionKind::QkvRms
                    | FusionKind::GateUpAddRms
            )
        {
            return Err(CliError::InvalidArguments(
                "fusion --rotate requires a positive copy count and a projection kind".into(),
            ));
        }
        let widths = match kind {
            FusionKind::Qkv | FusionKind::QkvRms => vec![
                query_heads * head_dim,
                kv_heads * head_dim,
                kv_heads * head_dim,
            ],
            FusionKind::GateUp | FusionKind::GateUpAddRms => vec![intermediate, intermediate],
            FusionKind::AddRmsNorm | FusionKind::QkRopeCache => unreachable!(),
        };
        return run_rotated_fusion(context, kind, k, &widths, copies, iterations, warmup, rows);
    }
    match kind {
        FusionKind::QkvRms | FusionKind::GateUpAddRms => Err(CliError::InvalidArguments(
            "fused norm projection benchmarks require --rotate".into(),
        )),
        FusionKind::Qkv => {
            let input = context.tensor_f16(&vec![0.01; k], &[1, k])?;
            let query_width = query_heads * head_dim;
            let kv_width = kv_heads * head_dim;
            let query = context.tensor_f16(&vec![0.02; query_width * k], &[query_width, k])?;
            let key = context.tensor_f16(&vec![0.03; kv_width * k], &[kv_width, k])?;
            let value = context.tensor_f16(&vec![0.04; kv_width * k], &[kv_width, k])?;
            Ok(measure_fusion_pair(
                context,
                format!("qkv_f16[k={k},{}]", projection_path(k)),
                iterations,
                warmup,
                || dispatch_qkv(context, &input, &query, &key, &value, false),
                || dispatch_qkv(context, &input, &query, &key, &value, true),
            )?)
        }
        FusionKind::GateUp => {
            let input = context.tensor_f16(&vec![0.01; k], &[1, k])?;
            let gate = context.tensor_f16(&vec![0.02; intermediate * k], &[intermediate, k])?;
            let up = context.tensor_f16(&vec![0.03; intermediate * k], &[intermediate, k])?;
            Ok(measure_fusion_pair(
                context,
                format!("gate_up_f16[k={k},{}]", projection_path(k)),
                iterations,
                warmup,
                || dispatch_gate_up(context, &input, &gate, &up, false),
                || dispatch_gate_up(context, &input, &gate, &up, true),
            )?)
        }
        FusionKind::AddRmsNorm => {
            let left = context.tensor_f16(&vec![0.01; k], &[1, k])?;
            let right = context.tensor_f16(&vec![0.02; k], &[1, k])?;
            let weight = context.tensor_f16(&vec![1.0; k], &[k])?;
            Ok(measure_fusion_pair(
                context,
                format!("add_rms_norm_f16[width={k}]"),
                iterations,
                warmup,
                || dispatch_add_rms_norm(context, &left, &right, &weight, false),
                || dispatch_add_rms_norm(context, &left, &right, &weight, true),
            )?)
        }
        FusionKind::QkRopeCache => {
            if !head_dim.is_multiple_of(2) {
                return Err(CliError::InvalidArguments(
                    "QK benchmark head-dim must be even".into(),
                ));
            }
            let query = context.tensor_f16(
                &vec![0.01; query_heads * head_dim],
                &[1, query_heads, head_dim],
            )?;
            let key =
                context.tensor_f16(&vec![0.02; kv_heads * head_dim], &[1, kv_heads, head_dim])?;
            let query_weight = context.tensor_f16(&vec![1.0; head_dim], &[head_dim])?;
            let key_weight = context.tensor_f16(&vec![1.0; head_dim], &[head_dim])?;
            let key_cache = context.tensor_f16(
                &vec![0.0; cache_capacity * kv_heads * head_dim],
                &[cache_capacity, kv_heads, head_dim],
            )?;
            let offset = cache_capacity / 2;
            Ok(measure_fusion_pair(
                context,
                format!("qk_rope_cache_f16[head_dim={head_dim}]"),
                iterations,
                warmup,
                || {
                    dispatch_qk_rope_cache(
                        context,
                        &query,
                        &key,
                        &query_weight,
                        &key_weight,
                        &key_cache,
                        offset,
                        false,
                    )
                },
                || {
                    dispatch_qk_rope_cache(
                        context,
                        &query,
                        &key,
                        &query_weight,
                        &key_weight,
                        &key_cache,
                        offset,
                        true,
                    )
                },
            )?)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_rotated_fusion(
    context: &MetalContext,
    kind: FusionKind,
    k: usize,
    widths: &[usize],
    copies: usize,
    iterations: usize,
    warmup: usize,
    rows: Option<usize>,
) -> Result<Report, CliError> {
    let elements = widths
        .iter()
        .try_fold(0usize, |sum, width| {
            width
                .checked_mul(k)
                .and_then(|count| sum.checked_add(count))
        })
        .ok_or_else(|| CliError::InvalidArguments("projection size overflow".into()))?;
    let weight_bytes = elements
        .checked_mul(2)
        .ok_or_else(|| CliError::InvalidArguments("projection byte count overflow".into()))?;
    let working_set = weight_bytes
        .checked_mul(copies)
        .ok_or_else(|| CliError::InvalidArguments("rotated working set overflow".into()))?;
    let input = context.tensor_f16(&vec![0.01; k], &[1, k])?;
    let right = context.tensor_f16(&vec![0.02; k], &[1, k])?;
    let norm = context.tensor_f16(&vec![1.0; k], &[k])?;
    let mut weights = Vec::with_capacity(copies);
    for _ in 0..copies {
        let mut projections = Vec::with_capacity(widths.len());
        for (index, &width) in widths.iter().enumerate() {
            projections.push(
                context.tensor_f16(&vec![0.02 + index as f32 * 0.01; width * k], &[width, k])?,
            );
        }
        weights.push(projections);
    }
    let dispatch = |fused: bool| -> Result<DispatchStats, metal_infer_core::CoreError> {
        let mut batch = context.begin_batch()?;
        let mut outputs = Vec::with_capacity(copies * widths.len());
        for projections in &weights {
            match (kind, fused) {
                (FusionKind::QkvRms, true) => {
                    let query = projections.first().expect("QKV widths validated");
                    let key = projections.get(1).expect("QKV widths validated");
                    let value = projections.get(2).expect("QKV widths validated");
                    let (a, b, c) =
                        batch.rms_norm_matmul3(&input, &norm, query, key, value, 1.0e-6)?;
                    outputs.extend([a, b, c]);
                }
                (FusionKind::QkvRms, false) => {
                    let normalized = batch.rms_norm(&input, &norm, 1.0e-6)?;
                    let query = projections.first().expect("QKV widths validated");
                    let key = projections.get(1).expect("QKV widths validated");
                    let value = projections.get(2).expect("QKV widths validated");
                    let (a, b, c) = batch.matmul3(&normalized, query, key, value)?;
                    outputs.extend([a, b, c]);
                }
                (FusionKind::GateUpAddRms, true) => {
                    let gate = projections.first().expect("gate-up widths validated");
                    let up = projections.get(1).expect("gate-up widths validated");
                    let (residual, a, b) =
                        batch.add_rms_norm_matmul2(&input, &right, &norm, gate, up, 1.0e-6)?;
                    outputs.extend([residual, a, b]);
                }
                (FusionKind::GateUpAddRms, false) => {
                    let (residual, normalized) =
                        batch.add_rms_norm(&input, &right, &norm, 1.0e-6)?;
                    let gate = projections.first().expect("gate-up widths validated");
                    let up = projections.get(1).expect("gate-up widths validated");
                    let (a, b) = batch.matmul2(&normalized, gate, up)?;
                    outputs.extend([residual, a, b]);
                }
                (FusionKind::Qkv, true) => {
                    let (q, kv) = projections.split_first().expect("QKV widths validated");
                    let key = kv.first().expect("QKV widths validated");
                    let value = kv.get(1).expect("QKV widths validated");
                    let (a, b, c) = batch.matmul3(&input, q, key, value)?;
                    outputs.extend([a, b, c]);
                }
                (FusionKind::GateUp, true) => {
                    let gate = projections.first().expect("gate-up widths validated");
                    let up = projections.get(1).expect("gate-up widths validated");
                    let (a, b) = batch.matmul2(&input, gate, up)?;
                    outputs.extend([a, b]);
                }
                _ => {
                    for projection in projections {
                        outputs.push(batch.matmul(&input, projection)?);
                    }
                }
            }
        }
        let stats = batch.finish()?;
        drop(outputs);
        Ok(stats)
    };
    let mut report = measure_fusion_pair(
        context,
        format!("{kind:?}_f16[k={k},rotate={copies},rows={rows:?},working_set={working_set}]"),
        iterations,
        warmup,
        || dispatch(false),
        || dispatch(true),
    )?;
    let divisor = copies as f64;
    if let Some(comparison) = report.comparison.as_mut() {
        for timing in [&mut comparison.unfused, &mut comparison.fused] {
            timing.wall_mean_ms /= divisor;
            timing.wall_median_ms /= divisor;
            timing.wall_p95_ms /= divisor;
            timing.gpu_mean_ms /= divisor;
            timing.gpu_median_ms /= divisor;
            timing.gpu_p95_ms /= divisor;
        }
        report.mean_ms = comparison.fused.wall_mean_ms;
        report.median_ms = comparison.fused.wall_median_ms;
        report.p95_ms = comparison.fused.wall_p95_ms;
        report.gpu_mean_ms = Some(comparison.fused.gpu_mean_ms);
        report.gpu_median_ms = Some(comparison.fused.gpu_median_ms);
        report.gpu_p95_ms = Some(comparison.fused.gpu_p95_ms);
        report.throughput = Some(weight_bytes as f64 / (comparison.fused.gpu_mean_ms * 1.0e6));
        report.throughput_unit = Some("GB/s GPU weight reads");
    }
    Ok(report)
}

fn projection_path(k: usize) -> &'static str {
    if k.is_multiple_of(4) {
        "half4"
    } else {
        "scalar"
    }
}

fn measure_fusion_pair<E>(
    context: &MetalContext,
    benchmark: String,
    iterations: usize,
    warmup: usize,
    mut unfused: impl FnMut() -> Result<DispatchStats, E>,
    mut fused: impl FnMut() -> Result<DispatchStats, E>,
) -> Result<Report, E> {
    for _ in 0..warmup {
        let _ = unfused()?;
        let _ = fused()?;
    }
    let allocated_before_measurement = context.allocated_bytes();
    let mut unfused_wall = Vec::with_capacity(iterations);
    let mut unfused_gpu = Vec::with_capacity(iterations);
    let mut fused_wall = Vec::with_capacity(iterations);
    let mut fused_gpu = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        if iteration.is_multiple_of(2) {
            push_dispatch_sample(&mut unfused, &mut unfused_wall, &mut unfused_gpu)?;
            push_dispatch_sample(&mut fused, &mut fused_wall, &mut fused_gpu)?;
        } else {
            push_dispatch_sample(&mut fused, &mut fused_wall, &mut fused_gpu)?;
            push_dispatch_sample(&mut unfused, &mut unfused_wall, &mut unfused_gpu)?;
        }
    }
    let unfused = timing_report(unfused_wall, unfused_gpu);
    let fused = timing_report(fused_wall, fused_gpu);
    let comparison = FusionComparison {
        gpu_speedup: unfused.gpu_mean_ms / fused.gpu_mean_ms,
        wall_speedup: unfused.wall_mean_ms / fused.wall_mean_ms,
        unfused,
        fused,
    };
    Ok(Report {
        benchmark,
        device: context.device_name(),
        iterations,
        mean_ms: comparison.fused.wall_mean_ms,
        median_ms: comparison.fused.wall_median_ms,
        p95_ms: comparison.fused.wall_p95_ms,
        gpu_mean_ms: Some(comparison.fused.gpu_mean_ms),
        gpu_median_ms: Some(comparison.fused.gpu_median_ms),
        gpu_p95_ms: Some(comparison.fused.gpu_p95_ms),
        throughput: None,
        throughput_unit: None,
        allocated_bytes: context.allocated_bytes(),
        matmul_backend: context.matmul_backend().name(),
        decode_gemv_config: None,
        allocation_growth_bytes: Some(
            context
                .allocated_bytes()
                .saturating_sub(allocated_before_measurement),
        ),
        fusions: None,
        comparison: Some(comparison),
        attention_comparison: None,
        prefill: None,
        decode: None,
        kernel_profile: None,
    })
}

fn push_dispatch_sample<E>(
    operation: &mut impl FnMut() -> Result<DispatchStats, E>,
    wall_samples: &mut Vec<Duration>,
    gpu_samples: &mut Vec<Duration>,
) -> Result<(), E> {
    let started = Instant::now();
    let stats = operation()?;
    wall_samples.push(started.elapsed());
    gpu_samples.push(stats.gpu_time);
    Ok(())
}

fn timing_report(
    wall: Vec<Duration>,
    gpu: Vec<Duration>,
) -> TimingReport {
    let wall = duration_stats(wall);
    let gpu = duration_stats(gpu);
    TimingReport {
        wall_mean_ms: wall.mean_ms,
        wall_median_ms: wall.median_ms,
        wall_p95_ms: wall.p95_ms,
        gpu_mean_ms: gpu.mean_ms,
        gpu_median_ms: gpu.median_ms,
        gpu_p95_ms: gpu.p95_ms,
    }
}

fn dispatch_qkv(
    context: &MetalContext,
    input: &metal_infer_core::Tensor,
    query: &metal_infer_core::Tensor,
    key: &metal_infer_core::Tensor,
    value: &metal_infer_core::Tensor,
    fused: bool,
) -> Result<DispatchStats, metal_infer_core::CoreError> {
    let mut batch = context.begin_batch()?;
    if fused {
        let _ = batch.matmul3(input, query, key, value)?;
    } else {
        let _ = batch.matmul(input, query)?;
        let _ = batch.matmul(input, key)?;
        let _ = batch.matmul(input, value)?;
    }
    batch.finish()
}

fn dispatch_gate_up(
    context: &MetalContext,
    input: &metal_infer_core::Tensor,
    gate: &metal_infer_core::Tensor,
    up: &metal_infer_core::Tensor,
    fused: bool,
) -> Result<DispatchStats, metal_infer_core::CoreError> {
    let mut batch = context.begin_batch()?;
    if fused {
        let _ = batch.matmul2(input, gate, up)?;
    } else {
        let _ = batch.matmul(input, gate)?;
        let _ = batch.matmul(input, up)?;
    }
    batch.finish()
}

fn dispatch_add_rms_norm(
    context: &MetalContext,
    left: &metal_infer_core::Tensor,
    right: &metal_infer_core::Tensor,
    weight: &metal_infer_core::Tensor,
    fused: bool,
) -> Result<DispatchStats, metal_infer_core::CoreError> {
    let mut batch = context.begin_batch()?;
    if fused {
        let _ = batch.add_rms_norm(left, right, weight, 1.0e-6)?;
    } else {
        let residual = batch.add(left, right)?;
        let _ = batch.rms_norm(&residual, weight, 1.0e-6)?;
    }
    batch.finish()
}

#[allow(clippy::too_many_arguments)]
fn dispatch_qk_rope_cache(
    context: &MetalContext,
    query: &metal_infer_core::Tensor,
    key: &metal_infer_core::Tensor,
    query_weight: &metal_infer_core::Tensor,
    key_weight: &metal_infer_core::Tensor,
    key_cache: &metal_infer_core::Tensor,
    offset: usize,
    fused: bool,
) -> Result<DispatchStats, metal_infer_core::CoreError> {
    let mut batch = context.begin_batch()?;
    if fused {
        let _ = batch.qk_norm_rope_cache(
            query,
            key,
            query_weight,
            key_weight,
            key_cache,
            QkNormRopeCacheConfig {
                offset,
                theta: 10_000.0,
                epsilon: 1.0e-6,
            },
        )?;
    } else {
        let query = batch.rms_norm(query, query_weight, 1.0e-6)?;
        let _ = batch.rope(&query, offset, 10_000.0)?;
        let key = batch.rms_norm(key, key_weight, 1.0e-6)?;
        let key = batch.rope(&key, offset, 10_000.0)?;
        batch.copy_into_cache(&key, key_cache, offset)?;
    }
    batch.finish()
}

fn report(
    benchmark: String,
    context: &MetalContext,
    mut samples: Vec<Duration>,
    gpu_samples: Option<Vec<Duration>>,
    throughput: Option<f64>,
    throughput_unit: Option<&'static str>,
) -> Report {
    samples.sort();
    let mean = mean_seconds(&samples);
    let median = samples
        .get(samples.len() / 2)
        .map_or(0.0, Duration::as_secs_f64);
    let p95_index = (samples.len().saturating_sub(1) as f64 * 0.95).round() as usize;
    let p95 = samples.get(p95_index).map_or(0.0, Duration::as_secs_f64);
    let gpu = gpu_samples.map(duration_stats);
    Report {
        benchmark,
        device: context.device_name(),
        iterations: samples.len(),
        mean_ms: mean * 1000.0,
        median_ms: median * 1000.0,
        p95_ms: p95 * 1000.0,
        gpu_mean_ms: gpu.as_ref().map(|timing| timing.mean_ms),
        gpu_median_ms: gpu.as_ref().map(|timing| timing.median_ms),
        gpu_p95_ms: gpu.as_ref().map(|timing| timing.p95_ms),
        throughput,
        throughput_unit,
        allocated_bytes: context.allocated_bytes(),
        matmul_backend: context.matmul_backend().name(),
        decode_gemv_config: None,
        allocation_growth_bytes: None,
        fusions: None,
        comparison: None,
        attention_comparison: None,
        prefill: None,
        decode: None,
        kernel_profile: None,
    }
}

fn run_model_iteration(
    context: &MetalContext,
    model: &Qwen3Model,
    prompt: &[u32],
    decode_tokens: usize,
    cache: &mut KvCache,
    profile_kernels: bool,
) -> Result<
    (
        PhaseSample,
        PhaseSample,
        Vec<KernelDispatchProfile>,
        Vec<KernelDispatchProfile>,
    ),
    CliError,
> {
    let started = Instant::now();
    let (_, prefill_stats) = model.prefill_with_stats(prompt, cache)?;
    let prefill = PhaseSample {
        wall: started.elapsed(),
        gpu: prefill_stats.gpu_time,
    };
    let prefill_profile = if profile_kernels {
        context.take_kernel_profiles()
    } else {
        Vec::new()
    };
    let started = Instant::now();
    let mut decode_gpu = Duration::ZERO;
    for _ in 0..decode_tokens {
        let (_, stats) = model.decode_with_stats(1, cache)?;
        decode_gpu += stats.gpu_time;
    }
    let decode_profile = if profile_kernels {
        context.take_kernel_profiles()
    } else {
        Vec::new()
    };
    Ok((
        prefill,
        PhaseSample {
            wall: started.elapsed(),
            gpu: decode_gpu,
        },
        prefill_profile,
        decode_profile,
    ))
}

fn kernel_phase_profile(
    samples: &[PhaseSample],
    dispatches: Vec<KernelDispatchProfile>,
) -> KernelPhaseProfile {
    let gpu_ms = samples
        .iter()
        .map(|sample| sample.gpu.as_secs_f64())
        .sum::<f64>()
        * 1000.0;
    let mut totals: HashMap<String, (usize, f64)> = HashMap::new();
    for dispatch in dispatches {
        let entry = totals.entry(dispatch.kernel).or_default();
        entry.0 += 1;
        entry.1 += dispatch.gpu_time.as_secs_f64() * 1000.0;
    }
    let attributed_ms = totals.values().map(|(_, total)| *total).sum::<f64>();
    let mut kernels: Vec<_> = totals
        .into_iter()
        .map(|(kernel, (calls, total_gpu_ms))| KernelProfileRow {
            kernel,
            calls,
            total_gpu_ms,
            mean_gpu_ms: total_gpu_ms / calls as f64,
        })
        .collect();
    kernels.sort_by(|a, b| {
        b.total_gpu_ms
            .total_cmp(&a.total_gpu_ms)
            .then_with(|| a.kernel.cmp(&b.kernel))
    });
    KernelPhaseProfile {
        gpu_ms,
        attributed_ms,
        unattributed_ms: gpu_ms - attributed_ms,
        kernels,
    }
}

fn print_kernel_phase(
    name: &str,
    profile: &KernelPhaseProfile,
) {
    println!(
        "{name} kernels: {:.3} ms attributed / {:.3} ms GPU ({:.3} ms unattributed)",
        profile.attributed_ms, profile.gpu_ms, profile.unattributed_ms
    );
    for row in &profile.kernels {
        println!(
            "  {:<32} {:>7} calls  {:>9.3} ms total  {:>8.4} ms/call",
            row.kernel, row.calls, row.total_gpu_ms, row.mean_gpu_ms
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn model_report(
    context: &MetalContext,
    fusion_options: FusionOptions,
    allocated_before_measurement: usize,
    prompt_tokens: usize,
    decode_tokens: usize,
    prefill_samples: Vec<PhaseSample>,
    decode_samples: Vec<PhaseSample>,
    kernel_profile: Option<KernelProfileReport>,
) -> Report {
    let prefill = phase_report(prompt_tokens, &prefill_samples);
    let decode = phase_report(decode_tokens, &decode_samples);
    Report {
        benchmark: "qwen3_model".into(),
        device: context.device_name(),
        iterations: prefill_samples.len(),
        mean_ms: prefill.wall_mean_ms + decode.wall_mean_ms,
        median_ms: prefill.wall_median_ms + decode.wall_median_ms,
        p95_ms: prefill.wall_p95_ms + decode.wall_p95_ms,
        gpu_mean_ms: Some(prefill.gpu_mean_ms + decode.gpu_mean_ms),
        gpu_median_ms: Some(prefill.gpu_median_ms + decode.gpu_median_ms),
        gpu_p95_ms: Some(prefill.gpu_p95_ms + decode.gpu_p95_ms),
        throughput: None,
        throughput_unit: None,
        allocated_bytes: context.allocated_bytes(),
        matmul_backend: context.matmul_backend().name(),
        decode_gemv_config: None,
        allocation_growth_bytes: Some(
            context
                .allocated_bytes()
                .saturating_sub(allocated_before_measurement),
        ),
        fusions: Some(fusion_options.into()),
        comparison: None,
        attention_comparison: None,
        prefill: Some(prefill),
        decode: Some(decode),
        kernel_profile,
    }
}

fn duration_stats(mut samples: Vec<Duration>) -> DurationStats {
    samples.sort();
    let mean = mean_seconds(&samples);
    let median = samples
        .get(samples.len() / 2)
        .map_or(0.0, Duration::as_secs_f64);
    let p95_index = (samples.len().saturating_sub(1) as f64 * 0.95).round() as usize;
    let p95 = samples.get(p95_index).map_or(0.0, Duration::as_secs_f64);
    DurationStats {
        mean_ms: mean * 1000.0,
        median_ms: median * 1000.0,
        p95_ms: p95 * 1000.0,
    }
}

fn phase_report(
    tokens: usize,
    samples: &[PhaseSample],
) -> PhaseReport {
    let wall = duration_stats(samples.iter().map(|sample| sample.wall).collect());
    let gpu = duration_stats(samples.iter().map(|sample| sample.gpu).collect());
    PhaseReport {
        tokens,
        wall_mean_ms: wall.mean_ms,
        wall_median_ms: wall.median_ms,
        wall_p95_ms: wall.p95_ms,
        gpu_mean_ms: gpu.mean_ms,
        gpu_median_ms: gpu.median_ms,
        gpu_p95_ms: gpu.p95_ms,
        tokens_per_second: tokens as f64 / (wall.mean_ms / 1000.0),
    }
}

fn mean_seconds(samples: &[Duration]) -> f64 {
    samples.iter().map(Duration::as_secs_f64).sum::<f64>() / samples.len() as f64
}

fn require_iterations(iterations: usize) -> Result<(), CliError> {
    if iterations == 0 {
        Err(CliError::InvalidArguments(
            "iterations must be greater than zero".into(),
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{AttentionBenchmarkKind, AttentionKind, attention_kind, max_abs_error};

    #[test]
    fn attention_selection_maps_to_explicit_kernel() {
        assert_eq!(
            attention_kind(AttentionBenchmarkKind::Reference),
            AttentionKind::Reference,
            "reference selection mismatch"
        );
        assert_eq!(
            attention_kind(AttentionBenchmarkKind::Tiled),
            AttentionKind::Tiled,
            "tiled selection mismatch"
        );
        assert_eq!(
            attention_kind(AttentionBenchmarkKind::FlashPrefill),
            AttentionKind::FlashPrefill,
            "flash-prefill selection mismatch"
        );
        assert_eq!(
            attention_kind(AttentionBenchmarkKind::DecodeSplitKv),
            AttentionKind::DecodeSplitKv,
            "split-KV selection mismatch"
        );
        assert_eq!(
            attention_kind(AttentionBenchmarkKind::FlashDecode),
            AttentionKind::FlashDecode,
            "flash-decode selection mismatch"
        );
    }

    #[test]
    fn maximum_absolute_error_uses_largest_difference() {
        let error = max_abs_error(&[1.0, -2.0, 4.5], &[0.5, -1.0, 4.25]);
        assert_eq!(error, 1.0, "maximum absolute error mismatch");
    }
}
