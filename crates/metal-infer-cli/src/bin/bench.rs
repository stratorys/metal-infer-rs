use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};
use metal_infer_cli::CliError;
use metal_infer_core::{AttentionConfig, AttentionKind, DispatchStats, MetalContext};
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
    },
    Attention {
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
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Table,
    Json,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum FusionKind {
    Qkv,
    GateUp,
    AddRmsNorm,
    QkRopeCache,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    allocation_growth_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fusions: Option<FusionSelection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparison: Option<FusionComparison>,
    prefill: Option<PhaseReport>,
    decode: Option<PhaseReport>,
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
        } => {
            require_iterations(iterations)?;
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
        } => {
            require_iterations(iterations)?;
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
            )?
        }
        Command::Attention {
            tokens,
            length,
            query_heads,
            kv_heads,
            head_dim,
            iterations,
            warmup,
        } => {
            require_iterations(iterations)?;
            if tokens == 0
                || length == 0
                || tokens > length
                || query_heads == 0
                || kv_heads == 0
                || head_dim == 0
            {
                return Err(CliError::InvalidArguments(
                    "attention dimensions must be non-zero and tokens must not exceed length"
                        .into(),
                ));
            }
            let query = context.tensor_f16(
                &vec![0.01; tokens * query_heads * head_dim],
                &[tokens, query_heads, head_dim],
            )?;
            let key = context.tensor_f16(
                &vec![0.02; length * kv_heads * head_dim],
                &[length, kv_heads, head_dim],
            )?;
            let value = context.tensor_f16(
                &vec![0.03; length * kv_heads * head_dim],
                &[length, kv_heads, head_dim],
            )?;
            let dispatch = || {
                dispatch_attention(
                    &context,
                    &query,
                    &key,
                    &value,
                    AttentionConfig {
                        query_heads,
                        kv_heads,
                        head_dim,
                        causal: true,
                        query_offset: length - tokens,
                    },
                )
            };
            for _ in 0..warmup {
                let _ = dispatch()?;
            }
            let (samples, gpu_samples) = measure_dispatch(iterations, dispatch)?;
            report(
                format!("attention_f16[tokens={tokens},length={length}]"),
                &context,
                samples,
                Some(gpu_samples),
                None,
                None,
            )
        }
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
        } => {
            if prompt == 0 || generate == 0 || iterations == 0 {
                return Err(CliError::InvalidArguments(
                    "model prompt, generate, and iterations must be greater than zero".into(),
                ));
            }
            let mut model = Qwen3Model::load(&model, &context)?;
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
                let _ = run_model_iteration(&model, &token_ids, generate, &mut cache)?;
            }
            let allocated_before_measurement = context.allocated_bytes();
            let mut prefill_samples = Vec::with_capacity(iterations);
            let mut decode_samples = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let (prefill, decode) =
                    run_model_iteration(&model, &token_ids, generate, &mut cache)?;
                prefill_samples.push(prefill);
                decode_samples.push(decode);
            }
            model_report(
                &context,
                fusion_options,
                allocated_before_measurement,
                prompt,
                generate,
                prefill_samples,
                decode_samples,
            )
        }
    };
    match arguments.format {
        Format::Json => println!("{}", serde_json::to_string_pretty(&report)?),
        Format::Table => {
            println!("benchmark: {}", report.benchmark);
            println!("device: {}", report.device);
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
            if let Some(comparison) = &report.comparison {
                println!(
                    "fusion speedup: {:.3}x GPU, {:.3}x wall",
                    comparison.gpu_speedup, comparison.wall_speedup
                );
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

fn dispatch_attention(
    context: &MetalContext,
    query: &metal_infer_core::Tensor,
    key: &metal_infer_core::Tensor,
    value: &metal_infer_core::Tensor,
    config: AttentionConfig,
) -> Result<DispatchStats, metal_infer_core::CoreError> {
    let mut batch = context.begin_batch()?;
    let _output = batch.attention(query, key, value, config, AttentionKind::Tiled)?;
    batch.finish()
}

fn run_fusion_benchmark(
    context: &MetalContext,
    kind: FusionKind,
    dimensions: FusionDimensions,
    iterations: usize,
    warmup: usize,
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
    match kind {
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
        allocation_growth_bytes: Some(
            context
                .allocated_bytes()
                .saturating_sub(allocated_before_measurement),
        ),
        fusions: None,
        comparison: Some(comparison),
        prefill: None,
        decode: None,
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
            offset,
            10_000.0,
            1.0e-6,
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
        allocation_growth_bytes: None,
        fusions: None,
        comparison: None,
        prefill: None,
        decode: None,
    }
}

fn run_model_iteration(
    model: &Qwen3Model,
    prompt: &[u32],
    decode_tokens: usize,
    cache: &mut KvCache,
) -> Result<(PhaseSample, PhaseSample), CliError> {
    let started = Instant::now();
    let (_, prefill_stats) = model.prefill_with_stats(prompt, cache)?;
    let prefill = PhaseSample {
        wall: started.elapsed(),
        gpu: prefill_stats.gpu_time,
    };
    let started = Instant::now();
    let mut decode_gpu = Duration::ZERO;
    for _ in 0..decode_tokens {
        let (_, stats) = model.decode_with_stats(1, cache)?;
        decode_gpu += stats.gpu_time;
    }
    Ok((
        prefill,
        PhaseSample {
            wall: started.elapsed(),
            gpu: decode_gpu,
        },
    ))
}

fn model_report(
    context: &MetalContext,
    fusion_options: FusionOptions,
    allocated_before_measurement: usize,
    prompt_tokens: usize,
    decode_tokens: usize,
    prefill_samples: Vec<PhaseSample>,
    decode_samples: Vec<PhaseSample>,
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
        allocation_growth_bytes: Some(
            context
                .allocated_bytes()
                .saturating_sub(allocated_before_measurement),
        ),
        fusions: Some(fusion_options.into()),
        comparison: None,
        prefill: Some(prefill),
        decode: Some(decode),
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
