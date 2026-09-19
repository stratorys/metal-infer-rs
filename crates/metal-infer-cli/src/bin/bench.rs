use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, Subcommand, ValueEnum};
use metal_infer_cli::CliError;
use metal_infer_core::{AttentionKind, MetalContext};
use metal_infer_models::{KvCache, Qwen3Model};
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
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Table,
    Json,
}

#[derive(Serialize)]
struct Report {
    benchmark: String,
    device: String,
    iterations: usize,
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    throughput: Option<f64>,
    throughput_unit: Option<&'static str>,
    allocated_bytes: usize,
    prefill: Option<PhaseReport>,
    decode: Option<PhaseReport>,
}

#[derive(Serialize)]
struct PhaseReport {
    tokens: usize,
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    tokens_per_second: f64,
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
                let _ = context.matmul(&input, &weight)?;
            }
            let samples = measure(iterations, || context.matmul(&input, &weight).map(|_| ()))?;
            let operations = 2.0 * m as f64 * n as f64 * k as f64;
            let throughput = (operations / 1.0e12) / mean_seconds(&samples);
            report(
                format!("matmul_f16[{m},{n},{k}]"),
                &context,
                samples,
                Some(throughput),
                Some("TFLOP/s"),
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
            report("qwen3_block".into(), &context, samples, None, None)
        }
        Command::Model {
            model,
            prompt,
            generate,
            iterations,
            warmup,
        } => {
            if prompt == 0 || generate == 0 || iterations == 0 {
                return Err(CliError::InvalidArguments(
                    "model prompt, generate, and iterations must be greater than zero".into(),
                ));
            }
            let model = Qwen3Model::load(&model, &context)?;
            let mut cache = KvCache::new(&context, model.config(), prompt + generate)?;
            let token_ids = vec![1; prompt];
            for _ in 0..warmup {
                let _ = run_model_iteration(&model, &token_ids, generate, &mut cache)?;
            }
            let mut prefill_samples = Vec::with_capacity(iterations);
            let mut decode_samples = Vec::with_capacity(iterations);
            for _ in 0..iterations {
                let (prefill, decode) =
                    run_model_iteration(&model, &token_ids, generate, &mut cache)?;
                prefill_samples.push(prefill);
                decode_samples.push(decode);
            }
            model_report(&context, prompt, generate, prefill_samples, decode_samples)
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
            if let (Some(value), Some(unit)) = (report.throughput, report.throughput_unit) {
                println!("throughput: {value:.3} {unit}");
            }
            if let Some(prefill) = &report.prefill {
                println!(
                    "prefill: {:.3} ms, {:.3} tokens/s",
                    prefill.mean_ms, prefill.tokens_per_second
                );
            }
            if let Some(decode) = &report.decode {
                println!(
                    "decode: {:.3} ms, {:.3} tokens/s",
                    decode.mean_ms, decode.tokens_per_second
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

fn report(
    benchmark: String,
    context: &MetalContext,
    mut samples: Vec<Duration>,
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
    Report {
        benchmark,
        device: context.device_name(),
        iterations: samples.len(),
        mean_ms: mean * 1000.0,
        median_ms: median * 1000.0,
        p95_ms: p95 * 1000.0,
        throughput,
        throughput_unit,
        allocated_bytes: context.allocated_bytes(),
        prefill: None,
        decode: None,
    }
}

fn run_model_iteration(
    model: &Qwen3Model,
    prompt: &[u32],
    decode_tokens: usize,
    cache: &mut KvCache,
) -> Result<(Duration, Duration), CliError> {
    let started = Instant::now();
    let _ = model.prefill(prompt, cache)?;
    let prefill = started.elapsed();
    let started = Instant::now();
    for _ in 0..decode_tokens {
        let _ = model.decode(1, cache)?;
    }
    Ok((prefill, started.elapsed()))
}

fn model_report(
    context: &MetalContext,
    prompt_tokens: usize,
    decode_tokens: usize,
    mut prefill_samples: Vec<Duration>,
    mut decode_samples: Vec<Duration>,
) -> Report {
    prefill_samples.sort();
    decode_samples.sort();
    let prefill = phase_report(prompt_tokens, &prefill_samples);
    let decode = phase_report(decode_tokens, &decode_samples);
    Report {
        benchmark: "qwen3_model".into(),
        device: context.device_name(),
        iterations: prefill_samples.len(),
        mean_ms: prefill.mean_ms + decode.mean_ms,
        median_ms: prefill.median_ms + decode.median_ms,
        p95_ms: prefill.p95_ms + decode.p95_ms,
        throughput: None,
        throughput_unit: None,
        allocated_bytes: context.allocated_bytes(),
        prefill: Some(prefill),
        decode: Some(decode),
    }
}

fn phase_report(
    tokens: usize,
    samples: &[Duration],
) -> PhaseReport {
    let mean = mean_seconds(samples);
    let median = samples
        .get(samples.len() / 2)
        .map_or(0.0, Duration::as_secs_f64);
    let p95_index = (samples.len().saturating_sub(1) as f64 * 0.95).round() as usize;
    let p95 = samples.get(p95_index).map_or(0.0, Duration::as_secs_f64);
    PhaseReport {
        tokens,
        mean_ms: mean * 1000.0,
        median_ms: median * 1000.0,
        p95_ms: p95 * 1000.0,
        tokens_per_second: tokens as f64 / mean,
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
