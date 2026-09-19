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
    },
    Block {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 128)]
        tokens: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
    },
    Model {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 512)]
        prompt: usize,
        #[arg(long, default_value_t = 128)]
        generate: usize,
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
        } => {
            require_iterations(iterations)?;
            let input = context.tensor_f16(&vec![0.01; m * k], &[m, k])?;
            let weight = context.tensor_f16(&vec![0.02; n * k], &[n, k])?;
            let _ = context.matmul(&input, &weight)?;
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
        } => {
            require_iterations(iterations)?;
            let mut model = Qwen3Model::load(&model, &context)?;
            model.set_attention_kind(AttentionKind::Tiled);
            let token_ids = vec![1; tokens];
            let _ = model.run_first_block(&token_ids)?;
            let samples = measure(iterations, || model.run_first_block(&token_ids).map(|_| ()))?;
            report("qwen3_block".into(), &context, samples, None, None)
        }
        Command::Model {
            model,
            prompt,
            generate,
        } => {
            if prompt == 0 || generate == 0 {
                return Err(CliError::InvalidArguments(
                    "model prompt and generate lengths must be greater than zero".into(),
                ));
            }
            let model = Qwen3Model::load(&model, &context)?;
            let mut cache = KvCache::new(&context, model.config(), prompt + generate)?;
            let token_ids = vec![1; prompt];
            let started = Instant::now();
            let _ = model.generate(&token_ids, generate, &mut cache)?;
            let elapsed = started.elapsed();
            report(
                "qwen3_model".into(),
                &context,
                vec![elapsed],
                Some((prompt + generate) as f64 / elapsed.as_secs_f64()),
                Some("tokens/s"),
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
            if let (Some(value), Some(unit)) = (report.throughput, report.throughput_unit) {
                println!("throughput: {value:.3} {unit}");
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
