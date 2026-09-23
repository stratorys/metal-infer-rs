use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use metal_infer_cli::{CliError, LogFormat, init_tracing, load_model_with, resolve_model_path};
use metal_infer_kernels::Kernels;
use metal_infer_models::{KvCache, Qwen3Model};
use metal_infer_runtime::{KernelDispatchProfile, MetalContext, Tensor};
use serde::Serialize;

#[derive(Parser)]
#[command(
    name = "metal-infer-bench",
    about = "Offline prompt processing and token generation benchmark"
)]
struct Arguments {
    #[arg(long, value_enum, default_value_t = LogFormat::Text)]
    log_format: LogFormat,
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
    #[arg(long, value_enum, default_value_t = BenchTest::Pg)]
    test: BenchTest,
    #[arg(long, default_value_t = 0)]
    depth: usize,
    #[arg(long = "with", value_name = "KEY=VALUE")]
    with: Vec<String>,
    #[arg(long)]
    profile: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
enum BenchTest {
    Pp,
    Tg,
    Pg,
}

#[derive(Clone, Copy)]
struct Iteration<'input> {
    prompt: &'input [u32],
    depth_prompt: &'input [u32],
    generate: usize,
    profile: bool,
    test: BenchTest,
}

#[derive(Serialize)]
struct Report {
    engine: &'static str,
    device: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    warmup: usize,
    test: BenchTest,
    depth: usize,
    load_ms: f64,
    allocated_bytes: usize,
    allocation_growth_bytes: usize,
    plan: BTreeMap<&'static str, String>,
    samples: Vec<Sample>,
    #[serde(skip_serializing_if = "Option::is_none")]
    kernel_profile: Option<KernelProfile>,
}

#[derive(Clone, Copy, Serialize)]
struct Sample {
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_ms: Option<f64>,
}

#[derive(Serialize)]
struct KernelProfile {
    prefill: Vec<KernelRow>,
    decode: Vec<KernelRow>,
}

#[derive(Serialize)]
struct KernelRow {
    kernel: String,
    calls: usize,
    gpu_ms_per_iteration: f64,
}

fn main() {
    let arguments = Arguments::parse();
    if let Err(error) = init_tracing(arguments.log_format) {
        eprintln!("{error}");
        std::process::exit(1);
    }
    if let Err(error) = run(arguments) {
        tracing::error!(message = "Benchmark failed.", %error);
        std::process::exit(1);
    }
}

fn run(arguments: Arguments) -> Result<(), CliError> {
    if arguments.iterations == 0
        || (matches!(arguments.test, BenchTest::Pp | BenchTest::Pg) && arguments.prompt == 0)
        || (matches!(arguments.test, BenchTest::Tg | BenchTest::Pg) && arguments.generate == 0)
        || (matches!(arguments.test, BenchTest::Tg) && arguments.depth == 0)
        || (arguments.profile && !matches!(arguments.test, BenchTest::Pg))
    {
        return Err(CliError::InvalidBenchArguments);
    }
    let context = MetalContext::new()?;
    let kernels = Kernels::new(&context)?;
    let model_path = resolve_model_path(&arguments.model)?;
    let started = Instant::now();
    let Some(model) = load_model_with(&model_path, kernels, &arguments.with)? else {
        return Ok(());
    };
    let load_ms = milliseconds(started.elapsed());
    let mut cache = KvCache::new(
        &context,
        model.config(),
        arguments.prompt.max(arguments.depth) + arguments.generate.max(1),
    )?;
    let prompt = vec![1; arguments.prompt];
    let depth_prompt = vec![1; arguments.depth];
    for _ in 0..arguments.warmup {
        run_iteration(
            &context,
            &model,
            &mut cache,
            Iteration {
                prompt: &prompt,
                depth_prompt: &depth_prompt,
                generate: arguments.generate,
                profile: false,
                test: arguments.test,
            },
        )?;
    }
    context.set_kernel_profiling(arguments.profile)?;
    let allocated_before = context.allocated_bytes();
    let mut samples = Vec::with_capacity(arguments.iterations);
    let mut prefill_dispatches = Vec::new();
    let mut decode_dispatches = Vec::new();
    for _ in 0..arguments.iterations {
        let (sample, prefill, decode) = run_iteration(
            &context,
            &model,
            &mut cache,
            Iteration {
                prompt: &prompt,
                depth_prompt: &depth_prompt,
                generate: arguments.generate,
                profile: arguments.profile,
                test: arguments.test,
            },
        )?;
        samples.push(sample);
        prefill_dispatches.extend(prefill);
        decode_dispatches.extend(decode);
    }
    context.set_kernel_profiling(false)?;
    let report = Report {
        engine: "metal-infer",
        device: context.device_name(),
        prompt_tokens: arguments.prompt,
        generated_tokens: arguments.generate,
        warmup: arguments.warmup,
        test: arguments.test,
        depth: arguments.depth,
        load_ms,
        allocated_bytes: context.allocated_bytes(),
        allocation_growth_bytes: context.allocated_bytes().saturating_sub(allocated_before),
        plan: model.plan().entries().into_iter().collect(),
        samples,
        kernel_profile: arguments.profile.then(|| KernelProfile {
            prefill: kernel_rows(prefill_dispatches, arguments.iterations),
            decode: kernel_rows(decode_dispatches, arguments.iterations),
        }),
    };
    tracing::info!(
        iterations = arguments.iterations,
        load_ms,
        allocated_bytes = report.allocated_bytes,
        "benchmark complete"
    );
    std::io::stdout()
        .write_all(format!("{}\n", serde_json::to_string_pretty(&report)?).as_bytes())?;
    Ok(())
}

fn run_iteration(
    context: &MetalContext,
    model: &Qwen3Model,
    cache: &mut KvCache,
    iteration: Iteration<'_>,
) -> Result<
    (
        Sample,
        Vec<KernelDispatchProfile>,
        Vec<KernelDispatchProfile>,
    ),
    CliError,
> {
    let Iteration {
        prompt,
        depth_prompt,
        generate,
        profile,
        test,
    } = iteration;
    if matches!(test, BenchTest::Pp) {
        let started = Instant::now();
        model.prefill(prompt, cache)?;
        return Ok((
            Sample {
                prefill_ms: Some(milliseconds(started.elapsed())),
                decode_ms: None,
            },
            Vec::new(),
            Vec::new(),
        ));
    }
    if matches!(test, BenchTest::Tg) {
        let slots = context.tensor_u32(&vec![u32::MAX; generate + 1], &[generate + 1])?;
        model.prefill_argmax(depth_prompt, cache, &slots.slice_1d(0, 1)?)?;
        let started = Instant::now();
        run_pipelined_decode(model, &slots, generate, cache)?;
        return Ok((
            Sample {
                prefill_ms: None,
                decode_ms: Some(milliseconds(started.elapsed())),
            },
            Vec::new(),
            Vec::new(),
        ));
    }
    let started = Instant::now();
    let (prefill_ms, mut logits, slots) = if profile {
        let (logits, _) = model.prefill_with_stats(prompt, cache)?;
        (milliseconds(started.elapsed()), Some(logits), None)
    } else {
        let slots = context.tensor_u32(&vec![u32::MAX; generate + 1], &[generate + 1])?;
        model.prefill_argmax(prompt, cache, &slots.slice_1d(0, 1)?)?;
        (milliseconds(started.elapsed()), None, Some(slots))
    };
    let prefill_dispatches = if profile {
        context.take_kernel_profiles()
    } else {
        Vec::new()
    };
    let started = Instant::now();
    if let Some(slots) = &slots {
        run_pipelined_decode(model, slots, generate, cache)?;
    } else if let Some(mut current) = logits.take() {
        for _ in 0..generate {
            let token = argmax(&current.to_f32_vec()?)?;
            current = model.decode_with_stats(token, cache)?.0;
        }
    }
    let decode_ms = milliseconds(started.elapsed());
    let decode_dispatches = if profile {
        context.take_kernel_profiles()
    } else {
        Vec::new()
    };
    Ok((
        Sample {
            prefill_ms: Some(prefill_ms),
            decode_ms: Some(decode_ms),
        },
        prefill_dispatches,
        decode_dispatches,
    ))
}

fn run_pipelined_decode(
    model: &Qwen3Model,
    slots: &Tensor,
    generate: usize,
    cache: &mut KvCache,
) -> Result<(), CliError> {
    let mut queued = None;
    for step in 0..generate {
        let current = if let Some(batch) = queued.take() {
            batch
        } else {
            let input = slots.slice_1d(step, 1)?;
            let output = slots.slice_1d(step + 1, 1)?;
            model.decode_argmax(&input, cache, &output)?.1
        };
        let next = if step + 1 < generate {
            let input = slots.slice_1d(step + 1, 1)?;
            let output = slots.slice_1d(step + 2, 1)?;
            match model.decode_argmax(&input, cache, &output) {
                Ok((_, batch)) => Some(batch),
                Err(error) => {
                    let _ = current.wait();
                    return Err(error.into());
                }
            }
        } else {
            None
        };
        if let Err(error) = current.wait() {
            drop(next);
            return Err(error.into());
        }
        if slots.slice_1d(step + 1, 1)?.to_u32_vec()?.first().copied() == Some(u32::MAX) {
            drop(next);
            return Err(CliError::NoFiniteLogit);
        }
        queued = next;
    }
    Ok(())
}

fn argmax(values: &[f32]) -> Result<u32, CliError> {
    let (index, _) = values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|left, right| left.1.total_cmp(right.1))
        .ok_or(CliError::NoFiniteLogit)?;
    u32::try_from(index).map_err(|_| CliError::TokenIdOverflow)
}

fn kernel_rows(
    dispatches: Vec<KernelDispatchProfile>,
    iterations: usize,
) -> Vec<KernelRow> {
    let mut totals: HashMap<String, (usize, f64)> = HashMap::new();
    for dispatch in dispatches {
        let entry = totals.entry(dispatch.kernel).or_default();
        entry.0 += 1;
        entry.1 += milliseconds(dispatch.gpu_time);
    }
    let mut rows: Vec<_> = totals
        .into_iter()
        .map(|(kernel, (calls, total))| KernelRow {
            kernel,
            calls: calls / iterations,
            gpu_ms_per_iteration: total / iterations as f64,
        })
        .collect();
    rows.sort_by(|left, right| {
        right
            .gpu_ms_per_iteration
            .total_cmp(&left.gpu_ms_per_iteration)
            .then_with(|| left.kernel.cmp(&right.kernel))
    });
    rows
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
