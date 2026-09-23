use std::fs;
use std::path::Path;
use std::time::Instant;

use clap::ValueEnum;
use metal_infer_cli::CliError;
use metal_infer_models::{KvCache, Qwen3Model};
use metal_infer_runtime::MetalContext;
use serde::{Deserialize, Serialize};

use super::Format;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(super) enum InferenceMode {
    Fixed,
    Autoregressive,
}

impl InferenceMode {
    const fn name(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Autoregressive => "autoregressive",
        }
    }
}

#[derive(Deserialize)]
struct Workload {
    schema_version: u32,
    case_id: String,
    prompt_text: String,
    prompt_token_ids: Vec<u32>,
    forced_decode_ids: Vec<u32>,
    output_tokens: usize,
}

impl Workload {
    fn validate(&self) -> Result<(), CliError> {
        if self.schema_version != 1
            || self.case_id.is_empty()
            || self.prompt_text.is_empty()
            || self.prompt_token_ids.is_empty()
            || self.output_tokens < 2
            || self.forced_decode_ids.len() != self.output_tokens - 1
        {
            return Err(CliError::InvalidArguments(
                "workload requires schema_version=1, nonempty prompt and case_id, and output_tokens-1 forced IDs".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct Sample {
    ttft_ms: f64,
    tpot_ms: f64,
    total_ms: f64,
    token_times_ms: Vec<f64>,
    sampled_token_ids: Vec<u32>,
    fed_token_ids: Vec<u32>,
}

#[derive(Serialize)]
struct InferenceReport<'workload> {
    schema_version: u32,
    kind: &'static str,
    backend: &'static str,
    mode: &'static str,
    case_id: &'workload str,
    device: String,
    prompt_tokens: usize,
    output_tokens: usize,
    warmup: usize,
    iterations: usize,
    samples: Vec<Sample>,
}

pub(super) fn run(
    model_path: &Path,
    workload_path: &Path,
    mode: InferenceMode,
    iterations: usize,
    warmup: usize,
    format: Format,
) -> Result<(), CliError> {
    if iterations == 0 {
        return Err(CliError::InvalidArguments(
            "iterations must be greater than zero".into(),
        ));
    }
    let workload: Workload = serde_json::from_slice(&fs::read(workload_path)?)?;
    workload.validate()?;
    let context = MetalContext::new()?;
    let model = Qwen3Model::load(model_path, &context)?;
    let mut cache = KvCache::new(
        &context,
        model.config(),
        workload.prompt_token_ids.len() + workload.output_tokens,
    )?;
    for _ in 0..warmup {
        let _ = run_sample(&model, &mut cache, &workload, mode)?;
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        samples.push(run_sample(&model, &mut cache, &workload, mode)?);
    }
    let report = InferenceReport {
        schema_version: 2,
        kind: "inference",
        backend: "metal-infer",
        mode: mode.name(),
        case_id: &workload.case_id,
        device: context.device_name(),
        prompt_tokens: workload.prompt_token_ids.len(),
        output_tokens: workload.output_tokens,
        warmup,
        iterations,
        samples,
    };
    match format {
        Format::Json => println!("{}", serde_json::to_string_pretty(&report)?),
        Format::Table => {
            let ttft = median(report.samples.iter().map(|sample| sample.ttft_ms).collect());
            let tpot = median(report.samples.iter().map(|sample| sample.tpot_ms).collect());
            let total = median(
                report
                    .samples
                    .iter()
                    .map(|sample| sample.total_ms)
                    .collect(),
            );
            println!("{} {} on {}", report.backend, report.mode, report.device);
            println!(
                "{}: {} prompt, {} output tokens",
                report.case_id, report.prompt_tokens, report.output_tokens
            );
            println!("median TTFT {ttft:.3} ms, TPOT {tpot:.3} ms, total {total:.3} ms");
        }
    }
    Ok(())
}

fn run_sample(
    model: &Qwen3Model,
    cache: &mut KvCache,
    workload: &Workload,
    mode: InferenceMode,
) -> Result<Sample, CliError> {
    cache.reset();
    let started = Instant::now();
    let mut logits = model.prefill(&workload.prompt_token_ids, cache)?;
    let mut sampled_token_ids = Vec::with_capacity(workload.output_tokens);
    let mut fed_token_ids = Vec::with_capacity(workload.output_tokens - 1);
    let mut token_times_ms = Vec::with_capacity(workload.output_tokens);
    for step in 0..workload.output_tokens {
        let token = argmax(&logits.to_f32_vec()?)?;
        sampled_token_ids.push(token);
        token_times_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        if step + 1 < workload.output_tokens {
            let input = match mode {
                InferenceMode::Fixed => *workload
                    .forced_decode_ids
                    .get(step)
                    .expect("validated forced token trace"),
                InferenceMode::Autoregressive => token,
            };
            fed_token_ids.push(input);
            logits = model.decode(input, cache)?;
        }
    }
    let ttft_ms = *token_times_ms.first().expect("at least two output tokens");
    let total_ms = *token_times_ms.last().expect("at least two output tokens");
    Ok(Sample {
        ttft_ms,
        tpot_ms: (total_ms - ttft_ms) / (workload.output_tokens - 1) as f64,
        total_ms,
        token_times_ms,
        sampled_token_ids,
        fed_token_ids,
    })
}

fn argmax(values: &[f32]) -> Result<u32, CliError> {
    let (index, _) = values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or_else(|| CliError::InvalidArguments("logits contain no finite value".into()))?;
    index
        .try_into()
        .map_err(|_| CliError::InvalidArguments("token ID exceeds u32".into()))
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    *values.get(values.len() / 2).expect("nonempty samples")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_misaligned_trace() {
        let workload = Workload {
            schema_version: 1,
            case_id: "case".into(),
            prompt_text: "prompt".into(),
            prompt_token_ids: vec![1],
            forced_decode_ids: vec![2],
            output_tokens: 3,
        };
        assert!(
            workload.validate().is_err(),
            "trace must have one fewer token than output"
        );
    }

    #[test]
    fn argmax_uses_generation_tie_behavior() {
        assert_eq!(argmax(&[2.0, 2.0]).expect("argmax"), 1);
    }
}
