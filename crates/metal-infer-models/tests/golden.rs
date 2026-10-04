use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use half::f16;
use metal_infer_kernels::MetalContext;
use metal_infer_models::{
    ChatMessage, GenerationOptions, KvCache, ModelSource, ModelTokenizer, Qwen3Model,
};
use serde::{Deserialize, Serialize};

const PROMPT_LENGTHS: [usize; 9] = [1, 63, 64, 65, 511, 512, 513, 2048, 4097];
const DECODE_STEPS: usize = 64;
const GENERATED_TOKENS: usize = 64;
const PROMPTS: [&str; 3] = [
    "Hello",
    "Explain the Metal shading language in one paragraph.",
    "Write a Rust function that reverses a string.",
];

#[test]
fn batched_decode_matches_independent_requests() {
    let context = MetalContext::new().expect("Metal device");
    let model = Qwen3Model::load(&model_directory(), &context).expect("load Qwen3");
    let prompts = [
        synthetic_prompt(64),
        synthetic_prompt(512),
        synthetic_prompt(129),
    ];
    let mut independent = prompts
        .iter()
        .map(|prompt| {
            let mut cache =
                KvCache::new(&context, model.config(), prompt.len() + 4).expect("KV cache");
            model.prefill(prompt, &mut cache).expect("prefill");
            cache
        })
        .collect::<Vec<_>>();
    let mut batched = prompts
        .iter()
        .map(|prompt| {
            let mut cache =
                KvCache::new(&context, model.config(), prompt.len() + 4).expect("KV cache");
            model.prefill(prompt, &mut cache).expect("prefill");
            cache
        })
        .collect::<Vec<_>>();
    let mut tokens = vec![1, 2, 3];
    for _ in 0..3 {
        let expected = independent
            .iter_mut()
            .zip(&tokens)
            .map(|(cache, token)| {
                let logits = model.decode(*token, cache).expect("independent decode");
                argmax(&logits.with_f16_bits(|bits| bits.to_vec()).expect("logits"))
            })
            .collect::<Vec<_>>();
        let mut cache_refs = batched.iter_mut().collect::<Vec<_>>();
        let logits = model
            .decode_batch(&tokens, &mut cache_refs)
            .expect("batched decode");
        let actual = (0..tokens.len())
            .map(|row| {
                argmax(
                    &logits
                        .row(row)
                        .expect("row")
                        .with_f16_bits(|bits| bits.to_vec())
                        .expect("logits"),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        tokens = actual;
    }
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
struct Golden {
    device: String,
    logits: BTreeMap<String, Vec<Step>>,
    generations: BTreeMap<String, Vec<u32>>,
    kernels: BTreeMap<String, BTreeMap<String, usize>>,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
struct Step {
    hash: String,
    token: u32,
}

#[test]
fn outputs_match_golden_reference() {
    let model_path = model_directory();
    let context = MetalContext::new().expect("Metal device");
    let model = Qwen3Model::load(&model_path, &context).expect("load Qwen3");
    let tokenizer = ModelTokenizer::from_directory(&model_path).expect("load tokenizer");

    let actual = Golden {
        device: context.device_name(),
        logits: record_logits(&context, &model),
        generations: record_generations(&context, &model, &tokenizer),
        kernels: record_kernels(&context, &model),
    };

    let path = golden_path(&actual.device);
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        fs::create_dir_all(path.parent().expect("golden directory")).expect("create directory");
        let json = serde_json::to_string_pretty(&actual).expect("serialize golden");
        fs::write(&path, json + "\n").expect("write golden");
        eprintln!("recorded {}", path.display());
        return;
    }
    let bytes = fs::read(&path).unwrap_or_else(|_| {
        panic!(
            "missing {}; record it with GOLDEN_RECORD=1 on this device",
            path.display()
        )
    });
    let expected: Golden = serde_json::from_slice(&bytes).expect("parse golden");
    let differences = compare(&expected, &actual);
    assert!(
        differences.is_empty(),
        "{} difference(s) from {}:\n{}",
        differences.len(),
        path.display(),
        differences.join("\n")
    );
}

fn record_logits(
    context: &MetalContext,
    model: &Qwen3Model,
) -> BTreeMap<String, Vec<Step>> {
    let mut logits = BTreeMap::new();
    for length in PROMPT_LENGTHS {
        let prompt = synthetic_prompt(length);
        let mut cache =
            KvCache::new(context, model.config(), length + DECODE_STEPS).expect("KV cache");
        let mut output = model.prefill(&prompt, &mut cache).expect("prefill");
        let mut steps = Vec::with_capacity(DECODE_STEPS + 1);
        for step in 0..=DECODE_STEPS {
            let bits = output
                .with_f16_bits(|bits| bits.to_vec())
                .expect("read logits");
            let token = argmax(&bits);
            steps.push(Step {
                hash: format!("{:016x}", fnv1a(&bits)),
                token,
            });
            if step < DECODE_STEPS {
                output = model.decode(token, &mut cache).expect("decode");
            }
        }
        logits.insert(format!("prompt_{length:04}"), steps);
    }
    logits
}

fn record_generations(
    context: &MetalContext,
    model: &Qwen3Model,
    tokenizer: &ModelTokenizer,
) -> BTreeMap<String, Vec<u32>> {
    let mut generations = BTreeMap::new();
    for (index, text) in PROMPTS.iter().enumerate() {
        let prompt = tokenizer
            .encode_chat(&[ChatMessage {
                role: "user".into(),
                content: (*text).into(),
            }])
            .expect("encode prompt");
        let modes = [
            ("greedy", GenerationOptions::default()),
            (
                "sampled",
                GenerationOptions {
                    temperature: 0.7,
                    top_p: 0.9,
                    top_k: 40,
                    seed: 42,
                    ..GenerationOptions::default()
                },
            ),
        ];
        for (mode, options) in modes {
            let options = GenerationOptions {
                max_tokens: GENERATED_TOKENS,
                ..options
            };
            let mut cache = KvCache::new(context, model.config(), prompt.len() + GENERATED_TOKENS)
                .expect("KV cache");
            let tokens = model
                .generate_with(&prompt, &options, &mut cache, |_| true)
                .expect("generate");
            generations.insert(format!("prompt_{index}_{mode}"), tokens);
        }
    }
    generations
}

fn record_kernels(
    context: &MetalContext,
    model: &Qwen3Model,
) -> BTreeMap<String, BTreeMap<String, usize>> {
    let prompt = synthetic_prompt(513);
    let mut cache = KvCache::new(context, model.config(), prompt.len() + 8).expect("KV cache");
    context
        .set_kernel_profiling(true)
        .expect("enable profiling");
    let mut output = model.prefill(&prompt, &mut cache).expect("prefill");
    let prefill = count_kernels(context);
    for _ in 0..8 {
        let token = output.with_f16_bits(argmax).expect("read logits");
        output = model.decode(token, &mut cache).expect("decode");
    }
    let decode = count_kernels(context);
    context
        .set_kernel_profiling(false)
        .expect("disable profiling");
    BTreeMap::from([("prefill".into(), prefill), ("decode".into(), decode)])
}

fn count_kernels(context: &MetalContext) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for profile in context.take_kernel_profiles() {
        *counts.entry(profile.kernel).or_insert(0) += 1;
    }
    counts
}

fn compare(
    expected: &Golden,
    actual: &Golden,
) -> Vec<String> {
    let mut differences = Vec::new();
    for (name, expected_steps) in &expected.logits {
        let Some(actual_steps) = actual.logits.get(name) else {
            differences.push(format!("logits {name}: missing"));
            continue;
        };
        let first = expected_steps
            .iter()
            .zip(actual_steps)
            .position(|(left, right)| left != right);
        if let Some(step) = first {
            let expected_step = expected_steps.get(step).expect("expected step");
            let actual_step = actual_steps.get(step).expect("actual step");
            differences.push(format!(
                "logits {name}: first difference at step {step} (0 = prefill): \
                 hash {} -> {}, token {} -> {}",
                expected_step.hash, actual_step.hash, expected_step.token, actual_step.token
            ));
        } else if expected_steps.len() != actual_steps.len() {
            differences.push(format!("logits {name}: step count differs"));
        }
    }
    for (name, expected_tokens) in &expected.generations {
        let actual_tokens = actual.generations.get(name);
        if actual_tokens != Some(expected_tokens) {
            let step = actual_tokens.map_or(0, |tokens| {
                expected_tokens
                    .iter()
                    .zip(tokens)
                    .position(|(left, right)| left != right)
                    .unwrap_or_else(|| expected_tokens.len().min(tokens.len()))
            });
            differences.push(format!(
                "generation {name}: first difference at token {step}"
            ));
        }
    }
    for (phase, expected_counts) in &expected.kernels {
        let empty = BTreeMap::new();
        let actual_counts = actual.kernels.get(phase).unwrap_or(&empty);
        let names = expected_counts.keys().chain(actual_counts.keys());
        let mut reported = Vec::new();
        for kernel in names {
            if reported.contains(&kernel) {
                continue;
            }
            reported.push(kernel);
            let before = expected_counts.get(kernel).copied().unwrap_or(0);
            let after = actual_counts.get(kernel).copied().unwrap_or(0);
            if before != after {
                differences.push(format!("kernels {phase}: {kernel} x{before} -> x{after}"));
            }
        }
    }
    differences
}

fn synthetic_prompt(length: usize) -> Vec<u32> {
    (0..length)
        .map(|index| u32::try_from((index * 7919 + 13) % 150_000).expect("token fits u32"))
        .collect()
}

fn argmax(bits: &[u16]) -> u32 {
    let mut best: Option<(usize, f32)> = None;
    for (index, value) in bits
        .iter()
        .map(|bits| f16::from_bits(*bits).to_f32())
        .enumerate()
    {
        if value.is_finite() && best.is_none_or(|(_, maximum)| value >= maximum) {
            best = Some((index, value));
        }
    }
    let (index, _) = best.expect("logits contain a finite value");
    u32::try_from(index).expect("token fits u32")
}

fn fnv1a(bits: &[u16]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bits.iter().flat_map(|value| value.to_le_bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

fn model_directory() -> PathBuf {
    if let Some(path) = std::env::var_os("QWEN3_MODEL") {
        return PathBuf::from(path);
    }
    ModelSource::resolve(Path::new("Qwen/Qwen3-0.6B"))
        .expect("Qwen3-0.6B must be cached; set QWEN3_MODEL or run `hf download Qwen/Qwen3-0.6B`")
        .directory
}

fn golden_path(device: &str) -> PathBuf {
    let slug: String = device
        .to_lowercase()
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '-'
            }
        })
        .collect();
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{slug}.json"))
}
