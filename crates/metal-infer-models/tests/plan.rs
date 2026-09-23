use std::fs;
use std::path::{Path, PathBuf};

use metal_infer_kernels::{Kernels, MetalContext};
use metal_infer_models::{KvCache, Qwen3Model};

const PROMPT_TOKENS: usize = 600;
const DECODE_STEPS: usize = 8;

#[test]
fn listed_plan_replays_bit_for_bit() {
    let path = model_directory();
    let context = MetalContext::new().expect("Metal device");
    let tuned = Qwen3Model::load(&path, &context).expect("load autotuned model");
    let overrides = tuned
        .plan()
        .entries()
        .into_iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>();
    let replayed =
        Qwen3Model::load_with(&path, Kernels::new(&context).expect("kernels"), &overrides)
            .expect("load model from listed plan");
    assert_eq!(
        replayed.plan(),
        tuned.plan(),
        "replaying every listed entry must reproduce the plan"
    );
    assert_eq!(
        logits(&context, &replayed),
        logits(&context, &tuned),
        "the replayed plan must produce bit-identical logits"
    );
}

#[test]
fn override_changes_executed_kernels() {
    let path = model_directory();
    let context = MetalContext::new().expect("Metal device");
    let fused = Qwen3Model::load(&path, &context).expect("load default model");
    let unfused = Qwen3Model::load_with(
        &path,
        Kernels::new(&context).expect("kernels"),
        &["fusion.qkv=off".to_owned()],
    )
    .expect("load model with fusion.qkv=off");
    assert!(
        fused.plan().fusions.qkv && !unfused.plan().fusions.qkv,
        "the override must only disable the QKV fusion in the second model"
    );
    assert!(
        decode_kernels(&context, &fused)
            .iter()
            .any(|kernel| kernel.starts_with("matvec3")),
        "the default plan must run a fused QKV projection"
    );
    assert!(
        !decode_kernels(&context, &unfused)
            .iter()
            .any(|kernel| kernel.starts_with("matvec3")),
        "fusion.qkv=off must run separate Q, K, and V projections"
    );
}

fn logits(
    context: &MetalContext,
    model: &Qwen3Model,
) -> Vec<Vec<u16>> {
    let prompt = synthetic_prompt();
    let mut cache =
        KvCache::new(context, model.config(), PROMPT_TOKENS + DECODE_STEPS).expect("KV cache");
    let mut output = model.prefill(&prompt, &mut cache).expect("prefill");
    let mut steps = Vec::with_capacity(DECODE_STEPS + 1);
    for step in 0..=DECODE_STEPS {
        let bits = output.with_f16_bits(<[u16]>::to_vec).expect("read logits");
        let token = argmax(&bits);
        steps.push(bits);
        if step < DECODE_STEPS {
            output = model.decode(token, &mut cache).expect("decode");
        }
    }
    steps
}

fn decode_kernels(
    context: &MetalContext,
    model: &Qwen3Model,
) -> Vec<String> {
    let prompt = synthetic_prompt();
    let mut cache =
        KvCache::new(context, model.config(), PROMPT_TOKENS + DECODE_STEPS).expect("KV cache");
    let output = model.prefill(&prompt, &mut cache).expect("prefill");
    let token = output.with_f16_bits(argmax).expect("read logits");
    context
        .set_kernel_profiling(true)
        .expect("enable profiling");
    model.decode(token, &mut cache).expect("decode");
    let kernels = context
        .take_kernel_profiles()
        .into_iter()
        .map(|profile| profile.kernel)
        .collect();
    context
        .set_kernel_profiling(false)
        .expect("disable profiling");
    kernels
}

fn synthetic_prompt() -> Vec<u32> {
    (0..PROMPT_TOKENS)
        .map(|index| u32::try_from((index * 7919 + 13) % 150_000).expect("token fits u32"))
        .collect()
}

fn argmax(bits: &[u16]) -> u32 {
    let mut best: Option<(usize, f32)> = None;
    for (index, value) in bits
        .iter()
        .map(|bits| half::f16::from_bits(*bits).to_f32())
        .enumerate()
    {
        if value.is_finite() && best.is_none_or(|(_, maximum)| value >= maximum) {
            best = Some((index, value));
        }
    }
    let (index, _) = best.expect("logits contain a finite value");
    u32::try_from(index).expect("token fits u32")
}

fn model_directory() -> PathBuf {
    if let Some(path) = std::env::var_os("QWEN3_MODEL") {
        return PathBuf::from(path);
    }
    let home = std::env::var_os("HOME").expect("HOME or QWEN3_MODEL must be set");
    let snapshots =
        Path::new(&home).join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    fs::read_dir(&snapshots)
        .ok()
        .and_then(|entries| entries.flatten().map(|entry| entry.path()).next())
        .unwrap_or_else(|| {
            panic!(
                "Qwen3-0.6B not found in {}; set QWEN3_MODEL or run \
                 `hf download Qwen/Qwen3-0.6B`",
                snapshots.display()
            )
        })
}
