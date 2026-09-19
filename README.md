# metal-infer

An experimental Transformer inference engine written in Rust for Apple Silicon,
built directly on `objc2-metal` and native Metal Shading Language kernels.

## Current status

- FP16 tensors and F16/BF16 Safetensors loading;
- tiled GEMM for prefill and GEMV for decode;
- RMSNorm, RoPE, GQA, online causal attention, and SwiGLU;
- dense Qwen3 blocks, KV cache, prefill, decode, and greedy generation;
- Rust benchmarks, MLX comparisons through `uv`, and llama.cpp commands;
- GPU path currently validated on an Apple M4 Pro.

## Quick start

```sh
# Verify Metal without downloading a model
cargo run --release --bin metal-infer-bench -- \
  kernel --m 64 --n 128 --k 128 --iterations 3

# Download Qwen3-0.6B
uvx --from huggingface-hub hf download Qwen/Qwen3-0.6B \
  --local-dir ./models/Qwen3-0.6B

# Generate text
cargo run --release --bin metal-infer -- \
  --model ./models/Qwen3-0.6B \
  --prompt 'Hello' --max-tokens 32 --context 2048
```

## Workspace layout

- `metal-infer-core`: Metal runtime, tensors, and GPU kernels;
- `metal-infer-models`: weight loading and the Qwen3 implementation;
- `metal-infer-cli`: text generation and benchmarks.

See [benchmarks](benchmarks/README.md) for the reproducible MLX and llama.cpp
comparison methodology and results.

## Validation

```sh
cargo +nightly fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

## License

Mozilla Public License 2.0. See [LICENSE](LICENSE).
