# metal-infer

An experimental Transformer inference engine written in Rust for Apple Silicon,
built directly on `objc2-metal` and native Metal Shading Language kernels.

## Current status

- FP16 execution and direct BF16-to-FP16 Safetensors loading without a
  tensor-sized conversion copy;
- tiled GEMM for prefill and GEMV for decode;
- RMSNorm, RoPE, GQA, online causal attention, and SwiGLU;
- dense Qwen3 blocks, KV cache, prefill, decode, greedy and sampled generation;
- an OpenAI-compatible chat completions server with SSE streaming;
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
  generate \
  --model ./models/Qwen3-0.6B \
  --prompt 'Hello' --max-tokens 32 --context 2048

# Start an OpenAI-compatible server for Open WebUI
cargo run --release --bin metal-infer -- \
  serve --model Qwen/Qwen3-0.6B --context 8192 --bind 127.0.0.1:8080

# Test the server
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":true}'
```

Open WebUI can use `http://host.docker.internal:8080/v1` when it runs in
Docker, or `http://127.0.0.1:8080/v1` when it runs directly on the Mac. The
first server version intentionally serializes generation requests. A Hugging
Face repository ID resolves to its locally cached snapshot; use
`hf download <owner/model>` first if it is not cached.

## Workspace layout

- `metal-infer-runtime`: Metal device, buffers, tensors, command batches, and profiling;
- `metal-infer-kernels`: Metal kernels, their Rust wrappers, and kernel variant selection;
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
