use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error(transparent)]
    Gpu(#[from] metal_infer_kernels::GpuError),
    #[error(transparent)]
    Kernel(#[from] metal_infer_kernels::KernelError),
    #[error(transparent)]
    Plan(#[from] metal_infer_planner::PlanError),
    #[error("model file I/O failed")]
    Io(#[from] std::io::Error),
    #[error("model JSON is invalid")]
    Json(#[from] serde_json::Error),
    #[error("safetensors file is invalid")]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("model path is not valid UTF-8")]
    NonUtf8ModelPath,
    #[error("model directory does not exist")]
    ModelDirectoryMissing,
    #[error(
        "cannot locate the Hugging Face cache because HOME, HF_HOME, and HUGGINGFACE_HUB_CACHE are unset"
    )]
    HuggingFaceCacheMissing,
    #[error("Hugging Face model is not in the local cache, download it with `hf download`")]
    ModelNotCached,
    #[error("cannot load the tokenizer")]
    TokenizerLoad,
    #[error("tokenizer cannot encode the text")]
    TokenizerEncode,
    #[error("tokenizer cannot decode the token sequence")]
    TokenizerDecode,
    #[error("model directory contains neither model.safetensors nor its index")]
    MissingWeights,
    #[error("model tensor is missing")]
    MissingTensor,
    #[error("model tensor dtype must be F16 or BF16")]
    UnsupportedTensorDType,
    #[error("model tensor has an unexpected shape")]
    WeightShape,
    #[error("model_type must be qwen3")]
    UnsupportedModelType,
    #[error("attention biases are not supported")]
    AttentionBias,
    #[error("model dimensions must be non-zero and head_dim even")]
    InvalidDimensions,
    #[error("num_attention_heads must be divisible by num_key_value_heads")]
    InvalidHeadRatio,
    #[error("model has no transformer layer")]
    NoLayer,
    #[error("KV cache capacity must be non-zero")]
    EmptyKvCache,
    #[error("KV cache capacity exceeded")]
    CacheCapacity,
    #[error("KV cache layer is missing")]
    MissingCacheLayer,
    #[error("KV cache layer count differs from the model")]
    CacheLayerCount,
    #[error("decode batch needs one cache per token")]
    DecodeBatchMismatch,
    #[error("prefill requires at least one token")]
    EmptyPrefill,
    #[error("decode token must be one u32")]
    DecodeTokenShape,
    #[error("tokens must be a nonempty u32 vector")]
    TokensShape,
    #[error("hidden state has no token dimension")]
    HiddenStateRank,
    #[error("argmax token is missing")]
    MissingArgmaxToken,
    #[error("logits contain no finite value")]
    NoFiniteLogit,
    #[error("token id does not fit in u32")]
    TokenIdOverflow,
    #[error("temperature must be finite and non-negative")]
    InvalidTemperature,
    #[error("top_p must be between 0 and 1")]
    InvalidTopP,
    #[error("failed to sample a token")]
    SamplingFailed,
    #[error("chat completion requires at least one message")]
    EmptyChat,
    #[error("chat role is not supported")]
    UnsupportedChatRole,
}
