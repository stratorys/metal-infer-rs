use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error(transparent)]
    Core(#[from] metal_infer_runtime::CoreError),
    #[error(transparent)]
    Plan(#[from] metal_infer_planner::PlanError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Safetensors(#[from] safetensors::SafeTensorError),
    #[error("could not load tokenizer from `{0}`")]
    TokenizerLoad(std::path::PathBuf),
    #[error("tokenizer could not encode the provided text")]
    TokenizerEncode,
    #[error("tokenizer could not decode the provided token sequence")]
    TokenizerDecode,
    #[error("missing model tensor `{0}`")]
    MissingTensor(String),
    #[error("tensor `{0}` has an invalid byte length for its dtype")]
    InvalidTensorBytes(String),
    #[error("unsupported model: {0}")]
    Unsupported(String),
    #[error("invalid model configuration: {0}")]
    Config(String),
    #[error("KV cache capacity {capacity} exceeded by requested length {requested}")]
    CacheCapacity { capacity: usize, requested: usize },
}
