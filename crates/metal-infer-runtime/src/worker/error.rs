use metal_infer_kernels::GpuError;
use metal_infer_models::ModelError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum WorkerError {
    #[error("cannot spawn the inference worker thread")]
    ThreadSpawn(#[source] std::io::Error),
    #[error("inference worker panicked")]
    Panicked,
    #[error("inference worker stopped before it was ready")]
    Stopped,
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error("cannot tokenize the chat prompt")]
    Tokenization(#[source] ModelError),
    #[error("prompt and generated tokens exceed the context")]
    ContextExceeded,
    #[error("invalid generation options")]
    InvalidOptions(#[source] ModelError),
    #[error("cannot allocate KV cache")]
    CacheAllocation(#[source] ModelError),
    #[error("inference failed")]
    Inference(#[source] ModelError),
    #[error("batched decode failed")]
    BatchFailed,
    #[error("active completion has no token to decode")]
    MissingDecodeInput,
    #[error("completion event channel is full")]
    EventOverflow,
}

impl WorkerError {
    pub const fn is_client_error(&self) -> bool {
        match self {
            Self::Tokenization(_) | Self::ContextExceeded | Self::InvalidOptions(_) => true,
            Self::ThreadSpawn(_)
            | Self::Panicked
            | Self::Stopped
            | Self::Model(_)
            | Self::Gpu(_)
            | Self::CacheAllocation(_)
            | Self::Inference(_)
            | Self::BatchFailed
            | Self::MissingDecodeInput
            | Self::EventOverflow => false,
        }
    }
}

#[derive(Debug, Error)]
pub enum SubmitError {
    #[error("request queue is full")]
    QueueFull,
    #[error("inference worker stopped")]
    Stopped,
}
