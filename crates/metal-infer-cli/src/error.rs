use metal_infer_kernels::GpuError;
use metal_infer_models::ModelError;
use metal_infer_runtime::{SubmitError, WorkerError};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CliError {
    #[error("I/O operation failed")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Worker(#[from] WorkerError),
    #[error("JSON serialization failed")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Server(#[from] ServerError),
    #[error("RUST_LOG filter is invalid")]
    LogFilter(#[source] tracing_subscriber::filter::ParseError),
    #[error("RUST_LOG is not valid Unicode")]
    LogEnvironment(#[source] std::env::VarError),
    #[error("cannot install the tracing subscriber")]
    TracingInit(#[source] tracing::subscriber::SetGlobalDefaultError),
    #[error("prompt and generated tokens exceed the context")]
    ContextExceeded,
    #[error("invalid test lengths, iterations, or profile mode (profile requires pg)")]
    InvalidBenchArguments,
    #[error("logits contain no finite value")]
    NoFiniteLogit,
    #[error("token index does not fit in u32")]
    TokenIdOverflow,
}

#[derive(Debug, Error)]
pub enum ServerError {
    #[error("cannot start the async runtime")]
    Runtime(#[source] std::io::Error),
    #[error("cannot bind the server address")]
    Bind(#[source] std::io::Error),
    #[error("HTTP server failed")]
    Http(#[source] std::io::Error),
}

#[derive(Debug, Error)]
pub enum RequestError {
    #[error("invalid completion request")]
    InvalidJson(#[source] serde_json::Error),
    #[error("requested model is not loaded")]
    ModelNotLoaded,
    #[error("message content part type is not supported")]
    UnsupportedContentPart,
    #[error("max_tokens exceeds the context")]
    MaxTokensExceedContext,
    #[error("request queue is full")]
    QueueFull,
    #[error("inference worker stopped")]
    WorkerStopped,
    #[error(transparent)]
    Worker(#[from] WorkerError),
}

impl From<SubmitError> for RequestError {
    fn from(error: SubmitError) -> Self {
        match error {
            SubmitError::QueueFull => Self::QueueFull,
            SubmitError::Stopped => Self::WorkerStopped,
        }
    }
}
