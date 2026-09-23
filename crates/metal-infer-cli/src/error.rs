use metal_infer_kernels::GpuError;
use metal_infer_models::ModelError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CliError {
    #[error("I/O operation failed")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Model(#[from] ModelError),
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
    #[error("--max-active-requests must be positive")]
    NoActiveRequests,
    #[error("requested model is not loaded")]
    ModelNotLoaded,
    #[error("message content part type is not supported")]
    UnsupportedContentPart,
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
    #[error("inference worker stopped before it was ready")]
    WorkerStopped(#[source] tokio::sync::oneshot::error::RecvError),
    #[error("inference worker panicked or was cancelled")]
    WorkerJoin(#[source] tokio::task::JoinError),
    #[error("model initialization failed")]
    ModelInitialization(#[source] Box<CliError>),
    #[error("the plan was listed, there is no model to serve")]
    PlanListed,
    #[error("invalid completion request")]
    InvalidJson(#[source] serde_json::Error),
    #[error("invalid completion request")]
    InvalidRequest(#[source] Box<CliError>),
    #[error("invalid generation options")]
    InvalidOptions(#[source] ModelError),
    #[error("stream flag does not match the reply channel")]
    StreamModeMismatch,
    #[error("cannot allocate KV cache")]
    CacheAllocation(#[source] ModelError),
    #[error("inference failed")]
    Inference(#[source] Box<CliError>),
    #[error("active completion has no token to decode")]
    MissingDecodeInput,
    #[error("client disconnected")]
    Disconnected,
    #[error("stream client is too slow")]
    SlowClient,
}
