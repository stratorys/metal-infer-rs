use metal_infer_kernels::GpuError;
use metal_infer_models::ModelError;
use metal_infer_runtime::ServerError;
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
    #[error("invalid test lengths, iterations, or profile mode (profile requires pg)")]
    InvalidBenchArguments,
    #[error("logits contain no finite value")]
    NoFiniteLogit,
    #[error("token index does not fit in u32")]
    TokenIdOverflow,
}
