use metal_infer_core::CoreError;
use metal_infer_models::ModelError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CliError {
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("invalid command-line arguments: {0}")]
    InvalidArguments(String),
}
