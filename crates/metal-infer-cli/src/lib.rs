mod error;

use std::io::Write;
use std::path::Path;

use clap::ValueEnum;
use metal_infer_kernels::{Kernels, MetalContext};
use metal_infer_models::Qwen3Model;
use metal_infer_planner::Plan;
use tracing_subscriber::EnvFilter;

pub use crate::error::{CliError, ServerError};

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum LogFormat {
    #[default]
    Text,
    Json,
}

pub fn init_tracing(format: LogFormat) -> Result<(), CliError> {
    let filter = match std::env::var("RUST_LOG") {
        Ok(value) => EnvFilter::try_new(value).map_err(CliError::LogFilter)?,
        Err(std::env::VarError::NotPresent) => {
            EnvFilter::new("warn,metal_infer=info,metal_infer_bench=info,metal_infer_cli=info")
        }
        Err(error) => return Err(CliError::LogEnvironment(error)),
    };
    match format {
        LogFormat::Text => tracing::subscriber::set_global_default(
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(std::io::stderr)
                .finish(),
        ),
        LogFormat::Json => tracing::subscriber::set_global_default(
            tracing_subscriber::fmt()
                .json()
                .with_current_span(true)
                .with_span_list(true)
                .with_env_filter(filter)
                .with_writer(std::io::stderr)
                .finish(),
        ),
    }
    .map_err(CliError::TracingInit)
}

pub fn load_model(
    model_path: &Path,
    context: &MetalContext,
    with: &[String],
) -> Result<Option<Qwen3Model>, CliError> {
    load_model_with(model_path, Kernels::new(context)?, with)
}

pub fn load_model_with(
    model_path: &Path,
    kernels: Kernels,
    with: &[String],
) -> Result<Option<Qwen3Model>, CliError> {
    let list = with.iter().any(|value| value == "list");
    let overrides = with
        .iter()
        .filter(|value| *value != "list")
        .cloned()
        .collect::<Vec<_>>();
    let model = Qwen3Model::load_with(model_path, kernels, &overrides)?;
    if list {
        print_plan(model.plan());
        return Ok(None);
    }
    Ok(Some(model))
}

pub fn print_plan(plan: &Plan) {
    let mut stdout = std::io::stdout().lock();
    for (key, value) in plan.entries() {
        let _ = stdout.write_all(format!("{key}={value}\n").as_bytes());
    }
}
