use std::path::{Path, PathBuf};

use metal_infer_kernels::Kernels;
use metal_infer_models::{ModelError, Qwen3Model};
use metal_infer_planner::Plan;
use metal_infer_runtime::{CoreError, MetalContext};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CliError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error(transparent)]
    Model(#[from] ModelError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error("invalid command-line arguments: {0}")]
    InvalidArguments(String),
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
        print_plan(&model.plan());
        return Ok(None);
    }
    Ok(Some(model))
}

pub fn print_plan(plan: &Plan) {
    for (key, value) in plan.entries() {
        println!("{key}={value}");
    }
}

pub fn resolve_model_path(model: &Path) -> Result<PathBuf, CliError> {
    if model.is_dir() {
        return Ok(model.to_owned());
    }
    let identifier = model
        .to_str()
        .ok_or_else(|| CliError::InvalidArguments("model path is not valid UTF-8".into()))?;
    if !is_hugging_face_id(identifier) {
        return Err(CliError::InvalidArguments(format!(
            "model directory `{}` does not exist",
            model.display()
        )));
    }
    let cache = hugging_face_cache_root().ok_or_else(|| {
        CliError::InvalidArguments(
            "cannot locate the Hugging Face cache because HOME, HF_HOME, and HUGGINGFACE_HUB_CACHE are unset"
                .into(),
        )
    })?;
    let repository = cache.join(format!("models--{}", identifier.replace('/', "--")));
    let snapshots = repository.join("snapshots");
    let revision = std::fs::read_to_string(repository.join("refs/main"))
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty() && !value.contains('/') && !value.contains(".."));
    let snapshot = if let Some(revision) = revision {
        snapshots.join(revision)
    } else {
        newest_snapshot(&snapshots)?.ok_or_else(|| missing_cached_model(identifier))?
    };
    if snapshot.join("config.json").is_file() && snapshot.join("tokenizer.json").is_file() {
        Ok(snapshot)
    } else {
        Err(missing_cached_model(identifier))
    }
}

pub fn hugging_face_model_id(model: &Path) -> Option<String> {
    let value = model.to_str()?;
    is_hugging_face_id(value).then(|| value.to_owned())
}

fn is_hugging_face_id(value: &str) -> bool {
    let mut parts = value.split('/');
    matches!((parts.next(), parts.next(), parts.next()), (Some(owner), Some(name), None) if !owner.is_empty() && !name.is_empty())
}

fn hugging_face_cache_root() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("HUGGINGFACE_HUB_CACHE") {
        return Some(PathBuf::from(path));
    }
    if let Some(path) = std::env::var_os("HF_HOME") {
        return Some(PathBuf::from(path).join("hub"));
    }
    std::env::var_os("HOME").map(|path| PathBuf::from(path).join(".cache/huggingface/hub"))
}

fn newest_snapshot(directory: &Path) -> Result<Option<PathBuf>, CliError> {
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mut snapshots = entries
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .collect::<Vec<_>>();
    snapshots.sort_by_key(|entry| {
        entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
    });
    Ok(snapshots.pop().map(|entry| entry.path()))
}

fn missing_cached_model(identifier: &str) -> CliError {
    CliError::InvalidArguments(format!(
        "Hugging Face model `{identifier}` is not available in the local cache; download it with `hf download {identifier}`"
    ))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::hugging_face_model_id;

    #[test]
    fn recognizes_hugging_face_repository_ids() {
        assert_eq!(
            hugging_face_model_id(Path::new("Qwen/Qwen3-8B")),
            Some("Qwen/Qwen3-8B".into())
        );
        assert_eq!(hugging_face_model_id(Path::new("local-model")), None);
        assert_eq!(hugging_face_model_id(Path::new("a/b/c")), None);
    }
}
