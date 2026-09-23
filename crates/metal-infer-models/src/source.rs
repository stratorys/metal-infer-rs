use std::path::{Path, PathBuf};

use crate::ModelError;

const FALLBACK_MODEL_ID: &str = "qwen3";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelSource {
    pub directory: PathBuf,
    pub model_id: String,
}

impl ModelSource {
    pub fn resolve(model: &Path) -> Result<Self, ModelError> {
        let directory = resolve_directory(model)?;
        let model_id = hugging_face_model_id(model).unwrap_or_else(|| infer_model_id(&directory));
        Ok(Self {
            directory,
            model_id,
        })
    }
}

fn resolve_directory(model: &Path) -> Result<PathBuf, ModelError> {
    if model.is_dir() {
        return Ok(model.to_owned());
    }
    let identifier = model.to_str().ok_or(ModelError::NonUtf8ModelPath)?;
    if !is_hugging_face_id(identifier) {
        return Err(ModelError::ModelDirectoryMissing);
    }
    let cache = hugging_face_cache_root().ok_or(ModelError::HuggingFaceCacheMissing)?;
    let repository = cache.join(format!("models--{}", identifier.replace('/', "--")));
    let snapshots = repository.join("snapshots");
    let revision = std::fs::read_to_string(repository.join("refs/main"))
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty() && !value.contains('/') && !value.contains(".."));
    let snapshot = if let Some(revision) = revision {
        snapshots.join(revision)
    } else {
        newest_snapshot(&snapshots)?.ok_or(ModelError::ModelNotCached)?
    };
    if snapshot.join("config.json").is_file() && snapshot.join("tokenizer.json").is_file() {
        Ok(snapshot)
    } else {
        Err(ModelError::ModelNotCached)
    }
}

fn hugging_face_model_id(model: &Path) -> Option<String> {
    let value = model.to_str()?;
    is_hugging_face_id(value).then(|| value.to_owned())
}

fn is_hugging_face_id(value: &str) -> bool {
    let mut parts = value.split('/');
    matches!((parts.next(), parts.next(), parts.next()), (Some(owner), Some(name), None) if !owner.is_empty() && !name.is_empty())
}

fn infer_model_id(path: &Path) -> String {
    for component in path.ancestors().filter_map(Path::file_name) {
        let Some(name) = component.to_str() else {
            continue;
        };
        if let Some(repository) = name.strip_prefix("models--") {
            return repository.replace("--", "/");
        }
    }
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(FALLBACK_MODEL_ID)
        .to_owned()
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

fn newest_snapshot(directory: &Path) -> Result<Option<PathBuf>, ModelError> {
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{hugging_face_model_id, infer_model_id};

    #[test]
    fn recognizes_hugging_face_repository_ids() {
        assert_eq!(
            hugging_face_model_id(Path::new("Qwen/Qwen3-8B")),
            Some("Qwen/Qwen3-8B".into())
        );
        assert_eq!(hugging_face_model_id(Path::new("local-model")), None);
        assert_eq!(hugging_face_model_id(Path::new("a/b/c")), None);
    }

    #[test]
    fn infers_the_model_id_from_the_cache_layout() {
        assert_eq!(
            infer_model_id(Path::new(
                "/cache/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de2"
            )),
            "Qwen/Qwen3-0.6B"
        );
        assert_eq!(infer_model_id(Path::new("/models/my-qwen")), "my-qwen");
    }
}
