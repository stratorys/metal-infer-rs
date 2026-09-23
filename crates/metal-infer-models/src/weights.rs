use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use metal_infer_kernels::{MetalContext, Tensor};
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;

use crate::ModelError;

pub(crate) struct WeightMap {
    tensors: HashMap<String, Tensor>,
}

#[derive(Deserialize)]
struct SafetensorIndex {
    weight_map: HashMap<String, String>,
}

impl WeightMap {
    pub(crate) fn load(
        directory: &Path,
        context: &MetalContext,
    ) -> Result<Self, ModelError> {
        let files = tensor_files(directory)?;
        let mut tensors = HashMap::new();
        for file in files {
            let bytes = fs::read(&file)?;
            let safetensors = SafeTensors::deserialize(&bytes)?;
            for name in safetensors.names() {
                let view = safetensors.tensor(name)?;
                let tensor = if view.dtype() == Dtype::F16 {
                    context.tensor_f16_bytes(view.data(), view.shape())?
                } else if view.dtype() == Dtype::BF16 {
                    context.tensor_bf16_as_f16_bytes(view.data(), view.shape())?
                } else {
                    return Err(ModelError::UnsupportedTensorDType);
                };
                tensors.insert(name.to_owned(), tensor);
            }
        }
        Ok(Self { tensors })
    }

    pub(crate) fn take(
        &mut self,
        name: &str,
    ) -> Result<Tensor, ModelError> {
        self.tensors.remove(name).ok_or(ModelError::MissingTensor)
    }
}

fn tensor_files(directory: &Path) -> Result<Vec<PathBuf>, ModelError> {
    let index_path = directory.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: SafetensorIndex = serde_json::from_slice(&fs::read(index_path)?)?;
        let mut names: Vec<_> = index.weight_map.into_values().collect();
        names.sort();
        names.dedup();
        return Ok(names.into_iter().map(|name| directory.join(name)).collect());
    }
    let single = directory.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    Err(ModelError::MissingWeights)
}
