use std::path::Path;

use metal_infer_kernels::{Kernels, MetalContext};
use metal_infer_models::{ModelSource, ModelTokenizer, Qwen3Model};

use crate::worker::WorkerInfo;
use crate::worker::error::WorkerError;

pub struct Engine {
    pub model_id: String,
    pub context: usize,
    pub tokenizer: ModelTokenizer,
    pub model: Qwen3Model,
}

impl Engine {
    pub fn load(
        model: &Path,
        model_id: Option<String>,
        context: usize,
        with: &[String],
    ) -> Result<Self, WorkerError> {
        let source = ModelSource::resolve(model)?;
        let metal = MetalContext::new()?;
        tracing::info!(message = "Metal device ready.", device = %metal.device_name());
        tracing::info!(message = "Loading model.", path = %source.directory.display());
        let tokenizer = ModelTokenizer::from_directory(&source.directory)?;
        let model = Qwen3Model::load_with(&source.directory, Kernels::new(&metal)?, with)?;
        Ok(Self {
            model_id: model_id.unwrap_or(source.model_id),
            context,
            tokenizer,
            model,
        })
    }

    pub fn info(&self) -> WorkerInfo {
        WorkerInfo {
            model_id: self.model_id.clone(),
            context: self.context,
        }
    }
}
