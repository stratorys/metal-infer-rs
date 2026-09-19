use std::path::Path;

use tokenizers::Tokenizer;

use crate::ModelError;

pub struct ModelTokenizer {
    inner: Tokenizer,
}

impl ModelTokenizer {
    pub fn from_directory(directory: &Path) -> Result<Self, ModelError> {
        let path = directory.join("tokenizer.json");
        let inner = Tokenizer::from_file(&path).map_err(|_| ModelError::TokenizerLoad(path))?;
        Ok(Self { inner })
    }

    pub fn encode(
        &self,
        text: &str,
    ) -> Result<Vec<u32>, ModelError> {
        self.inner
            .encode(text, true)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|_| ModelError::TokenizerEncode)
    }

    pub fn decode(
        &self,
        tokens: &[u32],
    ) -> Result<String, ModelError> {
        self.inner
            .decode(tokens, true)
            .map_err(|_| ModelError::TokenizerDecode)
    }
}
