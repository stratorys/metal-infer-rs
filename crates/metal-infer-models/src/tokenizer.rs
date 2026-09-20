use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::ModelError;

pub struct ModelTokenizer {
    inner: Tokenizer,
    eos_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ModelTokenizer {
    pub fn from_directory(directory: &Path) -> Result<Self, ModelError> {
        let path = directory.join("tokenizer.json");
        let inner = Tokenizer::from_file(&path).map_err(|_| ModelError::TokenizerLoad(path))?;
        let mut eos_token_ids = Vec::new();
        let tokenizer_config = directory.join("tokenizer_config.json");
        if tokenizer_config.exists() {
            let config: serde_json::Value = serde_json::from_slice(&fs::read(tokenizer_config)?)?;
            if let Some(token) = config.get("eos_token") {
                collect_special_tokens(token, &mut |value| {
                    if let Some(id) = inner.token_to_id(value)
                        && !eos_token_ids.contains(&id)
                    {
                        eos_token_ids.push(id);
                    }
                });
            }
        }
        Ok(Self {
            inner,
            eos_token_ids,
        })
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

    pub fn encode_chat(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Vec<u32>, ModelError> {
        let prompt = render_qwen_chat(messages)?;
        self.inner
            .encode(prompt, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|_| ModelError::TokenizerEncode)
    }

    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }
}

fn render_qwen_chat(messages: &[ChatMessage]) -> Result<String, ModelError> {
    if messages.is_empty() {
        return Err(ModelError::Config(
            "chat completion requires at least one message".into(),
        ));
    }
    let mut prompt = String::new();
    for message in messages {
        if !matches!(
            message.role.as_str(),
            "system" | "user" | "assistant" | "tool"
        ) {
            return Err(ModelError::Config(format!(
                "unsupported chat role `{}`",
                message.role
            )));
        }
        prompt.push_str("<|im_start|>");
        prompt.push_str(&message.role);
        prompt.push('\n');
        prompt.push_str(&message.content);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    Ok(prompt)
}

fn collect_special_tokens(
    value: &serde_json::Value,
    visitor: &mut impl FnMut(&str),
) {
    match value {
        serde_json::Value::String(token) => visitor(token),
        serde_json::Value::Array(tokens) => {
            for token in tokens {
                collect_special_tokens(token, visitor);
            }
        }
        serde_json::Value::Object(object) => {
            if let Some(serde_json::Value::String(token)) = object.get("content") {
                visitor(token);
            }
        }
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::{ChatMessage, render_qwen_chat};

    #[test]
    fn renders_qwen_chat_boundaries() {
        let prompt = render_qwen_chat(&[
            ChatMessage {
                role: "system".into(),
                content: "Be concise.".into(),
            },
            ChatMessage {
                role: "user".into(),
                content: "Hello".into(),
            },
        ])
        .expect("chat rendering should succeed");
        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
