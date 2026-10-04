use std::time::{Instant, SystemTime, UNIX_EPOCH};

use metal_infer_models::ChatMessage;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::Span;

use crate::server::error::RequestError;
use crate::worker::{Event, FinishReason, Job, JobOptions, Summary, WorkerInfo};

const DEFAULT_MAX_TOKENS: usize = 256;
const ASSISTANT_ROLE: &str = "assistant";
const INTERNAL_ERROR_MESSAGE: &str = "internal server error";

#[derive(Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<WireMessage>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    max_completion_tokens: Option<usize>,
    #[serde(default)]
    temperature: f32,
    #[serde(default = "default_top_p")]
    top_p: f32,
    #[serde(default)]
    top_k: usize,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<StopSequences>,
    #[serde(default)]
    ignore_eos: bool,
}

#[derive(Deserialize)]
struct WireMessage {
    role: String,
    content: MessageContent,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StopSequences {
    One(String),
    Many(Vec<String>),
}

pub struct Admission {
    pub job: Job,
    pub events: mpsc::Receiver<Event>,
    pub stream: bool,
}

#[derive(Serialize)]
pub struct ChatCompletion {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: [Choice; 1],
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    message: ResponseMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct ResponseMessage {
    role: &'static str,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<String>,
}

#[derive(Serialize)]
pub struct ChatCompletionChunk<'text> {
    id: &'text str,
    object: &'static str,
    created: u64,
    model: &'text str,
    choices: [ChunkChoice<'text>; 1],
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Serialize)]
struct ChunkChoice<'text> {
    index: u32,
    delta: Delta<'text>,
    finish_reason: Option<&'static str>,
}

#[derive(Default, Serialize)]
pub struct Delta<'text> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'text str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_content: Option<&'text str>,
}

#[derive(Clone, Copy, Serialize)]
pub struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Serialize)]
pub struct ModelList<'text> {
    object: &'static str,
    data: [ModelCard<'text>; 1],
}

#[derive(Serialize)]
struct ModelCard<'text> {
    id: &'text str,
    object: &'static str,
    owned_by: &'static str,
}

#[derive(Serialize)]
pub struct Health {
    status: &'static str,
}

#[derive(Serialize)]
pub struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

impl ChatCompletionRequest {
    pub fn into_admission(
        self,
        info: &WorkerInfo,
        id: String,
        span: Span,
    ) -> Result<Admission, RequestError> {
        let max_tokens = self.max_tokens();
        if self
            .model
            .as_ref()
            .is_some_and(|model| model != &info.model_id)
        {
            return Err(RequestError::ModelNotLoaded);
        }
        if max_tokens > info.context {
            return Err(RequestError::MaxTokensExceedContext);
        }
        let messages = self
            .messages
            .into_iter()
            .map(WireMessage::into_chat_message)
            .collect::<Result<Vec<_>, _>>()?;
        let options = JobOptions {
            max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            seed: self.seed,
            ignore_eos: self.ignore_eos,
            stop: self.stop.map_or_else(Vec::new, StopSequences::into_vec),
        };
        let (sender, events) = mpsc::channel(options.event_capacity());
        Ok(Admission {
            job: Job {
                id,
                span,
                received: Instant::now(),
                messages,
                options,
                events: sender,
            },
            events,
            stream: self.stream,
        })
    }

    pub const fn stream(&self) -> bool {
        self.stream
    }

    fn max_tokens(&self) -> usize {
        self.max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS)
    }
}

impl WireMessage {
    fn into_chat_message(self) -> Result<ChatMessage, RequestError> {
        let content = match self.content {
            MessageContent::Text(text) => text,
            MessageContent::Parts(parts) => parts
                .into_iter()
                .map(ContentPart::into_text)
                .collect::<Result<String, _>>()?,
        };
        Ok(ChatMessage {
            role: self.role,
            content,
        })
    }
}

impl ContentPart {
    fn into_text(self) -> Result<String, RequestError> {
        if self.kind == "text" {
            Ok(self.text.unwrap_or_default())
        } else {
            Err(RequestError::UnsupportedContentPart)
        }
    }
}

impl StopSequences {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

impl ChatCompletion {
    pub fn new(
        id: String,
        model: String,
        reasoning: Option<String>,
        content: String,
        summary: Summary,
    ) -> Self {
        let message = match reasoning {
            Some(reasoning) => ResponseMessage {
                role: ASSISTANT_ROLE,
                content: content.trim().to_owned(),
                reasoning_content: Some(reasoning.trim().to_owned()),
            },
            None => ResponseMessage {
                role: ASSISTANT_ROLE,
                content,
                reasoning_content: None,
            },
        };
        Self {
            id,
            object: "chat.completion",
            created: unix_seconds(),
            model,
            choices: [Choice {
                index: 0,
                message,
                finish_reason: finish_reason(summary.reason),
            }],
            usage: Usage::from(summary),
        }
    }
}

impl<'text> ChatCompletionChunk<'text> {
    pub fn delta(
        id: &'text str,
        model: &'text str,
        delta: Delta<'text>,
    ) -> Self {
        Self::new(id, model, delta, None, None)
    }

    pub fn finished(
        id: &'text str,
        model: &'text str,
        summary: Summary,
    ) -> Self {
        Self::new(
            id,
            model,
            Delta::default(),
            Some(finish_reason(summary.reason)),
            Some(Usage::from(summary)),
        )
    }

    fn new(
        id: &'text str,
        model: &'text str,
        delta: Delta<'text>,
        finish_reason: Option<&'static str>,
        usage: Option<Usage>,
    ) -> Self {
        Self {
            id,
            object: "chat.completion.chunk",
            created: unix_seconds(),
            model,
            choices: [ChunkChoice {
                index: 0,
                delta,
                finish_reason,
            }],
            usage,
        }
    }
}

impl<'text> Delta<'text> {
    pub fn role() -> Self {
        Self {
            role: Some(ASSISTANT_ROLE),
            ..Self::default()
        }
    }

    pub fn reasoning(text: &'text str) -> Self {
        Self {
            reasoning_content: Some(text),
            ..Self::default()
        }
    }

    pub fn content(text: &'text str) -> Self {
        Self {
            content: Some(text),
            ..Self::default()
        }
    }
}

impl From<Summary> for Usage {
    fn from(summary: Summary) -> Self {
        Self {
            prompt_tokens: summary.prompt_tokens,
            completion_tokens: summary.completion_tokens,
            total_tokens: summary
                .prompt_tokens
                .saturating_add(summary.completion_tokens),
        }
    }
}

impl<'text> ModelList<'text> {
    pub const fn new(model_id: &'text str) -> Self {
        Self {
            object: "list",
            data: [ModelCard {
                id: model_id,
                object: "model",
                owned_by: "metal-infer",
            }],
        }
    }
}

impl Health {
    pub const OK: Self = Self { status: "ok" };
}

impl ErrorBody {
    pub fn from_error(error: &RequestError) -> Self {
        let error = if error.status().is_server_error() {
            ErrorDetail {
                message: INTERNAL_ERROR_MESSAGE.to_owned(),
                kind: "server_error",
            }
        } else {
            ErrorDetail {
                message: error.to_string(),
                kind: "invalid_request_error",
            }
        };
        Self { error }
    }
}

const fn finish_reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

const fn default_top_p() -> f32 {
    1.0
}

pub fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use tracing::Span;

    use super::{Admission, ChatCompletion, ChatCompletionRequest};
    use crate::server::error::RequestError;
    use crate::worker::{FinishReason, Summary, WorkerInfo};

    const SUMMARY: Summary = Summary {
        reason: FinishReason::Length,
        prompt_tokens: 7,
        completion_tokens: 5,
    };

    fn admit(body: &str) -> Result<Admission, RequestError> {
        serde_json::from_str::<ChatCompletionRequest>(body)
            .expect("request parses")
            .into_admission(
                &WorkerInfo {
                    model_id: "qwen".to_owned(),
                    context: 1024,
                },
                "id".to_owned(),
                Span::none(),
            )
    }

    fn body(completion: &ChatCompletion) -> Value {
        serde_json::to_value(completion).expect("completion serializes")
    }

    #[test]
    fn max_completion_tokens_wins_over_max_tokens() {
        let admission = admit(
            r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":8,"max_completion_tokens":4}"#,
        )
        .expect("admitted");
        assert_eq!(
            admission.job.options.max_tokens, 4,
            "max_completion_tokens must take precedence"
        );
    }

    #[test]
    fn max_tokens_defaults_to_256() {
        let admission =
            admit(r#"{"messages":[{"role":"user","content":"hi"}]}"#).expect("admitted");
        assert_eq!(
            admission.job.options.max_tokens, 256,
            "a request without max_tokens must default to 256"
        );
    }

    #[test]
    fn stop_accepts_a_string_or_a_list() {
        let one =
            admit(r#"{"messages":[{"role":"user","content":"hi"}],"stop":"a"}"#).expect("admitted");
        let many = admit(r#"{"messages":[{"role":"user","content":"hi"}],"stop":["a","b"]}"#)
            .expect("admitted");
        assert_eq!(
            one.job.options.stop,
            vec!["a".to_owned()],
            "a single stop string must become a one-element list"
        );
        assert_eq!(
            many.job.options.stop,
            vec!["a".to_owned(), "b".to_owned()],
            "a stop list must be kept"
        );
    }

    #[test]
    fn stream_and_ignore_eos_are_forwarded() {
        let admission = admit(
            r#"{"messages":[{"role":"user","content":"hi"}],"stream":true,"ignore_eos":true}"#,
        )
        .expect("admitted");
        assert!(admission.stream, "stream must be forwarded");
        assert!(
            admission.job.options.ignore_eos,
            "ignore_eos must be forwarded"
        );
    }

    #[test]
    fn text_parts_are_concatenated() {
        let admission = admit(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}"#,
        )
        .expect("admitted");
        assert_eq!(
            admission
                .job
                .messages
                .first()
                .map(|message| message.content.as_str()),
            Some("ab"),
            "text parts must be concatenated"
        );
    }

    #[test]
    fn unknown_model_is_rejected() {
        assert!(
            matches!(
                admit(r#"{"model":"other","messages":[{"role":"user","content":"hi"}]}"#),
                Err(RequestError::ModelNotLoaded)
            ),
            "a request for another model must be rejected"
        );
    }

    #[test]
    fn non_text_part_is_rejected() {
        assert!(
            matches!(
                admit(r#"{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}"#),
                Err(RequestError::UnsupportedContentPart)
            ),
            "a non-text content part must be rejected"
        );
    }

    #[test]
    fn max_tokens_above_the_context_is_rejected() {
        assert!(
            matches!(
                admit(r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":2048}"#),
                Err(RequestError::MaxTokensExceedContext)
            ),
            "max_tokens above the context must be rejected before the queue"
        );
    }

    #[test]
    fn completion_trims_reasoning_and_content() {
        let value = body(&ChatCompletion::new(
            "id".to_owned(),
            "qwen".to_owned(),
            Some("\nplan\n".to_owned()),
            "\n\nanswer".to_owned(),
            SUMMARY,
        ));
        assert_eq!(
            value.pointer("/choices/0/message/reasoning_content"),
            Some(&Value::from("plan")),
            "reasoning must be trimmed"
        );
        assert_eq!(
            value.pointer("/choices/0/message/content"),
            Some(&Value::from("answer")),
            "content after a think block must be trimmed"
        );
    }

    #[test]
    fn completion_without_reasoning_omits_the_field() {
        let value = body(&ChatCompletion::new(
            "id".to_owned(),
            "qwen".to_owned(),
            None,
            " answer ".to_owned(),
            SUMMARY,
        ));
        assert_eq!(
            value.pointer("/choices/0/message/reasoning_content"),
            None,
            "reasoning_content must be omitted without a think block"
        );
        assert_eq!(
            value.pointer("/choices/0/message/content"),
            Some(&Value::from(" answer ")),
            "content without a think block must be kept as is"
        );
    }

    #[test]
    fn usage_totals_prompt_and_completion() {
        let value = body(&ChatCompletion::new(
            "id".to_owned(),
            "qwen".to_owned(),
            None,
            String::new(),
            SUMMARY,
        ));
        assert_eq!(
            value.pointer("/usage/total_tokens"),
            Some(&Value::from(12)),
            "total_tokens must be the sum of prompt and completion tokens"
        );
        assert_eq!(
            value.pointer("/choices/0/finish_reason"),
            Some(&Value::from("length")),
            "the finish reason must be serialized"
        );
    }
}
