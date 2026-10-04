use std::convert::Infallible;

use axum::body::Bytes;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt};

use crate::server::dto::{ChatCompletionChunk, Delta, ErrorBody};
use crate::server::error::RequestError;
use crate::worker::Event;

const DONE: &str = "[DONE]";
const SERIALIZATION_FAILED: &str =
    r#"{"error":{"message":"internal server error","type":"server_error"}}"#;

pub struct SseEncoder {
    id: String,
    model_id: String,
    role_sent: bool,
}

impl SseEncoder {
    pub const fn new(
        id: String,
        model_id: String,
    ) -> Self {
        Self {
            id,
            model_id,
            role_sent: false,
        }
    }

    pub fn into_stream(
        mut self,
        events: mpsc::Receiver<Event>,
    ) -> impl Stream<Item = Result<Bytes, Infallible>> {
        ReceiverStream::new(events).map(move |event| Ok(self.encode(event)))
    }

    fn encode(
        &mut self,
        event: Event,
    ) -> Bytes {
        let frames = match event {
            Event::Reasoning(text) => self.with_role(vec![json(&ChatCompletionChunk::delta(
                &self.id,
                &self.model_id,
                Delta::reasoning(&text),
            ))]),
            Event::Content(text) => self.with_role(vec![json(&ChatCompletionChunk::delta(
                &self.id,
                &self.model_id,
                Delta::content(&text),
            ))]),
            Event::Finished(summary) => self.with_role(vec![
                json(&ChatCompletionChunk::finished(
                    &self.id,
                    &self.model_id,
                    summary,
                )),
                DONE.to_owned(),
            ]),
            Event::Failed(error) => vec![json(&ErrorBody::from_error(&RequestError::from(error)))],
        };
        Bytes::from(
            frames
                .iter()
                .map(|frame| format!("data: {frame}\n\n"))
                .collect::<String>(),
        )
    }

    fn with_role(
        &mut self,
        frames: Vec<String>,
    ) -> Vec<String> {
        if self.role_sent {
            return frames;
        }
        self.role_sent = true;
        std::iter::once(json(&ChatCompletionChunk::delta(
            &self.id,
            &self.model_id,
            Delta::role(),
        )))
        .chain(frames)
        .collect()
    }
}

fn json(value: &impl Serialize) -> String {
    serde_json::to_string(value).unwrap_or_else(|error| {
        tracing::error!(message = "Cannot serialize a stream frame.", %error);
        SERIALIZATION_FAILED.to_owned()
    })
}

#[cfg(test)]
mod tests {
    use super::SseEncoder;
    use crate::worker::{Event, FinishReason, Summary, WorkerError};

    fn encode(events: Vec<Event>) -> Vec<Vec<String>> {
        encode_with(SseEncoder::new("id".to_owned(), "qwen".to_owned()), events)
    }

    fn encode_with(
        mut encoder: SseEncoder,
        events: Vec<Event>,
    ) -> Vec<Vec<String>> {
        events
            .into_iter()
            .map(|event| {
                String::from_utf8(encoder.encode(event).to_vec())
                    .expect("frames are UTF-8")
                    .split_terminator("\n\n")
                    .map(|frame| frame.strip_prefix("data: ").unwrap_or(frame).to_owned())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn first_delta_is_preceded_by_the_role() {
        let encoded = encode(vec![
            Event::Content("a".to_owned()),
            Event::Reasoning("b".to_owned()),
        ]);
        let [first, second] = encoded.as_slice() else {
            panic!("expected two encoded events, got {encoded:?}");
        };
        assert_eq!(first.len(), 2, "the first delta must carry the role frame");
        assert!(
            first
                .first()
                .is_some_and(|frame| frame.contains(r#""role":"assistant""#)),
            "the role frame must come first"
        );
        assert!(
            first
                .last()
                .is_some_and(|frame| frame.contains(r#""content":"a""#)),
            "the content delta must follow the role"
        );
        assert_eq!(
            second.len(),
            1,
            "later deltas must not repeat the role frame"
        );
        assert!(
            second
                .first()
                .is_some_and(|frame| frame.contains(r#""reasoning_content":"b""#)),
            "a reasoning event must become a reasoning_content delta"
        );
    }

    #[test]
    fn finished_emits_usage_then_done() {
        let encoded = encode(vec![Event::Finished(Summary {
            reason: FinishReason::Stop,
            prompt_tokens: 3,
            completion_tokens: 2,
        })]);
        let finished = encoded.concat();
        assert_eq!(
            finished.last().map(String::as_str),
            Some("[DONE]"),
            "the stream must end with [DONE]"
        );
        assert!(
            finished.iter().rev().nth(1).is_some_and(|frame| {
                frame.contains(r#""finish_reason":"stop""#) && frame.contains(r#""total_tokens":5"#)
            }),
            "the final chunk must carry the finish reason and usage"
        );
    }

    #[test]
    fn client_failure_keeps_its_message() {
        let failed = encode(vec![Event::Failed(WorkerError::ContextExceeded)]).concat();
        assert!(
            failed.first().is_some_and(|frame| {
                frame.contains("invalid_request_error")
                    && frame.contains("prompt and generated tokens exceed the context")
            }),
            "a client failure must return its static message"
        );
    }

    #[test]
    fn server_failure_hides_its_message() {
        let failed = encode(vec![Event::Failed(WorkerError::BatchFailed)]).concat();
        assert!(
            failed.first().is_some_and(|frame| {
                frame.contains("server_error") && frame.contains("internal server error")
            }),
            "a server failure must not expose its message"
        );
    }
}
