use std::time::Instant;

use metal_infer_models::ChatMessage;
use tokio::sync::mpsc;
use tracing::Span;

use crate::worker::error::WorkerError;

pub struct Job {
    pub id: String,
    pub span: Span,
    pub received: Instant,
    pub messages: Vec<ChatMessage>,
    pub options: JobOptions,
    pub events: mpsc::Sender<Event>,
}

#[derive(Clone, Debug)]
pub struct JobOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub ignore_eos: bool,
    pub stop: Vec<String>,
}

impl JobOptions {
    pub const fn event_capacity(&self) -> usize {
        self.max_tokens.saturating_mul(2).saturating_add(3)
    }
}

#[derive(Debug)]
pub enum Event {
    Reasoning(String),
    Content(String),
    Finished(Summary),
    Failed(WorkerError),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Summary {
    pub reason: FinishReason,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}
