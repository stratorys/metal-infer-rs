use std::sync::atomic::{AtomicU64, Ordering};

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{Instrument, Span};

use crate::server::dto::{
    Admission, ChatCompletion, ChatCompletionRequest, Health, ModelList, unix_seconds,
};
use crate::server::error::RequestError;
use crate::server::sse::SseEncoder;
use crate::worker::{Event, Summary, WorkerError, WorkerHandle, WorkerInfo};

const MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone)]
struct HttpState {
    worker: WorkerHandle,
    info: WorkerInfo,
}

#[derive(Default)]
struct Collected {
    reasoning: Option<String>,
    content: String,
    outcome: Option<Result<Summary, WorkerError>>,
}

pub fn build(
    worker: WorkerHandle,
    info: WorkerInfo,
) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(completion))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        .with_state(HttpState { worker, info })
}

async fn health() -> Json<Health> {
    Json(Health::OK)
}

async fn models(State(state): State<HttpState>) -> Response {
    Json(ModelList::new(&state.info.model_id)).into_response()
}

async fn completion(
    State(state): State<HttpState>,
    body: Bytes,
) -> Result<Response, RequestError> {
    let id = completion_id();
    let span = tracing::info_span!(
        "completion",
        request_id = %id,
        stream = tracing::field::Empty
    );
    respond(state, body, id).instrument(span).await
}

async fn respond(
    state: HttpState,
    body: Bytes,
    id: String,
) -> Result<Response, RequestError> {
    let request: ChatCompletionRequest =
        serde_json::from_slice(&body).map_err(RequestError::InvalidJson)?;
    Span::current().record("stream", request.stream());
    let Admission {
        job,
        events,
        stream,
    } = request.into_admission(&state.info, id.clone(), Span::current())?;
    state.worker.try_submit(job)?;
    if stream {
        Ok(stream_response(events, id, state.info.model_id))
    } else {
        collect_response(events, id, state.info.model_id).await
    }
}

fn stream_response(
    events: mpsc::Receiver<Event>,
    id: String,
    model_id: String,
) -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
            (header::ACCESS_CONTROL_ALLOW_ORIGIN, "*"),
        ],
        Body::from_stream(SseEncoder::new(id, model_id).into_stream(events)),
    )
        .into_response()
}

async fn collect_response(
    events: mpsc::Receiver<Event>,
    id: String,
    model_id: String,
) -> Result<Response, RequestError> {
    let collected = ReceiverStream::new(events)
        .fold(Collected::default(), Collected::push)
        .await;
    match collected.outcome {
        Some(Ok(summary)) => Ok(Json(ChatCompletion::new(
            id,
            model_id,
            collected.reasoning,
            collected.content,
            summary,
        ))
        .into_response()),
        Some(Err(error)) => Err(RequestError::Worker(error)),
        None => Err(RequestError::WorkerStopped),
    }
}

impl Collected {
    fn push(
        mut self,
        event: Event,
    ) -> Self {
        match event {
            Event::Reasoning(text) => {
                self.reasoning.get_or_insert_default().push_str(&text);
            }
            Event::Content(text) => self.content.push_str(&text),
            Event::Finished(summary) => self.outcome = Some(Ok(summary)),
            Event::Failed(error) => self.outcome = Some(Err(error)),
        }
        self
    }
}

fn completion_id() -> String {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    format!(
        "chatcmpl-metal-{}-{}",
        unix_seconds(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    )
}
