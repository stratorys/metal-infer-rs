use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use metal_infer_cli::{CliError, RequestError, ServerError};
use metal_infer_models::ChatMessage;
use metal_infer_runtime::{
    Event, FinishReason, Job, JobOptions, Summary, WorkerError, WorkerHandle, WorkerInfo,
    WorkerOptions,
};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::{Stream, StreamExt};
use tracing::{Instrument, Span};

const DEFAULT_MAX_TOKENS: usize = 256;
const MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;
const INTERNAL_ERROR_MESSAGE: &str = "internal server error";

pub struct ServerOptions {
    pub worker: WorkerOptions,
    pub bind: String,
}

#[derive(Clone)]
struct HttpState {
    worker: WorkerHandle,
    info: WorkerInfo,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
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

struct Admission {
    job: Job,
    events: mpsc::Receiver<Event>,
    stream: bool,
}

struct Submitted {
    events: mpsc::Receiver<Event>,
    stream: bool,
}

struct ClientError {
    status: StatusCode,
    message: String,
}

struct SseEncoder {
    id: String,
    model_id: String,
    role_sent: bool,
}

#[derive(Default)]
struct Collected {
    reasoning: Option<String>,
    content: String,
    outcome: Option<Result<Summary, WorkerError>>,
}

pub fn serve(options: ServerOptions) -> Result<(), CliError> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(ServerError::Runtime)?;
    let worker = runtime.block_on(metal_infer_runtime::spawn(options.worker))?;
    let state = HttpState {
        worker: worker.handle(),
        info: worker.info().clone(),
    };
    let served = runtime.block_on(listen(options.bind, state));
    let joined = runtime.block_on(worker.join());
    drop(runtime);
    served.and(joined.map_err(CliError::from))
}

async fn listen(
    bind: String,
    state: HttpState,
) -> Result<(), CliError> {
    let router = Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(completion))
        .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .map_err(ServerError::Bind)?;
    tracing::info!(message = "OpenAI-compatible server listening.", %bind);
    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(ServerError::Http)?;
    tracing::info!(message = "HTTP server stopped.");
    Ok(())
}

async fn shutdown_signal() {
    match tokio::signal::ctrl_c().await {
        Ok(()) => tracing::info!(message = "Shutdown requested."),
        Err(error) => {
            tracing::error!(message = "Cannot listen for the shutdown signal.", %error);
            std::future::pending::<()>().await;
        }
    }
}

async fn models(State(state): State<HttpState>) -> Json<Value> {
    Json(
        json!({"object": "list", "data": [{"id": state.info.model_id, "object": "model", "owned_by": "metal-infer"}]}),
    )
}

async fn completion(
    State(state): State<HttpState>,
    Json(body): Json<Value>,
) -> Response {
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
    body: Value,
    id: String,
) -> Response {
    match submit(&state, body, id.clone()) {
        Ok(Submitted {
            events,
            stream: true,
        }) => stream_response(events, id, state.info.model_id),
        Ok(Submitted {
            events,
            stream: false,
        }) => collect_response(events, id, state.info.model_id).await,
        Err(error) => reject(&error),
    }
}

fn submit(
    state: &HttpState,
    body: Value,
    id: String,
) -> Result<Submitted, RequestError> {
    let request: ChatCompletionRequest =
        serde_json::from_value(body).map_err(RequestError::InvalidJson)?;
    Span::current().record("stream", request.stream);
    let Admission {
        job,
        events,
        stream,
    } = request.into_admission(&state.info, id, Span::current())?;
    state.worker.try_submit(job)?;
    Ok(Submitted { events, stream })
}

fn stream_response(
    events: mpsc::Receiver<Event>,
    id: String,
    model_id: String,
) -> Response {
    let encoder = SseEncoder {
        id,
        model_id,
        role_sent: false,
    };
    (
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
            (header::ACCESS_CONTROL_ALLOW_ORIGIN, "*"),
        ],
        Body::from_stream(encoder.into_stream(events)),
    )
        .into_response()
}

async fn collect_response(
    events: mpsc::Receiver<Event>,
    id: String,
    model_id: String,
) -> Response {
    let collected = ReceiverStream::new(events)
        .fold(Collected::default(), Collected::push)
        .await;
    match collected.outcome {
        Some(Ok(summary)) => Json(completion_body(
            &id,
            &model_id,
            collected.reasoning,
            collected.content,
            summary,
        ))
        .into_response(),
        Some(Err(error)) => client_response(&ClientError::from_worker(&error)),
        None => reject(&RequestError::WorkerStopped),
    }
}

fn reject(error: &RequestError) -> Response {
    match error {
        RequestError::QueueFull | RequestError::WorkerStopped => {
            tracing::error!(message = "Completion request rejected.", %error);
        }
        RequestError::InvalidJson(_)
        | RequestError::ModelNotLoaded
        | RequestError::UnsupportedContentPart
        | RequestError::MaxTokensExceedContext
        | RequestError::Worker(_) => {}
    }
    client_response(&ClientError::from_request(error))
}

fn client_response(error: &ClientError) -> Response {
    (error.status, Json(error_body(error))).into_response()
}

fn error_body(error: &ClientError) -> Value {
    let kind = if error.status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    json!({"error": {"message": error.message, "type": kind}})
}

impl ClientError {
    fn from_request(error: &RequestError) -> Self {
        let status = match error {
            RequestError::InvalidJson(_)
            | RequestError::ModelNotLoaded
            | RequestError::UnsupportedContentPart
            | RequestError::MaxTokensExceedContext => StatusCode::BAD_REQUEST,
            RequestError::QueueFull => StatusCode::SERVICE_UNAVAILABLE,
            RequestError::WorkerStopped => StatusCode::INTERNAL_SERVER_ERROR,
            RequestError::Worker(error) => return Self::from_worker(error),
        };
        Self::new(status, error)
    }

    fn from_worker(error: &WorkerError) -> Self {
        let status = if error.is_client_error() {
            StatusCode::BAD_REQUEST
        } else {
            StatusCode::INTERNAL_SERVER_ERROR
        };
        Self::new(status, error)
    }

    fn new(
        status: StatusCode,
        error: &impl ToString,
    ) -> Self {
        let message = if status.is_server_error() {
            INTERNAL_ERROR_MESSAGE.to_owned()
        } else {
            error.to_string()
        };
        Self { status, message }
    }
}

impl SseEncoder {
    fn into_stream(
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
            Event::Reasoning(text) => {
                self.with_role(vec![self.delta_chunk("reasoning_content", text)])
            }
            Event::Content(text) => self.with_role(vec![self.delta_chunk("content", text)]),
            Event::Finished(summary) => {
                self.with_role(vec![self.final_chunk(summary), "[DONE]".to_owned()])
            }
            Event::Failed(error) => vec![error_body(&ClientError::from_worker(&error)).to_string()],
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
        std::iter::once(self.role_chunk()).chain(frames).collect()
    }

    fn role_chunk(&self) -> String {
        json!({"id": self.id, "object": "chat.completion.chunk", "created": unix_seconds(), "model": self.model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]})
        .to_string()
    }

    fn delta_chunk(
        &self,
        field: &str,
        value: String,
    ) -> String {
        json!({"id": self.id, "object": "chat.completion.chunk", "created": unix_seconds(), "model": self.model_id,
            "choices": [{"index": 0, "delta": {field: value}, "finish_reason": null}]})
        .to_string()
    }

    fn final_chunk(
        &self,
        summary: Summary,
    ) -> String {
        json!({"id": self.id, "object": "chat.completion.chunk", "created": unix_seconds(), "model": self.model_id,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason(summary.reason)}],
            "usage": usage(summary)})
        .to_string()
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

fn completion_body(
    id: &str,
    model_id: &str,
    reasoning: Option<String>,
    content: String,
    summary: Summary,
) -> Value {
    let message = match reasoning {
        Some(reasoning) => {
            json!({"role": "assistant", "content": content.trim(), "reasoning_content": reasoning.trim()})
        }
        None => json!({"role": "assistant", "content": content}),
    };
    json!({"id": id, "object": "chat.completion", "created": unix_seconds(), "model": model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(summary.reason)}],
        "usage": usage(summary)})
}

const fn finish_reason(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

fn usage(summary: Summary) -> Value {
    json!({"prompt_tokens": summary.prompt_tokens, "completion_tokens": summary.completion_tokens,
        "total_tokens": summary.prompt_tokens.saturating_add(summary.completion_tokens)})
}

impl ChatCompletionRequest {
    fn into_admission(
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

const fn default_top_p() -> f32 {
    1.0
}

fn completion_id() -> String {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    format!(
        "chatcmpl-metal-{}-{}",
        unix_seconds(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    )
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}
