use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use metal_infer_cli::{
    CliError, ServerError, hugging_face_model_id, load_model, resolve_model_path,
};
use metal_infer_models::{
    ChatMessage, GenerationOptions, KvCache, ModelTokenizer, Qwen3Model, TokenSampler,
};
use metal_infer_runtime::MetalContext;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::{mpsc as async_mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tracing::Instrument;

const DEFAULT_MAX_TOKENS: usize = 256;
const MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;
const COMMAND_CHANNEL_CAPACITY: usize = 16;
const WAITING_CAPACITY: usize = 16;
const STREAM_CHANNEL_CAPACITY: usize = 64;

pub struct ServerOptions {
    pub model: PathBuf,
    pub model_id: Option<String>,
    pub bind: String,
    pub context: usize,
    pub max_active_requests: usize,
    pub with: Vec<String>,
}

struct ServerState {
    model_id: String,
    tokenizer: ModelTokenizer,
    model: Qwen3Model,
    context: usize,
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

struct PreparedCompletion {
    prompt: Vec<u32>,
    options: GenerationOptions,
    stop_sequences: Vec<String>,
    stream: bool,
}

#[derive(Clone)]
struct HttpState {
    model_id: String,
    commands: async_mpsc::Sender<Command>,
}

struct Command {
    id: String,
    span: tracing::Span,
    received: Instant,
    body: Value,
    reply: Reply,
}

enum Reply {
    Stream(async_mpsc::Sender<Result<Bytes, std::convert::Infallible>>),
    Once(oneshot::Sender<Result<Value, ServerError>>),
}

impl Reply {
    fn error(
        self,
        error: ServerError,
    ) {
        match self {
            Self::Stream(sender) => {
                let value = error_body(error_status(&error), error.message());
                let _ = send_event(&sender, &value);
            }
            Self::Once(sender) => {
                let _ = sender.send(Err(error));
            }
        }
    }
    fn is_closed(&self) -> bool {
        match self {
            Self::Stream(sender) => sender.is_closed(),
            Self::Once(sender) => sender.is_closed(),
        }
    }
}

struct ActiveCompletion {
    span: tracing::Span,
    received: Instant,
    started: Option<Instant>,
    prefill_ms: Option<f64>,
    first_token_ms: Option<f64>,
    stream: bool,
    reply: Option<Reply>,
    id: String,
    prompt: Vec<u32>,
    options: GenerationOptions,
    stop_sequences: Vec<String>,
    cache: KvCache,
    sampler: TokenSampler,
    generated: Vec<u32>,
    emitted_reasoning: usize,
    emitted_content: usize,
    role_sent: bool,
    done: bool,
}

pub fn serve(options: ServerOptions) -> Result<(), CliError> {
    if options.max_active_requests == 0 {
        return Err(ServerError::InvalidOptions.into());
    }
    let bind = options.bind.clone();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|source| ServerError::Runtime { source })?;
    runtime.block_on(async move {
        let (commands, receiver) = async_mpsc::channel(COMMAND_CHANNEL_CAPACITY);
        let (ready_sender, ready_receiver) = oneshot::channel();
        let worker =
            tokio::task::spawn_blocking(move || run_inference(options, receiver, ready_sender));
        let model_id = match ready_receiver.await {
            Ok(Ok(model_id)) => model_id,
            Ok(Err(error)) => {
                worker
                    .await
                    .map_err(|source| ServerError::WorkerJoin { source })?;
                return Err(error.into());
            }
            Err(source) => {
                worker
                    .await
                    .map_err(|source| ServerError::WorkerJoin { source })?;
                return Err(ServerError::WorkerStoppedSource { source }.into());
            }
        };
        let state = HttpState { model_id, commands };
        let router = Router::new()
            .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(completion))
            .layer(DefaultBodyLimit::max(MAX_REQUEST_BYTES))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind(&bind)
            .await
            .map_err(|source| ServerError::Bind { source })?;
        tracing::info!(%bind, "OpenAI-compatible server listening");
        let result = axum::serve(listener, router).await;
        worker
            .await
            .map_err(|source| ServerError::WorkerJoin { source })?;
        result.map_err(|source| ServerError::Http { source }.into())
    })
}

async fn models(State(state): State<HttpState>) -> Json<Value> {
    Json(
        json!({"object": "list", "data": [{"id": state.model_id, "object": "model", "owned_by": "metal-infer"}]}),
    )
}

async fn completion(
    State(state): State<HttpState>,
    Json(body): Json<Value>,
) -> Response {
    let streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let id = completion_id();
    let span = tracing::info_span!("completion", request_id = %id, stream = streaming);
    completion_inner(state, body, streaming, id)
        .instrument(span)
        .await
}

async fn completion_inner(
    state: HttpState,
    body: Value,
    streaming: bool,
    id: String,
) -> Response {
    let received = Instant::now();
    if streaming {
        let (sender, receiver) = async_mpsc::channel(STREAM_CHANNEL_CAPACITY);
        if let Err(error) = state.commands.try_send(Command {
            id,
            span: tracing::Span::current(),
            received,
            body,
            reply: Reply::Stream(sender),
        }) {
            return admission_error_response(error);
        }
        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            HeaderValue::from_static("text/event-stream"),
        );
        headers.insert("cache-control", HeaderValue::from_static("no-cache"));
        headers.insert("access-control-allow-origin", HeaderValue::from_static("*"));
        (headers, Body::from_stream(ReceiverStream::new(receiver))).into_response()
    } else {
        let (sender, receiver) = oneshot::channel();
        if let Err(error) = state.commands.try_send(Command {
            id,
            span: tracing::Span::current(),
            received,
            body,
            reply: Reply::Once(sender),
        }) {
            return admission_error_response(error);
        }
        match receiver.await {
            Ok(Ok(value)) => Json(value).into_response(),
            Ok(Err(error)) => error_response(error_status(&error), error.message()),
            Err(error) => {
                tracing::error!(message = "Inference worker stopped before replying.", error = %error);
                error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "inference worker stopped",
                )
            }
        }
    }
}

fn error_response(
    status: StatusCode,
    message: &str,
) -> Response {
    (status, Json(error_body(status, message))).into_response()
}

fn error_body(
    status: StatusCode,
    message: &str,
) -> Value {
    let kind = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    json!({"error": {"message": message, "type": kind}})
}

const fn error_status(error: &ServerError) -> StatusCode {
    match error {
        ServerError::InvalidRequest
        | ServerError::InvalidRequestSource { .. }
        | ServerError::InvalidJson { .. }
        | ServerError::InvalidOptions
        | ServerError::InvalidOptionsSource { .. }
        | ServerError::Disconnected
        | ServerError::SlowClient => StatusCode::BAD_REQUEST,
        ServerError::Runtime { .. }
        | ServerError::Bind { .. }
        | ServerError::Http { .. }
        | ServerError::WorkerStopped
        | ServerError::WorkerStoppedSource { .. }
        | ServerError::WorkerJoin { .. }
        | ServerError::ModelInitialization
        | ServerError::ModelInitializationSource { .. }
        | ServerError::CacheAllocation { .. }
        | ServerError::Inference
        | ServerError::InferenceSource { .. } => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

fn admission_error_response(error: async_mpsc::error::TrySendError<Command>) -> Response {
    match error {
        async_mpsc::error::TrySendError::Full(_) => {
            tracing::warn!(message = "Request queue is full.");
            error_response(StatusCode::SERVICE_UNAVAILABLE, "request queue is full")
        }
        async_mpsc::error::TrySendError::Closed(_) => {
            tracing::error!(message = "Inference worker stopped.");
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference worker stopped",
            )
        }
    }
}

fn log_server_error(
    operation: &'static str,
    error: &ServerError,
) {
    match error {
        ServerError::Runtime { source }
        | ServerError::Bind { source }
        | ServerError::Http { source } => {
            tracing::error!(message = "Server operation failed.", operation, error = %source);
        }
        ServerError::ModelInitializationSource { source }
        | ServerError::InvalidRequestSource { source }
        | ServerError::InferenceSource { source } => {
            tracing::error!(message = "Server operation failed.", operation, error = %source);
        }
        ServerError::InvalidOptionsSource { source } | ServerError::CacheAllocation { source } => {
            tracing::error!(message = "Server operation failed.", operation, error = %source);
        }
        ServerError::WorkerStoppedSource { source } => {
            tracing::error!(message = "Server operation failed.", operation, error = %source);
        }
        ServerError::WorkerJoin { source } => {
            tracing::error!(message = "Server operation failed.", operation, error = %source);
        }
        ServerError::InvalidJson { source } => {
            tracing::error!(message = "Server operation failed.", operation, error = %source);
        }
        ServerError::WorkerStopped
        | ServerError::ModelInitialization
        | ServerError::InvalidRequest
        | ServerError::InvalidOptions
        | ServerError::Inference
        | ServerError::Disconnected
        | ServerError::SlowClient => {
            tracing::error!(message = "Server operation failed.", operation, error = %error);
        }
    }
}

fn run_inference(
    options: ServerOptions,
    mut incoming: async_mpsc::Receiver<Command>,
    ready: oneshot::Sender<Result<String, ServerError>>,
) {
    let initialized = (|| -> Result<ServerState, ServerError> {
        let requested_model_id = hugging_face_model_id(&options.model);
        let model_path = resolve_model_path(&options.model).map_err(|source| {
            ServerError::ModelInitializationSource {
                source: Box::new(source),
            }
        })?;
        let context =
            MetalContext::new().map_err(|source| ServerError::ModelInitializationSource {
                source: Box::new(source.into()),
            })?;
        tracing::info!(message = "Metal device ready.", device = %context.device_name());
        tracing::info!(message = "Loading model.", path = %model_path.display());
        let tokenizer = ModelTokenizer::from_directory(&model_path).map_err(|source| {
            ServerError::ModelInitializationSource {
                source: Box::new(source.into()),
            }
        })?;
        let model = load_model(&model_path, &context, &options.with)
            .map_err(|source| ServerError::ModelInitializationSource {
                source: Box::new(source),
            })?
            .ok_or(ServerError::ModelInitialization)?;
        let model_id = options
            .model_id
            .or(requested_model_id)
            .unwrap_or_else(|| infer_model_id(&model_path));
        Ok(ServerState {
            model_id,
            tokenizer,
            model,
            context: options.context,
        })
    })();
    let state = match initialized {
        Ok(state) => state,
        Err(error) => {
            log_server_error("startup", &error);
            let _ = ready.send(Err(error));
            return;
        }
    };
    let _ = ready.send(Ok(state.model_id.clone()));
    let mut waiting: VecDeque<Command> = VecDeque::new();
    let mut active: Vec<ActiveCompletion> = Vec::new();
    loop {
        while waiting.len() < WAITING_CAPACITY {
            let Ok(command) = incoming.try_recv() else {
                break;
            };
            waiting.push_back(command);
        }
        if active.is_empty() && waiting.is_empty() {
            let Some(command) = incoming.blocking_recv() else {
                break;
            };
            waiting.push_back(command);
        }
        while active.len() < options.max_active_requests {
            let Some(command) = waiting.pop_front() else {
                break;
            };
            if command.reply.is_closed() {
                continue;
            }
            let mut completion = match prepare_command(command, &state) {
                Ok(completion) => completion,
                Err(error) => {
                    log_server_error("request preparation", &error);
                    continue;
                }
            };
            if completion.reply.as_ref().is_some_and(Reply::is_closed) {
                continue;
            }
            let span = completion.span.clone();
            let _guard = span.enter();
            if let Err(source) = completion.start(&state) {
                let error = ServerError::InferenceSource {
                    source: Box::new(source),
                };
                log_server_error("prefill", &error);
                completion.fail(ServerError::Inference);
            }
            if !completion.done {
                active.push(completion);
            }
        }
        active.retain(|completion| {
            !completion.done && !completion.reply.as_ref().is_some_and(Reply::is_closed)
        });
        if active.is_empty() {
            continue;
        }
        let inputs: Option<Vec<u32>> = active
            .iter()
            .map(|completion| completion.generated.last().copied())
            .collect();
        let Some(inputs) = inputs else {
            let error = ServerError::Inference;
            log_server_error("decode input", &error);
            for completion in &mut active {
                completion.fail(ServerError::Inference);
            }
            active.clear();
            continue;
        };
        let logits = if active.len() == 1 {
            let Some(&token) = inputs.first() else {
                continue;
            };
            let Some(completion) = active.first_mut() else {
                continue;
            };
            state.model.decode(token, &mut completion.cache)
        } else {
            let mut caches: Vec<_> = active
                .iter_mut()
                .map(|completion| &mut completion.cache)
                .collect();
            state.model.decode_batch(&inputs, &mut caches)
        };
        let logits = match logits {
            Ok(logits) => logits,
            Err(source) => {
                let error = ServerError::InferenceSource {
                    source: Box::new(source.into()),
                };
                log_server_error("decode", &error);
                for completion in &mut active {
                    completion.fail(ServerError::Inference);
                }
                active.clear();
                continue;
            }
        };
        let sampled = active
            .iter_mut()
            .enumerate()
            .map(|(row, completion)| {
                logits
                    .row(row)
                    .map_err(|source| ServerError::InferenceSource {
                        source: Box::new(source.into()),
                    })
                    .and_then(|row| {
                        completion.sampler.sample(&row).map_err(|source| {
                            ServerError::InferenceSource {
                                source: Box::new(source.into()),
                            }
                        })
                    })
            })
            .collect::<Vec<_>>();
        for (completion, result) in active.iter_mut().zip(sampled) {
            let span = completion.span.clone();
            let _guard = span.enter();
            match result {
                Ok(token) => {
                    if let Err(source) = completion.accept_token(token, &state) {
                        let error = ServerError::InferenceSource {
                            source: Box::new(source),
                        };
                        log_server_error("completion", &error);
                        completion.fail(ServerError::Inference);
                    }
                }
                Err(error) => {
                    log_server_error("sampling", &error);
                    completion.fail(ServerError::Inference);
                }
            }
        }
        active.retain(|completion| !completion.done);
    }
}

fn prepare_command(
    command: Command,
    state: &ServerState,
) -> Result<ActiveCompletion, ServerError> {
    let Command {
        id,
        span,
        received,
        body,
        reply,
    } = command;
    let parsed: ChatCompletionRequest = match serde_json::from_value(body) {
        Ok(value) => value,
        Err(source) => {
            reply.error(ServerError::InvalidRequest);
            return Err(ServerError::InvalidJson { source });
        }
    };
    let prepared = match prepare_completion(state, parsed) {
        Ok(value) => value,
        Err(source) => {
            reply.error(ServerError::InvalidRequest);
            return Err(ServerError::InvalidRequestSource {
                source: Box::new(source),
            });
        }
    };
    let sampler = match TokenSampler::new(prepared.options.clone()) {
        Ok(value) => value,
        Err(source) => {
            reply.error(ServerError::InvalidOptions);
            return Err(ServerError::InvalidOptionsSource { source });
        }
    };
    let capacity = prepared
        .prompt
        .len()
        .saturating_add(prepared.options.max_tokens)
        .max(1);
    let cache = match KvCache::new(state.model.context(), state.model.config(), capacity) {
        Ok(value) => value,
        Err(source) => {
            reply.error(ServerError::Inference);
            return Err(ServerError::CacheAllocation { source });
        }
    };
    if prepared.stream != matches!(reply, Reply::Stream(_)) {
        reply.error(ServerError::InvalidRequest);
        return Err(ServerError::InvalidRequest);
    }
    Ok(ActiveCompletion {
        span,
        received,
        started: None,
        prefill_ms: None,
        first_token_ms: None,
        stream: prepared.stream,
        reply: Some(reply),
        id,
        prompt: prepared.prompt,
        options: prepared.options,
        stop_sequences: prepared.stop_sequences,
        cache,
        sampler,
        generated: Vec::new(),
        emitted_reasoning: 0,
        emitted_content: 0,
        role_sent: false,
        done: false,
    })
}

impl ActiveCompletion {
    fn start(
        &mut self,
        state: &ServerState,
    ) -> Result<(), CliError> {
        self.started = Some(Instant::now());
        if self.options.max_tokens == 0 {
            return self.finish(state);
        }
        let prefill_started = Instant::now();
        let logits = state.model.prefill(&self.prompt, &mut self.cache)?;
        self.prefill_ms = Some(prefill_started.elapsed().as_secs_f64() * 1000.0);
        let token = self.sampler.sample(&logits)?;
        self.accept_token(token, state)
    }

    fn accept_token(
        &mut self,
        token: u32,
        state: &ServerState,
    ) -> Result<(), CliError> {
        if self.options.stop_token_ids.contains(&token) {
            return self.finish(state);
        }
        self.generated.push(token);
        if self.generated.len() == 1 {
            self.first_token_ms = Some(self.received.elapsed().as_secs_f64() * 1000.0);
        }
        let decoded = state.tokenizer.decode(&self.generated)?;
        let stopped = self
            .stop_sequences
            .iter()
            .any(|stop| decoded.contains(stop));
        if self.is_stream() {
            if !self.role_sent {
                self.send_stream(role_chunk(&self.id, &state.model_id))?;
                self.role_sent = true;
            }
            let visible = truncate_stop(decoded, &self.stop_sequences);
            let safe_end = if stopped {
                visible.len()
            } else {
                safe_stream_boundary(&visible, &self.stop_sequences)
            };
            self.emit_delta(visible.get(..safe_end).unwrap_or_default(), &state.model_id)?;
        }
        if stopped || self.generated.len() == self.options.max_tokens {
            self.finish(state)?;
        }
        Ok(())
    }

    fn is_stream(&self) -> bool {
        self.stream
    }

    fn send_stream(
        &self,
        value: Value,
    ) -> Result<(), CliError> {
        let Some(Reply::Stream(sender)) = &self.reply else {
            return Ok(());
        };
        send_event(sender, &value).map_err(Into::into)
    }

    fn emit_delta(
        &mut self,
        visible: &str,
        model_id: &str,
    ) -> Result<(), CliError> {
        let (reasoning, content) = stream_channels(visible);
        if let Some(delta) = reasoning.get(self.emitted_reasoning..)
            && !delta.is_empty()
        {
            self.send_stream(delta_chunk(&self.id, model_id, "reasoning_content", delta))?;
            self.emitted_reasoning = reasoning.len();
        }
        if let Some(delta) = content.get(self.emitted_content..)
            && !delta.is_empty()
        {
            self.send_stream(delta_chunk(&self.id, model_id, "content", delta))?;
            self.emitted_content = content.len();
        }
        Ok(())
    }

    fn finish(
        &mut self,
        state: &ServerState,
    ) -> Result<(), CliError> {
        let decoded = truncate_stop(
            state.tokenizer.decode(&self.generated)?,
            &self.stop_sequences,
        );
        let reason = if self.generated.len() == self.options.max_tokens {
            "length"
        } else {
            "stop"
        };
        let usage = json!({"prompt_tokens": self.prompt.len(), "completion_tokens": self.generated.len(), "total_tokens": self.prompt.len() + self.generated.len()});
        if self.is_stream() {
            if !self.role_sent {
                self.send_stream(role_chunk(&self.id, &state.model_id))?;
                self.role_sent = true;
            }
            self.emit_delta(&decoded, &state.model_id)?;
            self.send_stream(json!({"id": self.id, "object": "chat.completion.chunk", "created": unix_seconds(),
                "model": state.model_id, "choices": [{"index": 0, "delta": {}, "finish_reason": reason}], "usage": usage}))?;
            if let Some(Reply::Stream(sender)) = &self.reply {
                let _ = sender.try_send(Ok(Bytes::from_static(b"data: [DONE]\n\n")));
            }
            self.reply.take();
        } else {
            let (reasoning, content) = split_reasoning(&decoded);
            let mut message = json!({"role": "assistant", "content": content});
            if let Some(reasoning) = reasoning
                && let Some(object) = message.as_object_mut()
            {
                object.insert("reasoning_content".into(), Value::String(reasoning));
            }
            let value = json!({"id": self.id, "object": "chat.completion", "created": unix_seconds(),
                "model": state.model_id, "choices": [{"index": 0, "message": message, "finish_reason": reason}], "usage": usage});
            if let Some(Reply::Once(sender)) = self.reply.take() {
                let _ = sender.send(Ok(value));
            }
        }
        self.done = true;
        tracing::info!(
            request_id = %self.id,
            prompt_tokens = self.prompt.len(),
            generated_tokens = self.generated.len(),
            queue_ms = self.started.map(|started| started.duration_since(self.received).as_secs_f64() * 1000.0),
            prefill_ms = self.prefill_ms,
            first_token_ms = self.first_token_ms,
            total_ms = self.received.elapsed().as_secs_f64() * 1000.0,
            message = "Completion finished.",
        );
        Ok(())
    }

    fn fail(
        &mut self,
        error: ServerError,
    ) {
        if let Some(reply) = self.reply.take() {
            reply.error(error);
        }
        self.done = true;
    }
}

fn send_event(
    sender: &async_mpsc::Sender<Result<Bytes, std::convert::Infallible>>,
    value: &Value,
) -> Result<(), ServerError> {
    sender
        .try_send(Ok(Bytes::from(format!("data: {value}\n\n"))))
        .map_err(|error| match error {
            async_mpsc::error::TrySendError::Full(_) => ServerError::SlowClient,
            async_mpsc::error::TrySendError::Closed(_) => ServerError::Disconnected,
        })
}

fn role_chunk(
    id: &str,
    model_id: &str,
) -> Value {
    json!({"id": id, "object": "chat.completion.chunk", "created": unix_seconds(), "model": model_id,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]})
}

fn delta_chunk(
    id: &str,
    model_id: &str,
    field: &str,
    value: &str,
) -> Value {
    let mut delta = serde_json::Map::new();
    delta.insert(field.to_owned(), Value::String(value.to_owned()));
    json!({"id": id, "object": "chat.completion.chunk", "created": unix_seconds(), "model": model_id,
        "choices": [{"index": 0, "delta": Value::Object(delta), "finish_reason": null}]})
}

fn prepare_completion(
    state: &ServerState,
    request: ChatCompletionRequest,
) -> Result<PreparedCompletion, CliError> {
    let max_tokens = request.max_tokens();
    if let Some(model) = &request.model
        && model != &state.model_id
    {
        return Err(CliError::InvalidArguments(format!(
            "requested model `{model}` is not loaded"
        )));
    }
    let messages = request
        .messages
        .into_iter()
        .map(WireMessage::into_chat_message)
        .collect::<Result<Vec<_>, _>>()?;
    let prompt = state.tokenizer.encode_chat(&messages)?;
    let required = prompt.len().saturating_add(max_tokens);
    if required > state.context {
        return Err(CliError::InvalidArguments(format!(
            "prompt + generated tokens ({required}) exceeds context {}",
            state.context
        )));
    }
    let stop_sequences = request.stop.map_or_else(Vec::new, StopSequences::into_vec);
    let options = GenerationOptions {
        max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        seed: request.seed,
        stop_token_ids: if request.ignore_eos {
            Vec::new()
        } else {
            state.tokenizer.eos_token_ids().to_vec()
        },
    };
    Ok(PreparedCompletion {
        prompt,
        options,
        stop_sequences,
        stream: request.stream,
    })
}

impl WireMessage {
    fn into_chat_message(self) -> Result<ChatMessage, CliError> {
        let content = match self.content {
            MessageContent::Text(text) => text,
            MessageContent::Parts(parts) => {
                let mut output = String::new();
                for part in parts {
                    if part.kind != "text" {
                        return Err(CliError::InvalidArguments(format!(
                            "message content type `{}` is not supported yet",
                            part.kind
                        )));
                    }
                    if let Some(text) = part.text {
                        output.push_str(&text);
                    }
                }
                output
            }
        };
        Ok(ChatMessage {
            role: self.role,
            content,
        })
    }
}

impl ChatCompletionRequest {
    fn max_tokens(&self) -> usize {
        self.max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(DEFAULT_MAX_TOKENS)
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

fn truncate_stop(
    mut value: String,
    stop_sequences: &[String],
) -> String {
    if let Some(index) = stop_sequences
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| value.find(stop))
        .min()
    {
        value.truncate(index);
    }
    value
}

fn safe_stream_boundary(
    value: &str,
    stop_sequences: &[String],
) -> usize {
    let mut withheld = 0usize;
    for stop in stop_sequences.iter().filter(|stop| !stop.is_empty()) {
        for (index, _) in stop.char_indices().skip(1) {
            if value.ends_with(&stop[..index]) {
                withheld = withheld.max(index);
            }
        }
    }
    value.len().saturating_sub(withheld)
}

fn split_reasoning(value: &str) -> (Option<String>, String) {
    let Some(rest) = value.strip_prefix("<think>") else {
        return (None, value.to_owned());
    };
    let Some((reasoning, content)) = rest.split_once("</think>") else {
        return (Some(rest.trim().to_owned()), String::new());
    };
    (Some(reasoning.trim().to_owned()), content.trim().to_owned())
}

fn stream_channels(value: &str) -> (&str, &str) {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    if value.len() < OPEN.len() && OPEN.starts_with(value) {
        return ("", "");
    }
    let Some(rest) = value.strip_prefix(OPEN) else {
        return ("", value);
    };
    if let Some((reasoning, content)) = rest.split_once(CLOSE) {
        return (reasoning, content);
    }
    let safe_end = safe_stream_boundary(rest, &[CLOSE.to_owned()]);
    (rest.get(..safe_end).unwrap_or_default(), "")
}

fn completion_id() -> String {
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    format!(
        "chatcmpl-metal-{}-{}",
        unix_seconds(),
        NEXT_ID.fetch_add(1, Ordering::Relaxed)
    )
}

fn infer_model_id(path: &std::path::Path) -> String {
    for component in path.ancestors().filter_map(std::path::Path::file_name) {
        let Some(name) = component.to_str() else {
            continue;
        };
        if let Some(repository) = name.strip_prefix("models--") {
            return repository.replace("--", "/");
        }
    }
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("qwen3")
        .to_owned()
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}
