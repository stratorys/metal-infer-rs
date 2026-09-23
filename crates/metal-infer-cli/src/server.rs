use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use metal_infer_cli::{CliError, hugging_face_model_id, resolve_model_path};
use metal_infer_models::{ChatMessage, GenerationOptions, KvCache, ModelTokenizer, Qwen3Model};
use metal_infer_runtime::MetalContext;
use serde::Deserialize;
use serde_json::{Value, json};

const MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;

pub struct ServerOptions {
    pub model: PathBuf,
    pub model_id: Option<String>,
    pub bind: String,
    pub context: usize,
}

struct ServerState {
    model_id: String,
    tokenizer: ModelTokenizer,
    model: Qwen3Model,
    cache: KvCache,
    context: usize,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<WireMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
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

struct HttpRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

pub fn serve(options: ServerOptions) -> Result<(), CliError> {
    let requested_model_id = hugging_face_model_id(&options.model);
    let model_path = resolve_model_path(&options.model)?;
    let context = MetalContext::new()?;
    eprintln!("Metal device: {}", context.device_name());
    eprintln!("Loading {}", model_path.display());
    let tokenizer = ModelTokenizer::from_directory(&model_path)?;
    let model = Qwen3Model::load(&model_path, &context)?;
    let cache = KvCache::new(&context, model.config(), options.context)?;
    let model_id = options
        .model_id
        .or(requested_model_id)
        .unwrap_or_else(|| infer_model_id(&model_path));
    let mut state = ServerState {
        model_id,
        tokenizer,
        model,
        cache,
        context: options.context,
    };
    let listener = TcpListener::bind(&options.bind).map_err(|error| {
        CliError::InvalidArguments(format!("cannot bind {}: {error}", options.bind))
    })?;
    eprintln!(
        "OpenAI-compatible server listening on http://{}",
        options.bind
    );
    for connection in listener.incoming() {
        match connection {
            Ok(mut stream) => {
                if let Err(error) = handle_connection(&mut stream, &mut state) {
                    eprintln!("request failed: {error}");
                }
            }
            Err(error) => eprintln!("connection failed: {error}"),
        }
    }
    Ok(())
}

fn handle_connection(
    stream: &mut TcpStream,
    state: &mut ServerState,
) -> Result<(), CliError> {
    let request = match read_request(stream) {
        Ok(request) => request,
        Err(error) => {
            write_json_error(stream, 400, &error.to_string())?;
            return Ok(());
        }
    };
    match (request.method.as_str(), request.path.as_str()) {
        ("OPTIONS", _) => write_empty(stream, 204)?,
        ("GET", "/health") => write_json(stream, 200, &json!({ "status": "ok" }))?,
        ("GET", "/v1/models") => write_json(
            stream,
            200,
            &json!({
                "object": "list",
                "data": [{
                    "id": state.model_id,
                    "object": "model",
                    "owned_by": "metal-infer"
                }]
            }),
        )?,
        ("POST", "/v1/chat/completions") => {
            let parsed: ChatCompletionRequest = match serde_json::from_slice(&request.body) {
                Ok(value) => value,
                Err(error) => {
                    write_json_error(stream, 400, &format!("invalid JSON: {error}"))?;
                    return Ok(());
                }
            };
            let streaming = parsed.stream;
            if let Err(error) = chat_completion(stream, state, parsed) {
                if streaming {
                    eprintln!("streaming completion failed: {error}");
                } else {
                    write_json_error(stream, 400, &error.to_string())?;
                }
            }
        }
        _ => write_json_error(stream, 404, "route not found")?,
    }
    Ok(())
}

fn chat_completion(
    stream: &mut TcpStream,
    state: &mut ServerState,
    request: ChatCompletionRequest,
) -> Result<(), CliError> {
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
    let required = prompt.len().saturating_add(request.max_tokens);
    if required > state.context {
        return Err(CliError::InvalidArguments(format!(
            "prompt + generated tokens ({required}) exceeds context {}",
            state.context
        )));
    }
    let stop_sequences = request.stop.map_or_else(Vec::new, StopSequences::into_vec);
    let options = GenerationOptions {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        seed: request.seed,
        stop_token_ids: state.tokenizer.eos_token_ids().to_vec(),
    };
    let id = completion_id();
    if request.stream {
        stream_completion(stream, state, &id, &prompt, &options, &stop_sequences)
    } else {
        complete_once(stream, state, &id, &prompt, &options, &stop_sequences)
    }
}

fn complete_once(
    stream: &mut TcpStream,
    state: &mut ServerState,
    id: &str,
    prompt: &[u32],
    options: &GenerationOptions,
    stop_sequences: &[String],
) -> Result<(), CliError> {
    let generated = state
        .model
        .generate_with(prompt, options, &mut state.cache, |_| true)?;
    let decoded = truncate_stop(state.tokenizer.decode(&generated)?, stop_sequences);
    let (reasoning, content) = split_reasoning(&decoded);
    let mut message = json!({ "role": "assistant", "content": content });
    if let Some(reasoning) = reasoning
        && let Some(object) = message.as_object_mut()
    {
        object.insert("reasoning_content".into(), Value::String(reasoning));
    }
    write_json(
        stream,
        200,
        &json!({
            "id": id,
            "object": "chat.completion",
            "created": unix_seconds(),
            "model": state.model_id,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": if generated.len() == options.max_tokens { "length" } else { "stop" }
            }],
            "usage": {
                "prompt_tokens": prompt.len(),
                "completion_tokens": generated.len(),
                "total_tokens": prompt.len() + generated.len()
            }
        }),
    )?;
    Ok(())
}

fn stream_completion(
    stream: &mut TcpStream,
    state: &mut ServerState,
    id: &str,
    prompt: &[u32],
    options: &GenerationOptions,
    stop_sequences: &[String],
) -> Result<(), CliError> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
    )?;
    write_sse(
        stream,
        &json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": unix_seconds(),
            "model": state.model_id,
            "choices": [{ "index": 0, "delta": { "role": "assistant" }, "finish_reason": null }]
        }),
    )?;
    let tokenizer = &state.tokenizer;
    let model_id = state.model_id.clone();
    let mut tokens = Vec::new();
    let mut emitted_reasoning = 0usize;
    let mut emitted_content = 0usize;
    let mut write_error = None;
    let generated = state
        .model
        .generate_with(prompt, options, &mut state.cache, |token| {
            tokens.push(token);
            let Ok(decoded) = tokenizer.decode(&tokens) else {
                return false;
            };
            let stopped = stop_sequences.iter().any(|stop| decoded.contains(stop));
            let visible = truncate_stop(decoded, stop_sequences);
            let safe_end = if stopped {
                visible.len()
            } else {
                safe_stream_boundary(&visible, stop_sequences)
            };
            let Some(safe) = visible.get(..safe_end) else {
                return false;
            };
            let (reasoning, content) = stream_channels(safe);
            if let Some(delta) = reasoning.get(emitted_reasoning..)
                && !delta.is_empty()
            {
                if let Err(error) =
                    write_stream_delta(stream, id, &model_id, "reasoning_content", delta)
                {
                    write_error = Some(error);
                    return false;
                }
                emitted_reasoning = reasoning.len();
            }
            if let Some(delta) = content.get(emitted_content..)
                && !delta.is_empty()
            {
                if let Err(error) = write_stream_delta(stream, id, &model_id, "content", delta) {
                    write_error = Some(error);
                    return false;
                }
                emitted_content = content.len();
            }
            !stopped
        })?;
    if let Some(error) = write_error {
        return Err(error.into());
    }
    let decoded = truncate_stop(tokenizer.decode(&generated)?, stop_sequences);
    let (reasoning, content) = stream_channels(&decoded);
    if let Some(delta) = reasoning.get(emitted_reasoning..)
        && !delta.is_empty()
    {
        write_stream_delta(stream, id, &state.model_id, "reasoning_content", delta)?;
    }
    if let Some(delta) = content.get(emitted_content..)
        && !delta.is_empty()
    {
        write_stream_delta(stream, id, &state.model_id, "content", delta)?;
    }
    write_sse(
        stream,
        &json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": unix_seconds(),
            "model": state.model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": if generated.len() == options.max_tokens { "length" } else { "stop" }
            }]
        }),
    )?;
    stream.write_all(b"data: [DONE]\n\n")?;
    stream.flush()?;
    Ok(())
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

impl StopSequences {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(value) => vec![value],
            Self::Many(values) => values,
        }
    }
}

const fn default_max_tokens() -> usize {
    256
}

const fn default_top_p() -> f32 {
    1.0
}

fn read_request(stream: &mut TcpStream) -> Result<HttpRequest, std::io::Error> {
    let mut bytes = Vec::new();
    let mut buffer = [0u8; 8192];
    let header_end = loop {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "connection closed before HTTP headers",
            ));
        }
        let chunk = buffer.get(..read).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid socket read size")
        })?;
        bytes.extend_from_slice(chunk);
        if bytes.len() > MAX_REQUEST_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "request is too large",
            ));
        }
        if let Some(position) = find_bytes(&bytes, b"\r\n\r\n") {
            break position + 4;
        }
    };
    let header_bytes = bytes.get(..header_end).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "invalid HTTP header length",
        )
    })?;
    let headers = std::str::from_utf8(header_bytes).map_err(|_| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "headers are not UTF-8")
    })?;
    let mut lines = headers.split("\r\n");
    let request_line = lines.next().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, "missing request line")
    })?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().unwrap_or_default().to_owned();
    let path = request_parts
        .next()
        .unwrap_or_default()
        .split('?')
        .next()
        .unwrap_or_default()
        .to_owned();
    let content_length = lines
        .filter_map(|line| line.split_once(':'))
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .map(|(_, value)| value.trim().parse::<usize>())
        .transpose()
        .map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid Content-Length")
        })?
        .unwrap_or(0);
    if header_end + content_length > MAX_REQUEST_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "request is too large",
        ));
    }
    while bytes.len() < header_end + content_length {
        let read = stream.read(&mut buffer)?;
        if read == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "connection closed before HTTP body",
            ));
        }
        let chunk = buffer.get(..read).ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid socket read size")
        })?;
        bytes.extend_from_slice(chunk);
    }
    let body = bytes
        .get(header_end..header_end + content_length)
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid HTTP body length")
        })?
        .to_vec();
    Ok(HttpRequest { method, path, body })
}

fn write_json(
    stream: &mut TcpStream,
    status: u16,
    value: &Value,
) -> std::io::Result<()> {
    let body = serde_json::to_vec(value).map_err(std::io::Error::other)?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)?;
    stream.flush()
}

fn write_empty(
    stream: &mut TcpStream,
    status: u16,
) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status} No Content\r\nContent-Length: 0\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n\r\n"
    )?;
    stream.flush()
}

fn write_json_error(
    stream: &mut TcpStream,
    status: u16,
    message: &str,
) -> std::io::Result<()> {
    write_json(
        stream,
        status,
        &json!({ "error": { "message": message, "type": "invalid_request_error" } }),
    )
}

fn write_sse(
    stream: &mut TcpStream,
    value: &Value,
) -> std::io::Result<()> {
    let bytes = serde_json::to_vec(value).map_err(std::io::Error::other)?;
    stream.write_all(b"data: ")?;
    stream.write_all(&bytes)?;
    stream.write_all(b"\n\n")?;
    stream.flush()
}

fn write_stream_delta(
    stream: &mut TcpStream,
    id: &str,
    model_id: &str,
    field: &str,
    value: &str,
) -> std::io::Result<()> {
    let mut delta = serde_json::Map::new();
    delta.insert(field.to_owned(), Value::String(value.to_owned()));
    write_sse(
        stream,
        &json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": unix_seconds(),
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": Value::Object(delta),
                "finish_reason": null
            }]
        }),
    )
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
    format!("chatcmpl-metal-{}", unix_seconds())
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

fn find_bytes(
    haystack: &[u8],
    needle: &[u8],
) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::{
        infer_model_id, safe_stream_boundary, split_reasoning, stream_channels, truncate_stop,
    };

    #[test]
    fn stop_sequences_are_not_returned() {
        assert_eq!(
            truncate_stop("hello STOP later".into(), &["STOP".into()]),
            "hello "
        );
        assert_eq!(safe_stream_boundary("hello ST", &["STOP".into()]), 6);
    }

    #[test]
    fn reasoning_is_split_from_answer() {
        assert_eq!(
            split_reasoning("<think>work</think> answer"),
            (Some("work".into()), "answer".into())
        );
        assert_eq!(stream_channels("<thi"), ("", ""));
        assert_eq!(stream_channels("<think>work</thi"), ("work", ""));
        assert_eq!(
            stream_channels("<think>work</think>answer"),
            ("work", "answer")
        );
    }

    #[test]
    fn hugging_face_cache_path_has_readable_model_id() {
        assert_eq!(
            infer_model_id(Path::new(
                "/cache/models--Qwen--Qwen3-8B/snapshots/revision"
            )),
            "Qwen/Qwen3-8B"
        );
    }
}
