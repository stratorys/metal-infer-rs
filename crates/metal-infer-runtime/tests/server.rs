use std::num::NonZeroUsize;
use std::path::PathBuf;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use http_body_util::BodyExt;
use metal_infer_runtime::{Worker, WorkerOptions, router, serve_with, spawn};
use serde_json::Value;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio_stream::StreamExt;
use tower::ServiceExt;

const COMPLETIONS: &str = "/v1/chat/completions";
const NON_STREAM: &str = r#"{"messages":[{"role":"user","content":"Count from 1 to 20."}],"max_tokens":16,"ignore_eos":true}"#;
const STREAM: &str = r#"{"messages":[{"role":"user","content":"Count from 1 to 20."}],"max_tokens":16,"ignore_eos":true,"stream":true}"#;
const LONG_STREAM: &str = r#"{"messages":[{"role":"user","content":"Write a long story."}],"max_tokens":256,"ignore_eos":true,"stream":true}"#;

#[tokio::test]
async fn non_stream_completion_returns_an_openai_body() {
    let worker = start_worker(1).await;
    let response = post(&router(&worker), NON_STREAM).await;
    assert_eq!(response.status(), StatusCode::OK, "completion must succeed");
    let body = json(response).await;
    assert_eq!(
        body.pointer("/object"),
        Some(&Value::from("chat.completion")),
        "the body must be a chat completion"
    );
    assert_eq!(
        body.pointer("/choices/0/finish_reason"),
        Some(&Value::from("length")),
        "the completion must end on its length"
    );
    assert_eq!(
        body.pointer("/usage/completion_tokens"),
        Some(&Value::from(16)),
        "usage must count the generated tokens"
    );
    worker.join().await.expect("worker joins");
}

#[tokio::test]
async fn stream_completion_ends_with_usage_then_done() {
    let worker = start_worker(1).await;
    let response = post(&router(&worker), STREAM).await;
    assert_eq!(response.status(), StatusCode::OK, "stream must start");
    let text = text(response).await;
    let frames = text.split_terminator("\n\n").collect::<Vec<_>>();
    assert!(
        frames
            .first()
            .is_some_and(|frame| frame.contains(r#""role":"assistant""#)),
        "the first frame must carry the role"
    );
    assert_eq!(
        frames.last().copied(),
        Some("data: [DONE]"),
        "the stream must end with [DONE]"
    );
    assert!(
        frames.iter().rev().nth(1).is_some_and(|frame| {
            frame.contains(r#""finish_reason":"length""#) && frame.contains(r#""usage""#)
        }),
        "the final chunk must carry the finish reason and usage"
    );
    worker.join().await.expect("worker joins");
}

#[tokio::test]
async fn invalid_requests_are_rejected_with_openai_errors() {
    let worker = start_worker(1).await;
    let app = router(&worker);
    let invalid_json = post(&app, "{nope").await;
    assert_eq!(
        invalid_json.status(),
        StatusCode::BAD_REQUEST,
        "invalid JSON must be a 400"
    );
    assert_eq!(
        json(invalid_json).await.pointer("/error/type"),
        Some(&Value::from("invalid_request_error")),
        "invalid JSON must return an OpenAI error body"
    );
    let unknown_model = post(
        &app,
        r#"{"model":"other","messages":[{"role":"user","content":"hi"}]}"#,
    )
    .await;
    assert_eq!(
        unknown_model.status(),
        StatusCode::BAD_REQUEST,
        "an unknown model must be a 400"
    );
    assert_eq!(
        json(unknown_model).await.pointer("/error/message"),
        Some(&Value::from("requested model is not loaded")),
        "an unknown model must return its static message"
    );
    drop(app);
    worker.join().await.expect("worker joins");
}

#[tokio::test]
async fn health_and_models_describe_the_server() {
    let worker = start_worker(1).await;
    let app = router(&worker);
    let health = json(get(&app, "/health").await).await;
    assert_eq!(
        health.pointer("/status"),
        Some(&Value::from("ok")),
        "health must report ok"
    );
    let models = json(get(&app, "/v1/models").await).await;
    assert_eq!(
        models.pointer("/data/0/id"),
        Some(&Value::from(worker.info().model_id.as_str())),
        "models must list the loaded model"
    );
    drop(app);
    worker.join().await.expect("worker joins");
}

#[tokio::test]
async fn full_queue_returns_service_unavailable() {
    let worker = start_worker(1).await;
    let app = router(&worker);
    let responses = post_streams(&app, 40).await;
    assert!(
        responses
            .iter()
            .any(|response| response.status() == StatusCode::SERVICE_UNAVAILABLE),
        "a burst larger than the queue must be rejected with 503"
    );
    drop(responses);
    drop(app);
    worker.join().await.expect("worker joins");
}

#[tokio::test]
async fn shutdown_finishes_the_inflight_stream() {
    let worker = start_worker(1).await;
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("listener binds");
    let address = listener.local_addr().expect("listener address");
    let (trigger, shutdown) = oneshot::channel::<()>();
    let server = tokio::spawn(serve_with(listener, worker, async move {
        let _ = shutdown.await;
    }));
    let stream = first_byte(
        send(
            TcpStream::connect(address).await.expect("client connects"),
            &raw_request(LONG_STREAM),
        )
        .await,
    )
    .await;
    trigger.send(()).expect("shutdown triggered");
    let response =
        String::from_utf8(read_to_end(stream, Vec::new()).await).expect("response is UTF-8");
    assert!(
        response.contains("data: [DONE]"),
        "the in-flight stream must finish after shutdown is requested"
    );
    server
        .await
        .expect("server task")
        .expect("server stops cleanly");
}

async fn start_worker(max_active_requests: usize) -> Worker {
    spawn(WorkerOptions {
        model: model_path(),
        model_id: None,
        context: 2048,
        max_active_requests: NonZeroUsize::new(max_active_requests)
            .expect("positive active request limit"),
        with: Vec::new(),
    })
    .await
    .expect("worker starts")
}

fn model_path() -> PathBuf {
    std::env::var_os("QWEN3_MODEL").map_or_else(|| PathBuf::from("Qwen/Qwen3-0.6B"), PathBuf::from)
}

async fn post(
    app: &Router,
    body: &str,
) -> Response<Body> {
    app.clone()
        .oneshot(
            Request::post(COMPLETIONS)
                .header("content-type", "application/json")
                .body(Body::from(body.to_owned()))
                .expect("request builds"),
        )
        .await
        .expect("router is infallible")
}

async fn get(
    app: &Router,
    path: &str,
) -> Response<Body> {
    app.clone()
        .oneshot(
            Request::get(path)
                .body(Body::empty())
                .expect("request builds"),
        )
        .await
        .expect("router is infallible")
}

async fn post_streams(
    app: &Router,
    count: usize,
) -> Vec<Response<Body>> {
    tokio_stream::iter(0..count)
        .then(|_| post(app, STREAM))
        .collect()
        .await
}

async fn text(response: Response<Body>) -> String {
    String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .expect("body collects")
            .to_bytes()
            .to_vec(),
    )
    .expect("body is UTF-8")
}

async fn json(response: Response<Body>) -> Value {
    serde_json::from_str(&text(response).await).expect("body is JSON")
}

fn raw_request(body: &str) -> String {
    format!(
        "POST {COMPLETIONS} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

async fn send(
    mut stream: TcpStream,
    request: &str,
) -> TcpStream {
    stream
        .write_all(request.as_bytes())
        .await
        .expect("request is written");
    stream
}

async fn first_byte(mut stream: TcpStream) -> TcpStream {
    stream.read_u8().await.expect("response starts");
    stream
}

async fn read_to_end(
    mut stream: TcpStream,
    mut response: Vec<u8>,
) -> Vec<u8> {
    stream
        .read_to_end(&mut response)
        .await
        .expect("response is read");
    response
}
