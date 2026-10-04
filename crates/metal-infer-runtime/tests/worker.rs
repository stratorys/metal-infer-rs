use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use metal_infer_models::ChatMessage;
use metal_infer_runtime::{
    Event, FinishReason, Job, JobOptions, SubmitError, Summary, Worker, WorkerError, WorkerHandle,
    WorkerOptions, spawn,
};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tracing::Span;

const PROMPT: &str = "Count from 1 to 20.";
const ABANDONED_TIMEOUT: Duration = Duration::from_secs(5);

struct Pending {
    job: Job,
    events: mpsc::Receiver<Event>,
}

#[tokio::test]
async fn completion_streams_deltas_then_finishes() {
    let worker = start_worker(2048, 1).await;
    let events = drain(submit(&worker.handle(), "length", options(16, &[]))).await;
    let summary = finished(&events);
    assert_eq!(
        summary.reason,
        FinishReason::Length,
        "a completion without stop must end on its length"
    );
    assert_eq!(
        summary.completion_tokens, 16,
        "the completion must generate max_tokens tokens"
    );
    assert!(
        summary.prompt_tokens > 0,
        "the summary must count the prompt tokens"
    );
    assert!(
        events.iter().rev().skip(1).all(is_delta),
        "every event before the terminal one must be a delta"
    );
    assert!(
        !text(&events).is_empty(),
        "the completion must produce text"
    );
    worker.join().await.expect("worker joins");
}

#[tokio::test]
async fn stop_string_truncates_and_stops() {
    let worker = start_worker(2048, 1).await;
    let events = drain(submit(&worker.handle(), "stop", options(64, &["20"]))).await;
    let summary = finished(&events);
    let output = text(&events);
    assert_eq!(
        summary.reason,
        FinishReason::Stop,
        "a matched stop string must end the completion with stop"
    );
    assert!(
        summary.completion_tokens < 64,
        "a matched stop string must end the completion early"
    );
    assert!(
        !output.contains("20"),
        "the stop string must not reach the client"
    );
    assert!(
        !output.ends_with('2'),
        "a stop string prefix must be withheld from the stream"
    );
    worker.join().await.expect("worker joins");
}

#[tokio::test]
async fn batched_requests_match_a_single_request() {
    let worker = start_worker(2048, 4).await;
    let single = text(&drain(submit(&worker.handle(), "single", options(32, &[]))).await);
    let batched = (0..4)
        .map(|index| {
            submit(
                &worker.handle(),
                &format!("batched-{index}"),
                options(32, &[]),
            )
        })
        .collect::<Vec<_>>();
    for events in batched {
        assert_eq!(
            text(&drain(events).await),
            single,
            "a greedy batched completion must match the single completion"
        );
    }
    worker.join().await.expect("worker joins");
}

#[tokio::test]
async fn dropped_receiver_frees_the_active_slot() {
    let worker = start_worker(2048, 1).await;
    let abandoned = submit(&worker.handle(), "abandoned", options(2000, &[]));
    drop(first_event(abandoned).await);
    let started = Instant::now();
    let events = drain(submit(&worker.handle(), "next", options(8, &[]))).await;
    assert!(
        started.elapsed() < ABANDONED_TIMEOUT,
        "the next completion must not wait for the abandoned one"
    );
    assert_eq!(
        finished(&events).reason,
        FinishReason::Length,
        "the next completion must run once the abandoned one is dropped"
    );
    worker.join().await.expect("worker joins");
}

#[tokio::test]
async fn context_overflow_is_a_client_error() {
    let worker = start_worker(32, 1).await;
    let events = drain(submit(&worker.handle(), "overflow", options(30, &[]))).await;
    let [Event::Failed(error)] = events.as_slice() else {
        panic!("expected a single failure event, got {events:?}");
    };
    assert!(
        matches!(error, WorkerError::ContextExceeded),
        "the failure must be a context overflow"
    );
    assert!(
        error.is_client_error(),
        "a context overflow must be a client error"
    );
    worker.join().await.expect("worker joins");
}

#[tokio::test]
async fn full_queue_rejects_and_accepted_jobs_finish() {
    let worker = start_worker(2048, 1).await;
    let handle = worker.handle();
    let submitted = (0..40)
        .map(|index| {
            let Pending { job, events } = pending(&format!("burst-{index}"), options(16, &[]));
            handle.try_submit(job).map(|()| events)
        })
        .collect::<Vec<Result<mpsc::Receiver<Event>, SubmitError>>>();
    drop(handle);
    assert!(
        submitted
            .iter()
            .any(|result| matches!(result, Err(SubmitError::QueueFull))),
        "a burst larger than the queue must be rejected with QueueFull"
    );
    for events in submitted.into_iter().filter_map(Result::ok) {
        assert_eq!(
            finished(&drain(events).await).reason,
            FinishReason::Length,
            "every accepted job must finish"
        );
    }
    worker.join().await.expect("worker joins");
}

#[tokio::test]
async fn join_drains_the_queued_jobs() {
    let worker = start_worker(2048, 2).await;
    let queued = (0..5)
        .map(|index| {
            submit(
                &worker.handle(),
                &format!("queued-{index}"),
                options(8, &[]),
            )
        })
        .collect::<Vec<_>>();
    worker
        .join()
        .await
        .expect("worker drains its queue and joins");
    for events in queued {
        assert_eq!(
            finished(&drain(events).await).reason,
            FinishReason::Length,
            "every queued job must finish before the worker stops"
        );
    }
}

async fn start_worker(
    context: usize,
    max_active_requests: usize,
) -> Worker {
    spawn(WorkerOptions {
        model: model_path(),
        model_id: None,
        context,
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

fn options(
    max_tokens: usize,
    stop: &[&str],
) -> JobOptions {
    JobOptions {
        max_tokens,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
        ignore_eos: true,
        stop: stop.iter().map(|value| (*value).to_owned()).collect(),
    }
}

fn pending(
    id: &str,
    options: JobOptions,
) -> Pending {
    let (sender, events) = mpsc::channel(options.event_capacity());
    Pending {
        job: Job {
            id: id.to_owned(),
            span: Span::none(),
            received: Instant::now(),
            messages: vec![ChatMessage {
                role: "user".to_owned(),
                content: PROMPT.to_owned(),
            }],
            options,
            events: sender,
        },
        events,
    }
}

fn submit(
    handle: &WorkerHandle,
    id: &str,
    options: JobOptions,
) -> mpsc::Receiver<Event> {
    let Pending { job, events } = pending(id, options);
    handle.try_submit(job).expect("job accepted");
    events
}

async fn drain(events: mpsc::Receiver<Event>) -> Vec<Event> {
    ReceiverStream::new(events).collect().await
}

async fn first_event(mut events: mpsc::Receiver<Event>) -> mpsc::Receiver<Event> {
    events.recv().await.expect("first event");
    events
}

fn finished(events: &[Event]) -> Summary {
    let Some(Event::Finished(summary)) = events.last() else {
        panic!("expected a finished completion, got {events:?}");
    };
    *summary
}

const fn is_delta(event: &Event) -> bool {
    matches!(event, Event::Reasoning(_) | Event::Content(_))
}

fn text(events: &[Event]) -> String {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Reasoning(text) | Event::Content(text) => Some(text.as_str()),
            Event::Finished(_) | Event::Failed(_) => None,
        })
        .collect()
}
