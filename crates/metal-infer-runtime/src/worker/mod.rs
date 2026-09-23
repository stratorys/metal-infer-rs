mod completion;
mod engine;
mod error;
mod job;
mod scheduler;
mod text;

use std::num::NonZeroUsize;
use std::path::PathBuf;

use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, oneshot};

use crate::worker::engine::Engine;
pub use crate::worker::error::{SubmitError, WorkerError};
pub use crate::worker::job::{Event, FinishReason, Job, JobOptions, Summary};
use crate::worker::scheduler::Scheduler;

const QUEUE_CAPACITY: usize = 16;
const THREAD_NAME: &str = "metal-worker";

pub struct WorkerOptions {
    pub model: PathBuf,
    pub model_id: Option<String>,
    pub context: usize,
    pub max_active_requests: NonZeroUsize,
    pub with: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct WorkerInfo {
    pub model_id: String,
    pub context: usize,
}

#[derive(Clone)]
pub struct WorkerHandle {
    jobs: mpsc::Sender<Job>,
}

pub struct Worker {
    handle: WorkerHandle,
    info: WorkerInfo,
    finished: oneshot::Receiver<()>,
}

pub async fn spawn(options: WorkerOptions) -> Result<Worker, WorkerError> {
    let (jobs, receiver) = mpsc::channel(QUEUE_CAPACITY);
    let (ready_sender, ready) = oneshot::channel();
    let (finished_sender, finished) = oneshot::channel();
    std::thread::Builder::new()
        .name(THREAD_NAME.to_owned())
        .spawn(move || {
            run(options, receiver, ready_sender);
            let _ = finished_sender.send(());
        })
        .map(drop)
        .map_err(WorkerError::ThreadSpawn)?;
    match ready.await {
        Ok(Ok(info)) => Ok(Worker {
            handle: WorkerHandle { jobs },
            info,
            finished,
        }),
        Ok(Err(error)) => wait(finished).await.and(Err(error)),
        Err(_) => wait(finished).await.and(Err(WorkerError::Stopped)),
    }
}

impl Worker {
    pub fn handle(&self) -> WorkerHandle {
        self.handle.clone()
    }

    pub const fn info(&self) -> &WorkerInfo {
        &self.info
    }

    pub async fn join(self) -> Result<(), WorkerError> {
        let Self {
            handle, finished, ..
        } = self;
        drop(handle);
        wait(finished).await
    }
}

impl WorkerHandle {
    pub fn try_submit(
        &self,
        job: Job,
    ) -> Result<(), SubmitError> {
        self.jobs.try_send(job).map_err(|error| match error {
            TrySendError::Full(_) => SubmitError::QueueFull,
            TrySendError::Closed(_) => SubmitError::Stopped,
        })
    }
}

fn run(
    options: WorkerOptions,
    jobs: mpsc::Receiver<Job>,
    ready: oneshot::Sender<Result<WorkerInfo, WorkerError>>,
) {
    match Engine::load(
        &options.model,
        options.model_id,
        options.context,
        &options.with,
    ) {
        Ok(engine) => {
            if ready.send(Ok(engine.info())).is_ok() {
                Scheduler::new(engine, jobs, options.max_active_requests).run();
            }
        }
        Err(error) => {
            let _ = ready.send(Err(error));
        }
    }
}

async fn wait(finished: oneshot::Receiver<()>) -> Result<(), WorkerError> {
    finished.await.map_err(|_| WorkerError::Panicked)
}
