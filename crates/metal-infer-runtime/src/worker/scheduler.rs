use std::collections::VecDeque;
use std::num::NonZeroUsize;

use metal_infer_kernels::Tensor;
use tokio::sync::mpsc;

use crate::worker::completion::Completion;
use crate::worker::engine::Engine;
use crate::worker::error::WorkerError;
use crate::worker::job::Job;

const WAITING_CAPACITY: usize = 16;

pub struct Scheduler {
    engine: Engine,
    jobs: mpsc::Receiver<Job>,
    active_max: NonZeroUsize,
    waiting: VecDeque<Job>,
    active: Vec<Completion>,
}

impl Scheduler {
    pub fn new(
        engine: Engine,
        jobs: mpsc::Receiver<Job>,
        active_max: NonZeroUsize,
    ) -> Self {
        Self {
            engine,
            jobs,
            active_max,
            waiting: VecDeque::with_capacity(WAITING_CAPACITY),
            active: Vec::with_capacity(active_max.get()),
        }
    }

    pub fn run(mut self) {
        while self.receive() {
            self.admit();
            self.step();
        }
        tracing::info!(message = "Inference worker stopped.");
    }

    fn receive(&mut self) -> bool {
        while self.waiting.len() < WAITING_CAPACITY
            && let Ok(job) = self.jobs.try_recv()
        {
            self.waiting.push_back(job);
        }
        if !self.active.is_empty() || !self.waiting.is_empty() {
            return true;
        }
        match self.jobs.blocking_recv() {
            Some(job) => {
                self.waiting.push_back(job);
                true
            }
            None => false,
        }
    }

    fn admit(&mut self) {
        while self.active.len() < self.active_max.get()
            && let Some(job) = self.waiting.pop_front()
        {
            if !job.events.is_closed()
                && let Some(completion) = Completion::start(job, &self.engine)
            {
                self.active.push(completion);
            }
        }
    }

    fn step(&mut self) {
        self.active
            .iter_mut()
            .for_each(Completion::check_disconnected);
        self.active.retain(Completion::is_running);
        if self.active.is_empty() {
            return;
        }
        match self.decode() {
            Ok(logits) => self
                .active
                .iter_mut()
                .enumerate()
                .for_each(|(row, completion)| completion.advance(&logits, row, &self.engine)),
            Err(error) => {
                tracing::error!(message = "Batch step failed.", operation = "decode", %error);
                self.active
                    .iter_mut()
                    .for_each(|completion| completion.fail(WorkerError::BatchFailed));
            }
        }
        self.active.retain(Completion::is_running);
    }

    fn decode(&mut self) -> Result<Tensor, WorkerError> {
        let inputs = self
            .active
            .iter()
            .map(Completion::last_token)
            .collect::<Option<Vec<u32>>>()
            .ok_or(WorkerError::MissingDecodeInput)?;
        match (inputs.as_slice(), self.active.as_mut_slice()) {
            ([token], [completion]) => self.engine.model.decode(*token, completion.cache_mut()),
            _ => self.engine.model.decode_batch(
                &inputs,
                &mut self
                    .active
                    .iter_mut()
                    .map(Completion::cache_mut)
                    .collect::<Vec<_>>(),
            ),
        }
        .map_err(WorkerError::Inference)
    }
}
