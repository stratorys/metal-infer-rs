use std::time::{Duration, Instant};

use metal_infer_kernels::Tensor;
use metal_infer_models::{ChatMessage, GenerationOptions, KvCache, ModelError, TokenSampler};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tracing::Span;

use crate::worker::engine::Engine;
use crate::worker::error::WorkerError;
use crate::worker::job::{Event, FinishReason, Job, JobOptions, Summary};
use crate::worker::text::{
    Channels, final_channels, safe_stream_boundary, stream_channels, truncate_stop,
};

pub struct Completion {
    span: Span,
    received: Instant,
    started: Instant,
    prefill_ms: Option<f64>,
    first_token_ms: Option<f64>,
    prompt: Vec<u32>,
    max_tokens: usize,
    stop_token_ids: Vec<u32>,
    stop: Vec<String>,
    cache: KvCache,
    sampler: TokenSampler,
    generated: Vec<u32>,
    emitted_reasoning: usize,
    emitted_content: usize,
    events: mpsc::Sender<Event>,
    done: bool,
}

struct Prepared {
    prompt: Vec<u32>,
    stop_token_ids: Vec<u32>,
    sampler: TokenSampler,
    cache: KvCache,
}

impl Completion {
    pub fn start(
        job: Job,
        engine: &Engine,
    ) -> Option<Self> {
        let job_span = job.span.clone();
        let _guard = job_span.enter();
        let Job {
            span,
            received,
            messages,
            options,
            events,
            ..
        } = job;
        match prepare(&messages, &options, engine) {
            Ok(prepared) => Self {
                span,
                received,
                started: Instant::now(),
                prefill_ms: None,
                first_token_ms: None,
                prompt: prepared.prompt,
                max_tokens: options.max_tokens,
                stop_token_ids: prepared.stop_token_ids,
                stop: options.stop,
                cache: prepared.cache,
                sampler: prepared.sampler,
                generated: Vec::with_capacity(options.max_tokens),
                emitted_reasoning: 0,
                emitted_content: 0,
                events,
                done: false,
            }
            .prefilled(engine),
            Err(error) => {
                tracing::error!(message = "Completion failed.", operation = "prepare", %error);
                let _ = events.try_send(Event::Failed(error));
                None
            }
        }
    }

    pub const fn is_running(&self) -> bool {
        !self.done
    }

    pub fn last_token(&self) -> Option<u32> {
        self.generated.last().copied()
    }

    pub const fn cache_mut(&mut self) -> &mut KvCache {
        &mut self.cache
    }

    pub fn check_disconnected(&mut self) {
        if !self.done && self.events.is_closed() {
            self.disconnect();
        }
    }

    pub fn advance(
        &mut self,
        logits: &Tensor,
        row: usize,
        engine: &Engine,
    ) {
        let span = self.span.clone();
        let _guard = span.enter();
        let result = logits
            .row(row)
            .map_err(ModelError::from)
            .and_then(|row| self.sampler.sample(&row))
            .map_err(WorkerError::Inference)
            .and_then(|token| self.accept_token(token, engine));
        if let Err(error) = result {
            self.report("decode", error);
        }
    }

    pub fn fail(
        &mut self,
        error: WorkerError,
    ) {
        let _ = self.events.try_send(Event::Failed(error));
        self.done = true;
    }

    fn prefilled(
        mut self,
        engine: &Engine,
    ) -> Option<Self> {
        if let Err(error) = self.prefill(engine) {
            self.report("prefill", error);
        }
        self.is_running().then_some(self)
    }

    fn prefill(
        &mut self,
        engine: &Engine,
    ) -> Result<(), WorkerError> {
        if self.max_tokens == 0 {
            return self.finish(engine);
        }
        let prefill_started = Instant::now();
        let logits = engine
            .model
            .prefill(&self.prompt, &mut self.cache)
            .map_err(WorkerError::Inference)?;
        self.prefill_ms = Some(milliseconds(prefill_started.elapsed()));
        let token = self
            .sampler
            .sample(&logits)
            .map_err(WorkerError::Inference)?;
        self.accept_token(token, engine)
    }

    fn accept_token(
        &mut self,
        token: u32,
        engine: &Engine,
    ) -> Result<(), WorkerError> {
        if self.stop_token_ids.contains(&token) {
            return self.finish(engine);
        }
        self.generated.push(token);
        if self.generated.len() == 1 {
            self.first_token_ms = Some(milliseconds(self.received.elapsed()));
        }
        let decoded = engine
            .tokenizer
            .decode(&self.generated)
            .map_err(WorkerError::Inference)?;
        let stopped = self.stop.iter().any(|stop| decoded.contains(stop.as_str()));
        let visible = truncate_stop(&decoded, &self.stop);
        let safe_end = if stopped {
            visible.len()
        } else {
            safe_stream_boundary(visible, &self.stop)
        };
        self.emit_channels(stream_channels(visible.get(..safe_end).unwrap_or_default()))?;
        if stopped || self.generated.len() == self.max_tokens {
            return self.finish(engine);
        }
        Ok(())
    }

    fn finish(
        &mut self,
        engine: &Engine,
    ) -> Result<(), WorkerError> {
        if self.done {
            return Ok(());
        }
        let decoded = engine
            .tokenizer
            .decode(&self.generated)
            .map_err(WorkerError::Inference)?;
        self.emit_channels(final_channels(truncate_stop(&decoded, &self.stop)))?;
        let reason = if self.generated.len() == self.max_tokens {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        self.emit(Event::Finished(Summary {
            reason,
            prompt_tokens: self.prompt.len(),
            completion_tokens: self.generated.len(),
        }))?;
        self.done = true;
        tracing::info!(
            prompt_tokens = self.prompt.len(),
            generated_tokens = self.generated.len(),
            queue_ms = milliseconds(self.started.duration_since(self.received)),
            prefill_ms = self.prefill_ms,
            first_token_ms = self.first_token_ms,
            total_ms = milliseconds(self.received.elapsed()),
            message = "Completion finished.",
        );
        Ok(())
    }

    fn emit_channels(
        &mut self,
        channels: Channels<'_>,
    ) -> Result<(), WorkerError> {
        if let Some(delta) = channels
            .reasoning
            .get(self.emitted_reasoning..)
            .filter(|delta| !delta.is_empty())
        {
            self.emit(Event::Reasoning(delta.to_owned()))?;
            self.emitted_reasoning = channels.reasoning.len();
        }
        if let Some(delta) = channels
            .content
            .get(self.emitted_content..)
            .filter(|delta| !delta.is_empty())
        {
            self.emit(Event::Content(delta.to_owned()))?;
            self.emitted_content = channels.content.len();
        }
        Ok(())
    }

    fn emit(
        &mut self,
        event: Event,
    ) -> Result<(), WorkerError> {
        if self.done {
            return Ok(());
        }
        match self.events.try_send(event) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err(WorkerError::EventOverflow),
            Err(TrySendError::Closed(_)) => {
                self.disconnect();
                Ok(())
            }
        }
    }

    fn disconnect(&mut self) {
        tracing::info!(message = "Client disconnected.");
        self.done = true;
    }

    fn report(
        &mut self,
        operation: &'static str,
        error: WorkerError,
    ) {
        tracing::error!(message = "Completion failed.", operation, %error);
        self.fail(error);
    }
}

fn prepare(
    messages: &[ChatMessage],
    options: &JobOptions,
    engine: &Engine,
) -> Result<Prepared, WorkerError> {
    let prompt = engine
        .tokenizer
        .encode_chat(messages)
        .map_err(WorkerError::Tokenization)?;
    let required = prompt.len().saturating_add(options.max_tokens);
    if required > engine.context {
        return Err(WorkerError::ContextExceeded);
    }
    let stop_token_ids = if options.ignore_eos {
        Vec::new()
    } else {
        engine.tokenizer.eos_token_ids().to_vec()
    };
    let sampler = TokenSampler::new(GenerationOptions {
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        top_k: options.top_k,
        seed: options.seed,
        stop_token_ids: stop_token_ids.clone(),
    })
    .map_err(WorkerError::InvalidOptions)?;
    let cache = KvCache::new(
        engine.model.context(),
        engine.model.config(),
        required.max(1),
    )
    .map_err(WorkerError::CacheAllocation)?;
    Ok(Prepared {
        prompt,
        stop_token_ids,
        sampler,
        cache,
    })
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
