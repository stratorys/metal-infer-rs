use std::time::{Duration, Instant};

use metal_infer_kernels::Tensor;
use metal_infer_models::{ChatMessage, GenerationOptions, KvCache, ModelError, TokenSampler};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TrySendError;
use tracing::Span;

use crate::worker::detokenizer::Detokenizer;
use crate::worker::engine::Engine;
use crate::worker::error::WorkerError;
use crate::worker::job::{Event, FinishReason, Job, JobOptions, Summary};
use crate::worker::text::{
    Channels, ThinkState, advance_think, final_channels, find_stop_in_tail, safe_stream_boundary,
    stream_channels, truncate_stop,
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
    detokenizer: Detokenizer,
    stop_at: Option<usize>,
    think: ThinkState,
    emitted_reasoning: usize,
    emitted_content: usize,
    events: mpsc::Sender<Event>,
    done: bool,
}

struct Delta {
    text: String,
    end: usize,
}

struct Deltas {
    reasoning: Option<Delta>,
    content: Option<Delta>,
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
                detokenizer: Detokenizer::with_capacity(options.max_tokens),
                stop_at: None,
                think: ThinkState::Undecided,
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
        self.detokenizer.last_token()
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
        let delta_bytes = self
            .detokenizer
            .push(token, &engine.tokenizer)
            .map_err(WorkerError::Inference)?;
        if self.detokenizer.token_count() == 1 {
            self.first_token_ms = Some(milliseconds(self.received.elapsed()));
        }
        if delta_bytes > 0 {
            self.stop_at = self
                .stop_at
                .or_else(|| find_stop_in_tail(self.detokenizer.text(), delta_bytes, &self.stop));
            let deltas = self.stream_deltas();
            self.emit_deltas(deltas)?;
        }
        if self.stop_at.is_some() || self.detokenizer.token_count() == self.max_tokens {
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
        self.detokenizer
            .flush(&engine.tokenizer)
            .map_err(WorkerError::Inference)?;
        let deltas = Deltas::unsent(
            final_channels(truncate_stop(self.detokenizer.text(), &self.stop)),
            self.emitted_reasoning,
            self.emitted_content,
        );
        self.emit_deltas(deltas)?;
        let reason = if self.detokenizer.token_count() == self.max_tokens {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        self.emit(Event::Finished(Summary {
            reason,
            prompt_tokens: self.prompt.len(),
            completion_tokens: self.detokenizer.token_count(),
        }))?;
        self.done = true;
        tracing::info!(
            prompt_tokens = self.prompt.len(),
            generated_tokens = self.detokenizer.token_count(),
            queue_ms = milliseconds(self.started.duration_since(self.received)),
            prefill_ms = self.prefill_ms,
            first_token_ms = self.first_token_ms,
            total_ms = milliseconds(self.received.elapsed()),
            message = "Completion finished.",
        );
        Ok(())
    }

    fn stream_deltas(&mut self) -> Deltas {
        let text = self.detokenizer.text();
        let released = self
            .stop_at
            .and_then(|end| text.get(..end))
            .unwrap_or_else(|| {
                text.get(..safe_stream_boundary(text, &self.stop))
                    .unwrap_or_default()
            });
        self.think = advance_think(self.think, released);
        Deltas::unsent(
            stream_channels(released, self.think),
            self.emitted_reasoning,
            self.emitted_content,
        )
    }

    fn emit_deltas(
        &mut self,
        deltas: Deltas,
    ) -> Result<(), WorkerError> {
        if let Some(delta) = deltas.reasoning {
            self.emit(Event::Reasoning(delta.text))?;
            self.emitted_reasoning = delta.end;
        }
        if let Some(delta) = deltas.content {
            self.emit(Event::Content(delta.text))?;
            self.emitted_content = delta.end;
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

impl Delta {
    fn unsent(
        channel: &str,
        emitted: usize,
    ) -> Option<Self> {
        channel
            .get(emitted..)
            .filter(|delta| !delta.is_empty())
            .map(|delta| Self {
                text: delta.to_owned(),
                end: channel.len(),
            })
    }
}

impl Deltas {
    fn unsent(
        channels: Channels<'_>,
        emitted_reasoning: usize,
        emitted_content: usize,
    ) -> Self {
        Self {
            reasoning: Delta::unsent(channels.reasoning, emitted_reasoning),
            content: Delta::unsent(channels.content, emitted_content),
        }
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
