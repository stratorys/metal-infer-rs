mod server;

use std::io::Write;
use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use metal_infer_cli::{CliError, LogFormat, init_tracing, load_model, resolve_model_path};
use metal_infer_kernels::MetalContext;
use metal_infer_models::{GenerationOptions, KvCache, ModelTokenizer};

use crate::server::{ServerOptions, serve};

#[derive(Parser)]
#[command(name = "metal-infer", about = "Qwen3 inference on Apple Metal")]
struct Arguments {
    #[arg(long, global = true, value_enum, default_value_t = LogFormat::Text)]
    log_format: LogFormat,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Generate(GenerateArguments),
    Serve(ServeArguments),
}

#[derive(Args)]
struct GenerateArguments {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 32)]
    max_tokens: usize,
    #[arg(long, default_value_t = 2048)]
    context: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    #[arg(long, default_value_t = 0)]
    top_k: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long = "with", value_name = "KEY=VALUE")]
    with: Vec<String>,
}

#[derive(Args)]
struct ServeArguments {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    model_id: Option<String>,
    #[arg(long, default_value = "127.0.0.1:8080")]
    bind: String,
    #[arg(long, default_value_t = 8192)]
    context: usize,
    #[arg(long, default_value_t = 4)]
    max_active_requests: usize,
    #[arg(long = "with", value_name = "KEY=VALUE")]
    with: Vec<String>,
}

fn main() {
    let arguments = Arguments::parse();
    if let Err(error) = init_tracing(arguments.log_format) {
        eprintln!("{error}");
        std::process::exit(1);
    }
    if let Err(error) = run(arguments) {
        tracing::error!(message = "Command failed.", %error);
        std::process::exit(1);
    }
}

fn run(arguments: Arguments) -> Result<(), CliError> {
    match arguments.command {
        Command::Generate(arguments) => generate(arguments),
        Command::Serve(arguments) => serve(ServerOptions {
            model: arguments.model,
            model_id: arguments.model_id,
            bind: arguments.bind,
            context: arguments.context,
            max_active_requests: arguments.max_active_requests,
            with: arguments.with,
        }),
    }
}

fn generate(arguments: GenerateArguments) -> Result<(), CliError> {
    let started = std::time::Instant::now();
    let span = tracing::info_span!(
        "generate",
        max_tokens = arguments.max_tokens,
        context = arguments.context
    );
    let _guard = span.enter();
    let model_path = resolve_model_path(&arguments.model)?;
    let context = MetalContext::new()?;
    tracing::info!(device = %context.device_name(), "Metal device ready");
    tracing::info!(path = %model_path.display(), "loading model");
    let load_started = std::time::Instant::now();
    let tokenizer = ModelTokenizer::from_directory(&model_path)?;
    let prompt = tokenizer.encode(&arguments.prompt)?;
    let Some(model) = load_model(&model_path, &context, &arguments.with)? else {
        return Ok(());
    };
    tracing::info!(
        load_ms = load_started.elapsed().as_secs_f64() * 1000.0,
        prompt_tokens = prompt.len(),
        "model loaded"
    );
    let required = prompt.len().saturating_add(arguments.max_tokens);
    if required > arguments.context {
        return Err(CliError::ContextExceeded);
    }
    let mut cache = KvCache::new(&context, model.config(), arguments.context)?;
    let generation_started = std::time::Instant::now();
    let generated = model.generate_with(
        &prompt,
        &GenerationOptions {
            max_tokens: arguments.max_tokens,
            temperature: arguments.temperature,
            top_p: arguments.top_p,
            top_k: arguments.top_k,
            seed: arguments.seed,
            stop_token_ids: tokenizer.eos_token_ids().to_vec(),
        },
        &mut cache,
        |_| true,
    )?;
    tracing::info!(
        generated_tokens = generated.len(),
        generation_ms = generation_started.elapsed().as_secs_f64() * 1000.0,
        total_ms = started.elapsed().as_secs_f64() * 1000.0,
        allocated_bytes = context.allocated_bytes(),
        "generation complete"
    );
    std::io::stdout().write_all(tokenizer.decode(&generated)?.as_bytes())?;
    Ok(())
}
