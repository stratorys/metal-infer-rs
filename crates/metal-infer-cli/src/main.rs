mod server;

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};
use metal_infer_cli::{CliError, load_model, resolve_model_path};
use metal_infer_models::{GenerationOptions, KvCache, ModelTokenizer};
use metal_infer_runtime::MetalContext;

use crate::server::{ServerOptions, serve};

#[derive(Parser)]
#[command(name = "metal-infer", about = "Qwen3 inference on Apple Metal")]
struct Arguments {
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
    #[arg(long = "with", value_name = "KEY=VALUE")]
    with: Vec<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), CliError> {
    match Arguments::parse().command {
        Command::Generate(arguments) => generate(arguments),
        Command::Serve(arguments) => serve(ServerOptions {
            model: arguments.model,
            model_id: arguments.model_id,
            bind: arguments.bind,
            context: arguments.context,
            with: arguments.with,
        }),
    }
}

fn generate(arguments: GenerateArguments) -> Result<(), CliError> {
    let model_path = resolve_model_path(&arguments.model)?;
    let context = MetalContext::new()?;
    eprintln!("Metal device: {}", context.device_name());
    eprintln!("Loading {}", model_path.display());
    let tokenizer = ModelTokenizer::from_directory(&model_path)?;
    let prompt = tokenizer.encode(&arguments.prompt)?;
    let Some(model) = load_model(&model_path, &context, &arguments.with)? else {
        return Ok(());
    };
    let required = prompt.len().saturating_add(arguments.max_tokens);
    if required > arguments.context {
        return Err(CliError::InvalidArguments(format!(
            "prompt + generated tokens ({required}) exceeds context {}",
            arguments.context
        )));
    }
    let mut cache = KvCache::new(&context, model.config(), arguments.context)?;
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
    print!("{}", tokenizer.decode(&generated)?);
    Ok(())
}
