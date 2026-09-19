use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use metal_infer_cli::CliError;
use metal_infer_core::{AttentionKind, MetalContext};
use metal_infer_models::{KvCache, ModelTokenizer, Qwen3Model};

#[derive(Parser)]
#[command(name = "metal-infer", about = "Qwen3 inference on Apple Metal")]
struct Arguments {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt: String,
    #[arg(long, default_value_t = 32)]
    max_tokens: usize,
    #[arg(long, default_value_t = 2048)]
    context: usize,
    #[arg(long, value_enum, default_value_t = Attention::Tiled)]
    attention: Attention,
}

#[derive(Clone, Copy, ValueEnum)]
enum Attention {
    Reference,
    Tiled,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), CliError> {
    let arguments = Arguments::parse();
    let context = MetalContext::new()?;
    eprintln!("Metal device: {}", context.device_name());
    let tokenizer = ModelTokenizer::from_directory(&arguments.model)?;
    let prompt = tokenizer.encode(&arguments.prompt)?;
    let mut model = Qwen3Model::load(&arguments.model, &context)?;
    model.set_attention_kind(match arguments.attention {
        Attention::Reference => AttentionKind::Reference,
        Attention::Tiled => AttentionKind::Tiled,
    });
    let required = prompt.len().saturating_add(arguments.max_tokens);
    if required > arguments.context {
        return Err(CliError::InvalidArguments(format!(
            "prompt + generated tokens ({required}) exceeds context {}",
            arguments.context
        )));
    }
    let mut cache = KvCache::new(&context, model.config(), arguments.context)?;
    let generated = model.generate(&prompt, arguments.max_tokens, &mut cache)?;
    print!("{}", tokenizer.decode(&generated)?);
    Ok(())
}
