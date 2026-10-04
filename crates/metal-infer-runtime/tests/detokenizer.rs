use std::collections::BTreeMap;
use std::path::PathBuf;

use metal_infer_models::{ModelSource, ModelTokenizer};
use metal_infer_runtime::Detokenizer;
use serde::Deserialize;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../metal-infer-models/tests/golden/apple-m4-pro.json"
);
const REPLACEMENT_CHARACTER: char = '\u{FFFD}';
const EMOJI: &str = "Emoji: 👋🏽 👨‍👩‍👧‍👦 🇫🇷 🧑‍💻 ✅ 🦀";
const TEXTS: &[&str] = &[
    "Café, crème brûlée, naïve façade, Ærøskøbing, Œuvre, ß.",
    EMOJI,
    "中文测试：你好，世界。日本語のテキスト、한국어 문장입니다.",
    "<think>\nThe user wants a count. Ça va: 1, 2, 3.\n</think>\n\nHere it is: 1 2 3",
    "<think>an unclosed reasoning block with ünïcödé and 漢字",
    "  leading spaces\n\n\tTabbed line\n   \n  - item one\n  - item two\n\n",
    "fn main() {\n    println!(\"{}\", 42);\n}\n",
];

#[derive(Deserialize)]
struct Golden {
    generations: BTreeMap<String, Vec<u32>>,
}

#[test]
fn golden_generations_decode_incrementally() {
    let tokenizer = tokenizer();
    let bytes = std::fs::read(GOLDEN).expect("read golden");
    let golden: Golden = serde_json::from_slice(&bytes).expect("parse golden");
    assert!(
        !golden.generations.is_empty(),
        "the golden file must contain generations"
    );
    golden
        .generations
        .iter()
        .for_each(|(name, tokens)| assert_incremental(&tokenizer, name, tokens));
}

#[test]
fn texts_decode_incrementally() {
    let tokenizer = tokenizer();
    TEXTS.iter().for_each(|text| {
        let tokens = tokenizer.encode(text).expect("encode text");
        assert_incremental(&tokenizer, text, &tokens);
    });
}

#[test]
fn split_characters_are_covered() {
    let tokenizer = tokenizer();
    let tokens = tokenizer.encode(EMOJI).expect("encode text");
    let split = (1..tokens.len()).any(|end| {
        tokenizer
            .decode(tokens.get(..end).expect("prefix"))
            .expect("decode prefix")
            .ends_with(REPLACEMENT_CHARACTER)
    });
    assert!(
        split,
        "the emoji text must split a character across tokens to cover the hold-back"
    );
}

fn assert_incremental(
    tokenizer: &ModelTokenizer,
    name: &str,
    tokens: &[u32],
) {
    let streamed = tokens.iter().enumerate().fold(
        Detokenizer::with_capacity(tokens.len()),
        |mut detokenizer, (index, token)| {
            detokenizer.push(*token, tokenizer).expect("push token");
            let full = tokenizer
                .decode(tokens.get(..=index).expect("prefix"))
                .expect("decode prefix");
            assert!(
                full.starts_with(detokenizer.text()),
                "{name}: step {index} must be a prefix of the full decode"
            );
            assert!(
                !detokenizer.text().ends_with(REPLACEMENT_CHARACTER),
                "{name}: step {index} must hold back an incomplete character"
            );
            detokenizer
        },
    );
    let finished = flushed(streamed, tokenizer);
    assert_eq!(
        finished.text(),
        tokenizer.decode(tokens).expect("decode all"),
        "{name}: the flushed text must equal the full decode"
    );
    assert_eq!(
        finished.token_count(),
        tokens.len(),
        "{name}: every token must be counted"
    );
}

fn flushed(
    mut detokenizer: Detokenizer,
    tokenizer: &ModelTokenizer,
) -> Detokenizer {
    detokenizer.flush(tokenizer).expect("flush");
    detokenizer
}

fn tokenizer() -> ModelTokenizer {
    let model = std::env::var_os("QWEN3_MODEL")
        .map_or_else(|| PathBuf::from("Qwen/Qwen3-0.6B"), PathBuf::from);
    let source = ModelSource::resolve(&model).expect("resolve model");
    ModelTokenizer::from_directory(&source.directory).expect("load tokenizer")
}
