use metal_infer_models::{ModelError, ModelTokenizer};

const REPLACEMENT_CHARACTER: char = '\u{FFFD}';

pub struct Detokenizer {
    tokens: Vec<u32>,
    prefix_offset: usize,
    read_offset: usize,
    text: String,
}

impl Detokenizer {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            tokens: Vec::with_capacity(capacity),
            prefix_offset: 0,
            read_offset: 0,
            text: String::new(),
        }
    }

    pub fn push(
        &mut self,
        token: u32,
        tokenizer: &ModelTokenizer,
    ) -> Result<usize, ModelError> {
        self.tokens.push(token);
        let delta = self
            .pending_delta(tokenizer)?
            .filter(|delta| !delta.ends_with(REPLACEMENT_CHARACTER));
        Ok(delta.map_or(0, |delta| self.commit(&delta)))
    }

    pub fn flush(
        &mut self,
        tokenizer: &ModelTokenizer,
    ) -> Result<(), ModelError> {
        if let Some(delta) = self.pending_delta(tokenizer)? {
            self.commit(&delta);
        }
        Ok(())
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub const fn token_count(&self) -> usize {
        self.tokens.len()
    }

    pub fn last_token(&self) -> Option<u32> {
        self.tokens.last().copied()
    }

    fn pending_delta(
        &self,
        tokenizer: &ModelTokenizer,
    ) -> Result<Option<String>, ModelError> {
        let before = tokenizer.decode(
            self.tokens
                .get(self.prefix_offset..self.read_offset)
                .unwrap_or_default(),
        )?;
        let after = tokenizer.decode(self.tokens.get(self.prefix_offset..).unwrap_or_default())?;
        Ok(after
            .get(before.len()..)
            .filter(|delta| !delta.is_empty())
            .map(str::to_owned))
    }

    fn commit(
        &mut self,
        delta: &str,
    ) -> usize {
        self.text.push_str(delta);
        self.prefix_offset = self.read_offset;
        self.read_offset = self.tokens.len();
        delta.len()
    }
}
