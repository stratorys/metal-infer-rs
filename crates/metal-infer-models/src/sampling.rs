use half::f16;
use metal_infer_kernels::Tensor;

use crate::ModelError;

#[derive(Clone, Debug)]
pub struct GenerationOptions {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub stop_token_ids: Vec<u32>,
}

impl Default for GenerationOptions {
    fn default() -> Self {
        Self {
            max_tokens: 32,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
            stop_token_ids: Vec::new(),
        }
    }
}

pub struct TokenSampler {
    options: GenerationOptions,
    random: XorShift64,
}

impl TokenSampler {
    pub fn new(options: GenerationOptions) -> Result<Self, ModelError> {
        validate_generation_options(&options)?;
        let random = XorShift64::new(options.seed);
        Ok(Self { options, random })
    }

    pub fn sample(
        &mut self,
        logits: &Tensor,
    ) -> Result<u32, ModelError> {
        logits.with_f16_bits(|bits| {
            if self.options.temperature == 0.0 {
                bits.iter()
                    .enumerate()
                    .filter_map(|(index, bits)| {
                        let value = f16::from_bits(*bits).to_f32();
                        value.is_finite().then_some((index, value))
                    })
                    .max_by(|left, right| {
                        left.1
                            .total_cmp(&right.1)
                            .then_with(|| right.0.cmp(&left.0))
                    })
                    .map(|(index, _)| index as u32)
                    .ok_or(ModelError::NoFiniteLogit)
            } else {
                sample_token_f16(bits, &self.options, &mut self.random)
            }
        })?
    }
}

#[cfg(test)]
pub(crate) fn argmax(values: &[f32]) -> Result<u32, ModelError> {
    let (index, _) = values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .ok_or(ModelError::NoFiniteLogit)?;
    index.try_into().map_err(|_| ModelError::TokenIdOverflow)
}

pub(crate) fn validate_generation_options(options: &GenerationOptions) -> Result<(), ModelError> {
    if !options.temperature.is_finite() || options.temperature < 0.0 {
        return Err(ModelError::InvalidTemperature);
    }
    if !options.top_p.is_finite() || !(0.0..=1.0).contains(&options.top_p) {
        return Err(ModelError::InvalidTopP);
    }
    Ok(())
}

#[cfg(test)]
fn sample_token(
    values: &[f32],
    options: &GenerationOptions,
    random: &mut XorShift64,
) -> Result<u32, ModelError> {
    if options.temperature == 0.0 {
        return argmax(values);
    }
    let candidates: Vec<(usize, f32)> = values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .collect();
    sample_candidates(candidates, options, random)
}

pub(crate) fn sample_token_f16(
    bits: &[u16],
    options: &GenerationOptions,
    random: &mut XorShift64,
) -> Result<u32, ModelError> {
    let candidates: Vec<(usize, f32)> = bits
        .iter()
        .enumerate()
        .map(|(index, bits)| (index, f16::from_bits(*bits).to_f32()))
        .filter(|(_, value)| value.is_finite())
        .collect();
    sample_candidates(candidates, options, random)
}

fn sample_candidates(
    mut candidates: Vec<(usize, f32)>,
    options: &GenerationOptions,
    random: &mut XorShift64,
) -> Result<u32, ModelError> {
    if candidates.is_empty() {
        return Err(ModelError::NoFiniteLogit);
    }
    if options.top_k > 0 && candidates.len() > options.top_k {
        let mut original = candidates.clone();
        candidates.select_nth_unstable_by(options.top_k, |left, right| right.1.total_cmp(&left.1));
        let (top, rest) = candidates.split_at_mut(options.top_k);
        top.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
        let has_tie = top.windows(2).any(|pair| {
            pair.first()
                .zip(pair.get(1))
                .is_some_and(|(left, right)| left.1 == right.1)
        }) || top
            .last()
            .zip(rest.first())
            .is_some_and(|(last, next)| last.1 == next.1);
        if has_tie {
            original.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
            original.truncate(options.top_k);
            candidates = original;
        } else {
            candidates.truncate(options.top_k);
        }
    } else {
        candidates.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
    }
    let max_logit = candidates.first().ok_or(ModelError::NoFiniteLogit)?.1;
    let inverse_temperature = options.temperature.recip();
    let mut total = 0.0f64;
    for (_, value) in &mut candidates {
        *value = ((*value - max_logit) * inverse_temperature).exp();
        total += f64::from(*value);
    }
    if options.top_p < 1.0 {
        let threshold = total * f64::from(options.top_p);
        let mut cumulative = 0.0f64;
        let mut keep = 0usize;
        for (_, probability) in &candidates {
            cumulative += f64::from(*probability);
            keep += 1;
            if cumulative >= threshold {
                break;
            }
        }
        candidates.truncate(keep.max(1));
        total = candidates
            .iter()
            .map(|(_, probability)| f64::from(*probability))
            .sum();
    }
    let mut target = random.next_f64() * total;
    for (index, probability) in candidates {
        target -= f64::from(probability);
        if target <= 0.0 {
            return index.try_into().map_err(|_| ModelError::TokenIdOverflow);
        }
    }
    Err(ModelError::SamplingFailed)
}

pub(crate) struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    pub(crate) const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_f64(&mut self) -> f64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        (value as f64) / (u64::MAX as f64 + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{GenerationOptions, XorShift64, sample_token};

    #[test]
    fn zero_temperature_is_greedy() {
        let options = GenerationOptions::default();
        let token = sample_token(&[1.0, 4.0, 2.0], &options, &mut XorShift64::new(1));
        assert_eq!(token.expect("sampling should succeed"), 1);
    }

    #[test]
    fn top_k_one_is_greedy_even_with_temperature() {
        let options = GenerationOptions {
            temperature: 1.0,
            top_k: 1,
            ..GenerationOptions::default()
        };
        let token = sample_token(&[1.0, 4.0, 2.0], &options, &mut XorShift64::new(1));
        assert_eq!(token.expect("sampling should succeed"), 1);
    }
}
