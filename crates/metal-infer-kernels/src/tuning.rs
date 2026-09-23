use std::cell::RefCell;
use std::rc::Rc;
use std::time::{Duration, Instant};

use metal_infer_runtime::CoreError;

use crate::{AttentionConfig, Kernels};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeGemvConfig {
    Baseline,
    Tuned,
}

impl DecodeGemvConfig {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Tuned => "tuned",
        }
    }
}

// (device, N, K, baseline rows, tuned rows). The tuned values were measured
// with distinct FP16 matrices in a rotating working set larger than 512 MiB.
const SINGLE_SHAPES: &[(&str, usize, usize, usize, usize)] = &[
    ("Apple M4 Pro", 1024, 1024, 1, 1),
    ("Apple M4 Pro", 1024, 2048, 4, 2),
    ("Apple M4 Pro", 1024, 3072, 4, 1),
    ("Apple M4 Pro", 151_936, 1024, 0, 2),
];

// (device, output widths, K, baseline rows, tuned rows).
const FUSED_NORM_SHAPES: &[(&str, [usize; 3], usize, usize, usize)] = &[
    ("Apple M4 Pro", [2048, 1024, 1024], 1024, 2, 1),
    ("Apple M4 Pro", [3072, 3072, 0], 1024, 2, 1),
];

pub(crate) fn single_rows(
    device: &str,
    n: usize,
    k: usize,
    fallback: usize,
    config: Option<DecodeGemvConfig>,
) -> usize {
    let Some(config) = config else {
        return fallback;
    };
    SINGLE_SHAPES
        .iter()
        .find(|&&(target, shape_n, shape_k, _, _)| target == device && shape_n == n && shape_k == k)
        .map_or(fallback, |&(_, _, _, baseline, tuned)| match config {
            DecodeGemvConfig::Baseline => baseline,
            DecodeGemvConfig::Tuned => tuned,
        })
}

pub(crate) fn fused_norm_rows(
    device: &str,
    widths: [usize; 3],
    k: usize,
    fallback: usize,
    config: Option<DecodeGemvConfig>,
) -> usize {
    let Some(config) = config else {
        return fallback;
    };
    FUSED_NORM_SHAPES
        .iter()
        .find(|&&(target, shape_widths, shape_k, _, _)| {
            target == device && shape_widths == widths && shape_k == k
        })
        .map_or(fallback, |&(_, _, _, baseline, tuned)| match config {
            DecodeGemvConfig::Baseline => baseline,
            DecodeGemvConfig::Tuned => tuned,
        })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlashDecodeBlock {
    pub max_length: usize,
    pub block: usize,
    pub threads: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KernelSelection {
    pub decode_gemv: Option<DecodeGemvConfig>,
    pub flash_decode_blocks: Vec<FlashDecodeBlock>,
}

impl KernelSelection {
    pub fn validate(&self) -> Result<(), CoreError> {
        for entry in &self.flash_decode_blocks {
            if !matches!(entry.block, 32 | 64 | 128 | 256) || !matches!(entry.threads, 128 | 256) {
                return Err(CoreError::Shape(
                    "flash decode blocks must use 32, 64, 128, or 256 keys and 128 or 256 threads"
                        .into(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct Tuning {
    device_name: String,
    is_m4_pro: bool,
    selection: Rc<RefCell<KernelSelection>>,
}

impl Tuning {
    pub(crate) fn new(device_name: &str) -> Self {
        Self {
            device_name: device_name.to_owned(),
            is_m4_pro: device_name == "Apple M4 Pro",
            selection: Rc::new(RefCell::new(KernelSelection::default())),
        }
    }
}

impl Kernels {
    pub fn selection(&self) -> KernelSelection {
        self.tuning.selection.borrow().clone()
    }

    pub fn select(
        &self,
        selection: &KernelSelection,
    ) -> Result<(), CoreError> {
        selection.validate()?;
        *self.tuning.selection.borrow_mut() = selection.clone();
        Ok(())
    }

    pub(crate) fn is_m4_pro(&self) -> bool {
        self.tuning.is_m4_pro
    }

    pub(crate) fn flash_decode_configuration_for_length(
        &self,
        length: usize,
    ) -> (usize, usize) {
        self.tuning
            .selection
            .borrow()
            .flash_decode_blocks
            .iter()
            .find(|entry| length <= entry.max_length)
            .map_or((64, 256), |entry| (entry.block, entry.threads))
    }

    pub(crate) fn matvec_rows_for_shape(
        &self,
        n: usize,
        k: usize,
        vocabulary: bool,
    ) -> usize {
        single_rows(
            &self.tuning.device_name,
            n,
            k,
            if vocabulary { 0 } else { 4 },
            self.tuning.selection.borrow().decode_gemv,
        )
    }

    pub(crate) fn fused_norm_matvec_rows_for_shape(
        &self,
        widths: [usize; 3],
        k: usize,
    ) -> usize {
        fused_norm_rows(
            &self.tuning.device_name,
            widths,
            k,
            2,
            self.tuning.selection.borrow().decode_gemv,
        )
    }

    pub fn tune_flash_decode(
        &self,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<(), CoreError> {
        if !self.tuning.is_m4_pro || query_heads != kv_heads * 2 || head_dim == 0 {
            return Ok(());
        }
        let started = Instant::now();
        let lengths = [512, 640, 1024, 2048, 4096, 8192];
        let maximum = 8192;
        let query = self.context().tensor_f16_bits(
            &vec![0x3c00; query_heads * head_dim],
            &[1, query_heads, head_dim],
        )?;
        let cache_values = vec![0x3800; maximum * kv_heads * head_dim];
        let key = self
            .context()
            .tensor_f16_bits(&cache_values, &[maximum, kv_heads, head_dim])?;
        let value = self
            .context()
            .tensor_f16_bits(&cache_values, &[maximum, kv_heads, head_dim])?;
        let mut chosen = Vec::with_capacity(lengths.len());
        for length in lengths {
            if started.elapsed() >= Duration::from_secs(2) {
                break;
            }
            let active_key = key.prefix(length)?;
            let active_value = value.prefix(length)?;
            let config = AttentionConfig {
                query_heads,
                kv_heads,
                head_dim,
                causal: true,
                query_offset: length - 1,
            };
            let mut best = (u128::MAX, 64, 256);
            for (block, threads) in [
                (64, 256),
                (32, 256),
                (128, 256),
                (256, 256),
                (64, 128),
                (32, 128),
                (128, 128),
                (256, 128),
            ] {
                if started.elapsed() >= Duration::from_secs(2) {
                    break;
                }
                let mut samples = [0_u128; 3];
                for sample in &mut samples {
                    let mut batch = self.begin_batch()?;
                    batch.attention_flash_decode_with_configuration(
                        &query,
                        &active_key,
                        &active_value,
                        config,
                        block,
                        threads,
                    )?;
                    *sample = batch.finish()?.gpu_time.as_nanos();
                }
                samples.sort_unstable();
                if samples[1] < best.0 {
                    best = (samples[1], block, threads);
                }
            }
            chosen.push(FlashDecodeBlock {
                max_length: length,
                block: best.1,
                threads: best.2,
            });
        }
        self.tuning.selection.borrow_mut().flash_decode_blocks = chosen;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuned_shapes_and_fallbacks() {
        let tuned = Some(DecodeGemvConfig::Tuned);
        assert_eq!(single_rows("Apple M4 Pro", 1024, 3072, 4, tuned), 1);
        assert_eq!(single_rows("Apple M4 Pro", 1024, 2048, 4, tuned), 2);
        assert_eq!(single_rows("Apple M4 Pro", 151_936, 1024, 0, tuned), 2);
        assert_eq!(single_rows("Apple M3", 1024, 3072, 4, tuned), 4);
        assert_eq!(
            fused_norm_rows("Apple M4 Pro", [3072, 3072, 0], 1024, 2, tuned),
            1
        );
        assert_eq!(
            fused_norm_rows("Apple M4 Pro", [4096, 4096, 0], 1024, 2, tuned),
            2
        );
    }

    #[test]
    fn missing_config_and_baseline_use_their_rows() {
        assert_eq!(single_rows("Apple M4 Pro", 1024, 3072, 4, None), 4);
        assert_eq!(
            single_rows(
                "Apple M4 Pro",
                1024,
                3072,
                4,
                Some(DecodeGemvConfig::Baseline)
            ),
            4
        );
    }
}
