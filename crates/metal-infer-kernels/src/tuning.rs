use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

use metal_infer_runtime::{CoreError, Tensor};

use crate::{AttentionConfig, KernelBatch, Kernels};

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
    manual: bool,
    auto_enabled: bool,
) -> usize {
    if manual || (!auto_enabled && config.is_none()) {
        return fallback;
    }
    SINGLE_SHAPES
        .iter()
        .find(|&&(target, shape_n, shape_k, _, _)| target == device && shape_n == n && shape_k == k)
        .map_or(fallback, |&(_, _, _, baseline, tuned)| match config {
            Some(DecodeGemvConfig::Baseline) => baseline,
            Some(DecodeGemvConfig::Tuned) => tuned,
            None if n == 1024 && k == 1024 => 1,
            None => fallback,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MatmulBackend {
    #[default]
    Auto,
    ReferenceMsl,
    NativeMsl,
    Mps,
}

#[derive(Clone, Copy)]
pub(crate) struct AutoMatvecRows {
    pub(crate) single: usize,
    pub(crate) fused2: usize,
    pub(crate) fused3: usize,
    pub(crate) vocab: usize,
}

impl Default for AutoMatvecRows {
    fn default() -> Self {
        Self {
            single: 4,
            fused2: 2,
            fused3: 2,
            vocab: 0,
        }
    }
}

impl MatmulBackend {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ReferenceMsl => "reference-msl",
            Self::NativeMsl => "native-msl",
            Self::Mps => "mps",
        }
    }
}

#[derive(Clone)]
pub(crate) struct Tuning {
    device_name: String,
    is_m4_pro: bool,
    matmul_backend: Rc<Cell<MatmulBackend>>,
    auto_matvec_rows: Rc<Cell<AutoMatvecRows>>,
    manual_auto_matvec_rows: Rc<Cell<bool>>,
    auto_matvec_enabled: Rc<Cell<bool>>,
    auto_matvec_split_k: Rc<Cell<usize>>,
    auto_matvec_half8: Rc<Cell<bool>>,
    fused_norm_matvec_rows: Rc<Cell<(usize, usize)>>,
    shared_gate_up_input: Rc<Cell<bool>>,
    decode_gemv_config: Rc<Cell<Option<DecodeGemvConfig>>>,
    flash_decode_blocks: Rc<RefCell<Vec<(usize, usize, usize)>>>,
}

impl Tuning {
    pub(crate) fn new(device_name: &str) -> Self {
        Self {
            device_name: device_name.to_owned(),
            is_m4_pro: device_name == "Apple M4 Pro",
            matmul_backend: Rc::new(Cell::new(MatmulBackend::Auto)),
            auto_matvec_rows: Rc::new(Cell::new(AutoMatvecRows::default())),
            manual_auto_matvec_rows: Rc::new(Cell::new(false)),
            auto_matvec_enabled: Rc::new(Cell::new(true)),
            auto_matvec_split_k: Rc::new(Cell::new(1)),
            auto_matvec_half8: Rc::new(Cell::new(false)),
            fused_norm_matvec_rows: Rc::new(Cell::new((2, 2))),
            shared_gate_up_input: Rc::new(Cell::new(false)),
            decode_gemv_config: Rc::new(Cell::new(None)),
            flash_decode_blocks: Rc::new(RefCell::new(Vec::new())),
        }
    }
}

impl Kernels {
    pub(crate) fn is_m4_pro(&self) -> bool {
        self.tuning.is_m4_pro
    }

    pub(crate) fn flash_decode_configuration_for_length(
        &self,
        length: usize,
    ) -> (usize, usize) {
        self.tuning
            .flash_decode_blocks
            .borrow()
            .iter()
            .find(|(limit, _, _)| length <= *limit)
            .map_or((64, 256), |(_, block, threads)| (*block, *threads))
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
            chosen.push((length, best.1, best.2));
        }
        *self.tuning.flash_decode_blocks.borrow_mut() = chosen;
        Ok(())
    }

    pub fn set_matmul_backend(
        &self,
        backend: MatmulBackend,
    ) {
        self.tuning.matmul_backend.set(backend);
    }

    pub fn matmul_backend(&self) -> MatmulBackend {
        self.tuning.matmul_backend.get()
    }

    pub(crate) fn auto_matvec_rows(&self) -> AutoMatvecRows {
        if self.tuning.auto_matvec_enabled.get() {
            self.tuning.auto_matvec_rows.get()
        } else {
            AutoMatvecRows::default()
        }
    }

    pub fn set_decode_gemv_config(
        &self,
        config: DecodeGemvConfig,
    ) {
        self.tuning.decode_gemv_config.set(Some(config));
    }

    pub fn decode_gemv_config(&self) -> Option<DecodeGemvConfig> {
        self.tuning.decode_gemv_config.get()
    }

    pub(crate) fn auto_matvec_rows_for_shape(
        &self,
        n: usize,
        k: usize,
        vocabulary: bool,
    ) -> usize {
        let selected = self.auto_matvec_rows();
        let fallback = if vocabulary {
            selected.vocab
        } else {
            selected.single
        };
        single_rows(
            &self.tuning.device_name,
            n,
            k,
            fallback,
            self.tuning.decode_gemv_config.get(),
            self.tuning.manual_auto_matvec_rows.get(),
            self.tuning.auto_matvec_enabled.get(),
        )
    }

    pub fn set_auto_matvec_enabled(
        &self,
        enabled: bool,
    ) {
        self.tuning.auto_matvec_enabled.set(enabled);
    }

    pub fn set_auto_matvec_split_k(
        &self,
        splits: usize,
    ) -> Result<(), CoreError> {
        if !matches!(splits, 1 | 2 | 4 | 8) {
            return Err(CoreError::Shape(
                "split-K count must be 1, 2, 4, or 8".into(),
            ));
        }
        self.tuning.auto_matvec_split_k.set(splits);
        Ok(())
    }

    pub(crate) fn auto_matvec_split_k(&self) -> usize {
        self.tuning.auto_matvec_split_k.get()
    }

    pub fn set_auto_matvec_half8(
        &self,
        enabled: bool,
    ) {
        self.tuning.auto_matvec_half8.set(enabled);
    }

    pub(crate) fn auto_matvec_half8(&self) -> bool {
        self.tuning.auto_matvec_half8.get()
    }

    /// Rows per SIMD group for the decode-only fused RMSNorm projections.
    /// The first value selects QKV and the second selects gate/up.
    pub fn set_fused_norm_matvec_rows(
        &self,
        qkv: usize,
        gate_up: usize,
    ) -> Result<(), CoreError> {
        if !matches!(qkv, 1 | 2 | 4 | 8) || !matches!(gate_up, 1 | 2 | 4 | 8) {
            return Err(CoreError::Shape(
                "fused norm matvec rows must be 1, 2, 4, or 8".into(),
            ));
        }
        self.tuning.fused_norm_matvec_rows.set((qkv, gate_up));
        Ok(())
    }

    /// Reuse normalized gate/up input within each threadgroup of the fused decode GEMV.
    pub fn set_shared_gate_up_input(
        &self,
        enabled: bool,
    ) {
        self.tuning.shared_gate_up_input.set(enabled);
    }

    pub fn shared_gate_up_input(&self) -> bool {
        self.tuning.shared_gate_up_input.get()
    }

    pub(crate) fn fused_norm_matvec_rows_for_shape(
        &self,
        widths: [usize; 3],
        k: usize,
    ) -> usize {
        let [_, _, n2] = widths;
        let fallback = if n2 == 0 {
            self.tuning.fused_norm_matvec_rows.get().1
        } else {
            self.tuning.fused_norm_matvec_rows.get().0
        };
        fused_norm_rows(
            &self.tuning.device_name,
            widths,
            k,
            fallback,
            self.tuning.decode_gemv_config.get(),
        )
    }

    pub fn set_auto_matvec_rows(
        &self,
        single: usize,
        fused2: usize,
        fused3: usize,
        vocab: usize,
    ) -> Result<(), CoreError> {
        if !matches!(single, 0 | 1 | 2 | 4 | 8)
            || !matches!(fused2, 0 | 2 | 4 | 8)
            || !matches!(fused3, 0 | 2 | 4 | 8)
            || !matches!(vocab, 0 | 2 | 4 | 8)
        {
            return Err(CoreError::Shape("invalid auto matvec row count".into()));
        }
        self.tuning.auto_matvec_rows.set(AutoMatvecRows {
            single,
            fused2,
            fused3,
            vocab,
        });
        self.tuning.manual_auto_matvec_rows.set(true);
        Ok(())
    }

    fn measure_matvec_variant(
        &self,
        mut dispatch: impl FnMut(&mut KernelBatch<'_>) -> Result<(), CoreError>,
    ) -> Result<u128, CoreError> {
        let mut samples = [0_u128; 3];
        for sample in &mut samples {
            let mut batch = self.begin_batch()?;
            dispatch(&mut batch)?;
            *sample = batch.finish()?.gpu_time.as_nanos();
        }
        samples.sort_unstable();
        Ok(samples[1])
    }

    pub fn tune_auto_matvec_variants(
        &self,
        single: &Tensor,
        fused2: [&Tensor; 2],
        fused3: [&Tensor; 3],
        vocab: &Tensor,
    ) -> Result<(), CoreError> {
        if !self.tuning.is_m4_pro || self.matmul_backend() != MatmulBackend::Auto {
            return Ok(());
        }
        let started = Instant::now();
        let input_for = |weight: &Tensor| -> Result<Tensor, CoreError> {
            let width = *weight
                .shape()
                .get(1)
                .ok_or_else(|| CoreError::Shape("matvec tuning requires matrix weights".into()))?;
            self.context()
                .tensor_f16_bits(&vec![0x3800; width], &[1, width])
        };
        let single_input = input_for(single)?;
        let fused2_input = input_for(fused2[0])?;
        let fused3_input = input_for(fused3[0])?;
        let vocab_input = input_for(vocab)?;
        let mut selected = AutoMatvecRows::default();
        for family in 0..4 {
            let candidates: &[usize] = if family == 3 {
                &[0, 2, 4, 8]
            } else if family == 0 {
                &[4, 0, 2, 8]
            } else {
                &[2, 0, 4, 8]
            };
            let mut best = (
                u128::MAX,
                *candidates
                    .first()
                    .ok_or_else(|| CoreError::Profiling("no GEMV candidates".into()))?,
            );
            for &rows in candidates {
                if started.elapsed() >= Duration::from_secs(3) {
                    break;
                }
                match family {
                    0 => selected.single = rows,
                    1 => selected.fused2 = rows,
                    2 => selected.fused3 = rows,
                    _ => selected.vocab = rows,
                }
                self.tuning.auto_matvec_rows.set(selected);
                let elapsed = match family {
                    0 => self.measure_matvec_variant(|batch| {
                        batch.matmul(&single_input, single)?;
                        Ok(())
                    }),
                    1 => self.measure_matvec_variant(|batch| {
                        batch.matmul2(&fused2_input, fused2[0], fused2[1])?;
                        Ok(())
                    }),
                    2 => self.measure_matvec_variant(|batch| {
                        batch.matmul3(&fused3_input, fused3[0], fused3[1], fused3[2])?;
                        Ok(())
                    }),
                    _ => self.measure_matvec_variant(|batch| {
                        batch.matmul(&vocab_input, vocab)?;
                        Ok(())
                    }),
                };
                let Ok(elapsed) = elapsed else {
                    continue;
                };
                if elapsed < best.0 {
                    best = (elapsed, rows);
                }
            }
            match family {
                0 => selected.single = best.1,
                1 => selected.fused2 = best.1,
                2 => selected.fused3 = best.1,
                _ => selected.vocab = best.1,
            }
            self.tuning.auto_matvec_rows.set(selected);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuned_shapes_and_fallbacks() {
        assert_eq!(
            single_rows(
                "Apple M4 Pro",
                1024,
                3072,
                4,
                Some(DecodeGemvConfig::Tuned),
                false,
                true
            ),
            1
        );
        assert_eq!(
            single_rows(
                "Apple M4 Pro",
                151_936,
                1024,
                0,
                Some(DecodeGemvConfig::Tuned),
                false,
                true
            ),
            2
        );
        assert_eq!(
            single_rows(
                "Apple M3",
                1024,
                3072,
                4,
                Some(DecodeGemvConfig::Tuned),
                false,
                true
            ),
            4
        );
        assert_eq!(
            fused_norm_rows(
                "Apple M4 Pro",
                [3072, 3072, 0],
                1024,
                2,
                Some(DecodeGemvConfig::Tuned)
            ),
            1
        );
        assert_eq!(
            fused_norm_rows(
                "Apple M4 Pro",
                [4096, 4096, 0],
                1024,
                2,
                Some(DecodeGemvConfig::Tuned)
            ),
            2
        );
    }

    #[test]
    fn explicit_shape_choice_survives_general_autotuning() {
        let tuned = Some(DecodeGemvConfig::Tuned);
        assert_eq!(
            single_rows("Apple M4 Pro", 1024, 2048, 4, tuned, false, false),
            2
        );
        assert_eq!(
            single_rows("Apple M4 Pro", 1024, 3072, 4, tuned, false, false),
            1
        );
        assert_eq!(
            single_rows("Apple M4 Pro", 151_936, 1024, 0, tuned, false, false),
            2
        );
        assert_eq!(
            single_rows("Apple M4 Pro", 1024, 3072, 4, tuned, true, false),
            4
        );
        assert_eq!(
            single_rows("Apple M4 Pro", 1024, 3072, 4, None, false, false),
            4
        );
        assert_eq!(
            single_rows(
                "Apple M4 Pro",
                1024,
                3072,
                4,
                Some(DecodeGemvConfig::Baseline),
                false,
                false
            ),
            4
        );
    }
}
