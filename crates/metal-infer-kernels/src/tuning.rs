use std::cell::RefCell;
use std::rc::Rc;

use metal_infer_runtime::CoreError;

use crate::Kernels;

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

#[derive(Clone, Copy)]
struct GemvRows {
    baseline: usize,
    tuned: usize,
}

impl GemvRows {
    const fn for_config(
        self,
        config: DecodeGemvConfig,
    ) -> usize {
        match config {
            DecodeGemvConfig::Baseline => self.baseline,
            DecodeGemvConfig::Tuned => self.tuned,
        }
    }
}

struct SingleShape {
    device: &'static str,
    n: usize,
    k: usize,
    rows: GemvRows,
}

struct FusedNormShape {
    device: &'static str,
    widths: [usize; 3],
    k: usize,
    rows: GemvRows,
}

const SINGLE_SHAPES: &[SingleShape] = &[
    SingleShape {
        device: "Apple M4 Pro",
        n: 1024,
        k: 1024,
        rows: GemvRows {
            baseline: 1,
            tuned: 1,
        },
    },
    SingleShape {
        device: "Apple M4 Pro",
        n: 1024,
        k: 2048,
        rows: GemvRows {
            baseline: 4,
            tuned: 2,
        },
    },
    SingleShape {
        device: "Apple M4 Pro",
        n: 1024,
        k: 3072,
        rows: GemvRows {
            baseline: 4,
            tuned: 1,
        },
    },
    SingleShape {
        device: "Apple M4 Pro",
        n: 151_936,
        k: 1024,
        rows: GemvRows {
            baseline: 0,
            tuned: 2,
        },
    },
];

const FUSED_NORM_SHAPES: &[FusedNormShape] = &[
    FusedNormShape {
        device: "Apple M4 Pro",
        widths: [2048, 1024, 1024],
        k: 1024,
        rows: GemvRows {
            baseline: 2,
            tuned: 1,
        },
    },
    FusedNormShape {
        device: "Apple M4 Pro",
        widths: [3072, 3072, 0],
        k: 1024,
        rows: GemvRows {
            baseline: 2,
            tuned: 1,
        },
    },
];

const M4_PRO_TWO_QUERY_HEADS_FLASH_DECODE_BLOCKS: &[FlashDecodeBlock] = &[
    FlashDecodeBlock {
        max_length: 1024,
        block: 64,
    },
    FlashDecodeBlock {
        max_length: 3072,
        block: 128,
    },
    FlashDecodeBlock {
        max_length: usize::MAX,
        block: 256,
    },
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
        .find(|shape| shape.device == device && shape.n == n && shape.k == k)
        .map_or(fallback, |shape| shape.rows.for_config(config))
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
        .find(|shape| shape.device == device && shape.widths == widths && shape.k == k)
        .map_or(fallback, |shape| shape.rows.for_config(config))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FlashDecodeBlock {
    pub max_length: usize,
    pub block: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct KernelSelection {
    pub decode_gemv: Option<DecodeGemvConfig>,
    pub flash_decode_blocks: Vec<FlashDecodeBlock>,
}

impl KernelSelection {
    pub fn validate(&self) -> Result<(), CoreError> {
        for entry in &self.flash_decode_blocks {
            if !matches!(entry.block, 32 | 64 | 128 | 256) {
                return Err(CoreError::FlashDecodeBlockSize);
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

    pub(crate) fn flash_decode_block_for_length(
        &self,
        length: usize,
    ) -> usize {
        self.tuning
            .selection
            .borrow()
            .flash_decode_blocks
            .iter()
            .find(|entry| length <= entry.max_length)
            .map_or(64, |entry| entry.block)
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

    pub fn device_selection(
        &self,
        query_heads: usize,
        kv_heads: usize,
    ) -> KernelSelection {
        if !self.tuning.is_m4_pro {
            return KernelSelection::default();
        }
        KernelSelection {
            decode_gemv: Some(DecodeGemvConfig::Tuned),
            flash_decode_blocks: if query_heads == kv_heads * 2 {
                M4_PRO_TWO_QUERY_HEADS_FLASH_DECODE_BLOCKS.to_vec()
            } else {
                Vec::new()
            },
        }
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
