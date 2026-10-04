use crate::KernelError;

const M4_PRO_DEVICE_NAME: &str = "Apple M4 Pro";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceProfile {
    M4Pro,
    Generic,
}

impl DeviceProfile {
    pub fn from_device_name(name: &str) -> Self {
        if name == M4_PRO_DEVICE_NAME {
            Self::M4Pro
        } else {
            Self::Generic
        }
    }
}

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
    device: DeviceProfile,
    n: usize,
    k: usize,
    rows: GemvRows,
}

struct FusedNormShape {
    device: DeviceProfile,
    widths: [usize; 3],
    k: usize,
    rows: GemvRows,
}

const SINGLE_SHAPES: &[SingleShape] = &[
    SingleShape {
        device: DeviceProfile::M4Pro,
        n: 1024,
        k: 1024,
        rows: GemvRows {
            baseline: 1,
            tuned: 1,
        },
    },
    SingleShape {
        device: DeviceProfile::M4Pro,
        n: 1024,
        k: 2048,
        rows: GemvRows {
            baseline: 4,
            tuned: 2,
        },
    },
    SingleShape {
        device: DeviceProfile::M4Pro,
        n: 1024,
        k: 3072,
        rows: GemvRows {
            baseline: 4,
            tuned: 1,
        },
    },
    SingleShape {
        device: DeviceProfile::M4Pro,
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
        device: DeviceProfile::M4Pro,
        widths: [2048, 1024, 1024],
        k: 1024,
        rows: GemvRows {
            baseline: 2,
            tuned: 1,
        },
    },
    FusedNormShape {
        device: DeviceProfile::M4Pro,
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

pub fn single_rows(
    device: DeviceProfile,
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

pub fn fused_norm_rows(
    device: DeviceProfile,
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
    pub fn for_device(
        device: DeviceProfile,
        query_heads: usize,
        kv_heads: usize,
    ) -> Self {
        match device {
            DeviceProfile::Generic => Self::default(),
            DeviceProfile::M4Pro => Self {
                decode_gemv: Some(DecodeGemvConfig::Tuned),
                flash_decode_blocks: if query_heads == kv_heads * 2 {
                    M4_PRO_TWO_QUERY_HEADS_FLASH_DECODE_BLOCKS.to_vec()
                } else {
                    Vec::new()
                },
            },
        }
    }

    pub fn validate(&self) -> Result<(), KernelError> {
        for entry in &self.flash_decode_blocks {
            if !matches!(entry.block, 32 | 64 | 128 | 256) {
                return Err(KernelError::FlashDecodeBlockSize);
            }
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct Tuning {
    device: DeviceProfile,
    selection: KernelSelection,
}

impl Tuning {
    pub fn new(device_name: &str) -> Self {
        Self {
            device: DeviceProfile::from_device_name(device_name),
            selection: KernelSelection::default(),
        }
    }

    pub const fn device(&self) -> DeviceProfile {
        self.device
    }

    pub const fn selection(&self) -> &KernelSelection {
        &self.selection
    }

    pub fn with_selection(
        self,
        selection: KernelSelection,
    ) -> Result<Self, KernelError> {
        selection.validate()?;
        Ok(Self { selection, ..self })
    }

    pub const fn is_m4_pro(&self) -> bool {
        matches!(self.device, DeviceProfile::M4Pro)
    }

    pub fn flash_decode_block_for_length(
        &self,
        length: usize,
    ) -> usize {
        self.selection
            .flash_decode_blocks
            .iter()
            .find(|entry| length <= entry.max_length)
            .map_or(64, |entry| entry.block)
    }

    pub fn matvec_rows_for_shape(
        &self,
        n: usize,
        k: usize,
        vocabulary: bool,
    ) -> usize {
        single_rows(
            self.device,
            n,
            k,
            if vocabulary { 0 } else { 4 },
            self.selection.decode_gemv,
        )
    }

    pub fn fused_norm_matvec_rows_for_shape(
        &self,
        widths: [usize; 3],
        k: usize,
    ) -> usize {
        fused_norm_rows(self.device, widths, k, 2, self.selection.decode_gemv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuned_shapes_and_fallbacks() {
        let tuned = Some(DecodeGemvConfig::Tuned);
        let m4_pro = DeviceProfile::M4Pro;
        assert_eq!(single_rows(m4_pro, 1024, 3072, 4, tuned), 1);
        assert_eq!(single_rows(m4_pro, 1024, 2048, 4, tuned), 2);
        assert_eq!(single_rows(m4_pro, 151_936, 1024, 0, tuned), 2);
        assert_eq!(single_rows(DeviceProfile::Generic, 1024, 3072, 4, tuned), 4);
        assert_eq!(fused_norm_rows(m4_pro, [3072, 3072, 0], 1024, 2, tuned), 1);
        assert_eq!(fused_norm_rows(m4_pro, [4096, 4096, 0], 1024, 2, tuned), 2);
    }

    #[test]
    fn missing_config_and_baseline_use_their_rows() {
        assert_eq!(single_rows(DeviceProfile::M4Pro, 1024, 3072, 4, None), 4);
        assert_eq!(
            single_rows(
                DeviceProfile::M4Pro,
                1024,
                3072,
                4,
                Some(DecodeGemvConfig::Baseline)
            ),
            4
        );
    }

    #[test]
    fn device_profile_is_derived_from_the_device_name() {
        assert_eq!(
            DeviceProfile::from_device_name("Apple M4 Pro"),
            DeviceProfile::M4Pro
        );
        assert_eq!(
            DeviceProfile::from_device_name("Apple M3"),
            DeviceProfile::Generic
        );
    }

    #[test]
    fn device_selection_tunes_only_the_m4_pro() {
        assert_eq!(
            KernelSelection::for_device(DeviceProfile::Generic, 16, 8),
            KernelSelection::default()
        );
        let tuned = KernelSelection::for_device(DeviceProfile::M4Pro, 16, 8);
        assert_eq!(tuned.decode_gemv, Some(DecodeGemvConfig::Tuned));
        assert_eq!(
            tuned.flash_decode_blocks,
            M4_PRO_TWO_QUERY_HEADS_FLASH_DECODE_BLOCKS.to_vec()
        );
        assert!(
            KernelSelection::for_device(DeviceProfile::M4Pro, 32, 8)
                .flash_decode_blocks
                .is_empty(),
            "flash decode blocks are tuned only for two query heads per KV head"
        );
    }
}
