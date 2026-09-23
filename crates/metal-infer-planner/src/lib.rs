use metal_infer_kernels::{
    AttentionKind, DecodeGemvConfig, DeviceProfile, FlashDecodeBlock, KernelSelection,
};
use thiserror::Error;

const FLASH_PREFILL_MIN_TOKENS: usize = 32;
const FLASH_DECODE_MIN_LENGTH: usize = 256;
const FLASH_DECODE_QUERY_HEADS_PER_KV: usize = 2;
const DECODE_NORM_WIDTH_MULTIPLE: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Fusions {
    pub qkv: bool,
    pub gate_up: bool,
    pub add_rms_norm: bool,
    pub qk_rope_cache: bool,
    pub decode_norm: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Plan {
    pub kernels: KernelSelection,
    pub fusions: Fusions,
    pub attention: AttentionKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanInputs {
    pub device: DeviceProfile,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub hidden_size: usize,
}

#[derive(Debug, Error)]
pub enum PlanError {
    #[error("plan override must have the form key=value")]
    Syntax,
    #[error("unknown plan key")]
    UnknownKey,
    #[error("plan switch must be on or off")]
    InvalidSwitch,
    #[error("attention must be reference, tiled, flash-prefill, decode-split-kv, or flash-decode")]
    InvalidAttention,
    #[error("gemv.config must be none, baseline, or tuned")]
    InvalidGemvConfig,
    #[error(
        "flash_decode.blocks must be default, BLOCK, or LENGTH:BLOCK entries separated by commas"
    )]
    InvalidFlashDecodeBlocks,
}

impl Plan {
    pub fn resolve(
        inputs: PlanInputs,
        overrides: &[String],
    ) -> Result<Self, PlanError> {
        overrides
            .iter()
            .try_fold(Self::device_default(inputs), |mut plan, assignment| {
                plan.apply_override(assignment)?;
                Ok(plan)
            })
    }

    pub fn device_default(inputs: PlanInputs) -> Self {
        Self {
            kernels: KernelSelection::for_device(
                inputs.device,
                inputs.query_heads,
                inputs.kv_heads,
            ),
            fusions: Fusions {
                qkv: true,
                gate_up: true,
                add_rms_norm: true,
                qk_rope_cache: true,
                decode_norm: matches!(inputs.device, DeviceProfile::M4Pro)
                    && inputs
                        .hidden_size
                        .is_multiple_of(DECODE_NORM_WIDTH_MULTIPLE),
            },
            attention: AttentionKind::Tiled,
        }
    }

    pub const fn attention_for_tokens(
        &self,
        tokens: usize,
        active_length: usize,
        query_heads_per_kv: usize,
    ) -> AttentionKind {
        attention_kind_for_tokens(self.attention, tokens, active_length, query_heads_per_kv)
    }

    pub fn apply_override(
        &mut self,
        assignment: &str,
    ) -> Result<(), PlanError> {
        let (key, value) = assignment.split_once('=').ok_or(PlanError::Syntax)?;
        let key = key.trim();
        let value = value.trim();
        let kernels = &mut self.kernels;
        match key {
            "attention" => self.attention = parse_attention(value)?,
            "fusion.qkv" => self.fusions.qkv = parse_switch(value)?,
            "fusion.gate_up" => self.fusions.gate_up = parse_switch(value)?,
            "fusion.add_rms_norm" => self.fusions.add_rms_norm = parse_switch(value)?,
            "fusion.qk_rope_cache" => self.fusions.qk_rope_cache = parse_switch(value)?,
            "fusion.decode_norm" => self.fusions.decode_norm = parse_switch(value)?,
            "gemv.config" => kernels.decode_gemv = parse_gemv_config(value)?,
            "flash_decode.blocks" => kernels.flash_decode_blocks = parse_blocks(value)?,
            _ => return Err(PlanError::UnknownKey),
        }
        Ok(())
    }

    pub fn entries(&self) -> Vec<(&'static str, String)> {
        let kernels = &self.kernels;
        vec![
            ("attention", attention_name(self.attention).to_owned()),
            ("fusion.qkv", switch_name(self.fusions.qkv)),
            ("fusion.gate_up", switch_name(self.fusions.gate_up)),
            (
                "fusion.add_rms_norm",
                switch_name(self.fusions.add_rms_norm),
            ),
            (
                "fusion.qk_rope_cache",
                switch_name(self.fusions.qk_rope_cache),
            ),
            ("fusion.decode_norm", switch_name(self.fusions.decode_norm)),
            (
                "gemv.config",
                kernels
                    .decode_gemv
                    .map_or("none", DecodeGemvConfig::name)
                    .to_owned(),
            ),
            (
                "flash_decode.blocks",
                blocks_name(&kernels.flash_decode_blocks),
            ),
        ]
    }
}

const fn attention_kind_for_tokens(
    configured: AttentionKind,
    tokens: usize,
    active_length: usize,
    query_heads_per_kv: usize,
) -> AttentionKind {
    let flash_decode = active_length >= FLASH_DECODE_MIN_LENGTH
        && query_heads_per_kv == FLASH_DECODE_QUERY_HEADS_PER_KV;
    match (configured, tokens) {
        (AttentionKind::Tiled, FLASH_PREFILL_MIN_TOKENS..) => AttentionKind::FlashPrefill,
        (AttentionKind::Tiled | AttentionKind::FlashPrefill, 1) if flash_decode => {
            AttentionKind::FlashDecode
        }
        (AttentionKind::Tiled | AttentionKind::DecodeSplitKv | AttentionKind::FlashPrefill, 1) => {
            AttentionKind::DecodeSplitKv
        }
        (AttentionKind::DecodeSplitKv, _) => AttentionKind::Tiled,
        (AttentionKind::FlashDecode, 1)
            if query_heads_per_kv == FLASH_DECODE_QUERY_HEADS_PER_KV =>
        {
            AttentionKind::FlashDecode
        }
        (AttentionKind::FlashDecode, 1) => AttentionKind::DecodeSplitKv,
        (AttentionKind::FlashDecode, _) => AttentionKind::Tiled,
        (kind, _) => kind,
    }
}

fn parse_switch(value: &str) -> Result<bool, PlanError> {
    match value {
        "on" | "true" => Ok(true),
        "off" | "false" => Ok(false),
        _ => Err(PlanError::InvalidSwitch),
    }
}

fn switch_name(enabled: bool) -> String {
    String::from(if enabled { "on" } else { "off" })
}

fn parse_attention(value: &str) -> Result<AttentionKind, PlanError> {
    match value {
        "reference" => Ok(AttentionKind::Reference),
        "tiled" => Ok(AttentionKind::Tiled),
        "flash-prefill" => Ok(AttentionKind::FlashPrefill),
        "decode-split-kv" => Ok(AttentionKind::DecodeSplitKv),
        "flash-decode" => Ok(AttentionKind::FlashDecode),
        _ => Err(PlanError::InvalidAttention),
    }
}

const fn attention_name(kind: AttentionKind) -> &'static str {
    match kind {
        AttentionKind::Reference => "reference",
        AttentionKind::Tiled => "tiled",
        AttentionKind::FlashPrefill => "flash-prefill",
        AttentionKind::DecodeSplitKv => "decode-split-kv",
        AttentionKind::FlashDecode => "flash-decode",
    }
}

fn parse_gemv_config(value: &str) -> Result<Option<DecodeGemvConfig>, PlanError> {
    match value {
        "none" => Ok(None),
        "baseline" => Ok(Some(DecodeGemvConfig::Baseline)),
        "tuned" => Ok(Some(DecodeGemvConfig::Tuned)),
        _ => Err(PlanError::InvalidGemvConfig),
    }
}

fn parse_blocks(value: &str) -> Result<Vec<FlashDecodeBlock>, PlanError> {
    if value == "default" {
        return Ok(Vec::new());
    }
    value
        .split(',')
        .map(|entry| {
            let (max_length, block) = match entry.split_once(':') {
                Some((length, block)) => (
                    length
                        .trim()
                        .parse()
                        .map_err(|_| PlanError::InvalidFlashDecodeBlocks)?,
                    block,
                ),
                None => (usize::MAX, entry),
            };
            Ok(FlashDecodeBlock {
                max_length,
                block: block
                    .trim()
                    .parse()
                    .map_err(|_| PlanError::InvalidFlashDecodeBlocks)?,
            })
        })
        .collect()
}

fn blocks_name(blocks: &[FlashDecodeBlock]) -> String {
    if blocks.is_empty() {
        return "default".to_owned();
    }
    blocks
        .iter()
        .map(|entry| {
            if entry.max_length == usize::MAX {
                entry.block.to_string()
            } else {
                format!("{}:{}", entry.max_length, entry.block)
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use metal_infer_kernels::{AttentionKind, DecodeGemvConfig, DeviceProfile, KernelSelection};

    use super::{Fusions, Plan, PlanError, PlanInputs, attention_kind_for_tokens};

    const QWEN3_0_6B: PlanInputs = PlanInputs {
        device: DeviceProfile::M4Pro,
        query_heads: 16,
        kv_heads: 8,
        hidden_size: 1024,
    };

    #[test]
    fn device_default_enables_decode_norm_only_on_m4_pro_aligned_widths() {
        let m4_pro = Plan::device_default(QWEN3_0_6B);
        assert!(
            m4_pro.fusions.decode_norm,
            "M4 Pro with aligned width fuses decode norm"
        );
        assert_eq!(m4_pro.attention, AttentionKind::Tiled);
        assert_eq!(
            m4_pro.kernels,
            KernelSelection::for_device(DeviceProfile::M4Pro, 16, 8)
        );
        let generic = Plan::device_default(PlanInputs {
            device: DeviceProfile::Generic,
            ..QWEN3_0_6B
        });
        assert!(
            !generic.fusions.decode_norm,
            "generic devices keep decode norm off"
        );
        assert_eq!(generic.kernels, KernelSelection::default());
        let unaligned = Plan::device_default(PlanInputs {
            hidden_size: 1000,
            ..QWEN3_0_6B
        });
        assert!(
            !unaligned.fusions.decode_norm,
            "unaligned widths keep decode norm off"
        );
    }

    #[test]
    fn resolve_applies_overrides_over_the_device_default() -> Result<(), PlanError> {
        let resolved = Plan::resolve(
            QWEN3_0_6B,
            &[
                "fusion.decode_norm=off".to_owned(),
                "gemv.config=baseline".to_owned(),
            ],
        )?;
        let mut expected = Plan::device_default(QWEN3_0_6B);
        expected.fusions.decode_norm = false;
        expected.kernels.decode_gemv = Some(DecodeGemvConfig::Baseline);
        assert_eq!(
            resolved, expected,
            "overrides must win over device defaults"
        );
        Ok(())
    }

    #[test]
    fn resolve_rejects_invalid_overrides() {
        assert!(
            matches!(
                Plan::resolve(QWEN3_0_6B, &["attention=unknown".to_owned()]),
                Err(PlanError::InvalidAttention)
            ),
            "an invalid override must be rejected"
        );
    }

    #[test]
    fn tiled_attention_selects_flash_decode_for_long_gqa_decode() {
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Tiled, 1, 640, 2),
            AttentionKind::FlashDecode,
            "long single-token decode should select flash decode"
        );
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Tiled, 1, 255, 2),
            AttentionKind::DecodeSplitKv,
            "short single-token decode should select split-KV attention"
        );
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Tiled, 1, 640, 4),
            AttentionKind::DecodeSplitKv,
            "other GQA ratios should select split-KV attention"
        );
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Tiled, 31, 31, 2),
            AttentionKind::Tiled,
            "short prefill should keep tiled attention"
        );
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Tiled, 32, 32, 2),
            AttentionKind::FlashPrefill,
            "32-token prefill should select flash prefill"
        );
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Tiled, 512, 512, 2),
            AttentionKind::FlashPrefill,
            "long prefill should select flash prefill"
        );
        assert_eq!(
            attention_kind_for_tokens(AttentionKind::Reference, 1, 640, 2),
            AttentionKind::Reference,
            "reference attention should remain explicitly selectable"
        );
    }

    #[test]
    fn explicit_attention_kinds_fall_back_by_token_count() {
        let cases = [
            (
                AttentionKind::DecodeSplitKv,
                1,
                640,
                2,
                AttentionKind::DecodeSplitKv,
            ),
            (AttentionKind::DecodeSplitKv, 8, 8, 2, AttentionKind::Tiled),
            (
                AttentionKind::FlashDecode,
                1,
                8,
                2,
                AttentionKind::FlashDecode,
            ),
            (
                AttentionKind::FlashDecode,
                1,
                640,
                4,
                AttentionKind::DecodeSplitKv,
            ),
            (AttentionKind::FlashDecode, 8, 8, 2, AttentionKind::Tiled),
            (
                AttentionKind::FlashPrefill,
                1,
                640,
                2,
                AttentionKind::FlashDecode,
            ),
            (
                AttentionKind::FlashPrefill,
                1,
                8,
                2,
                AttentionKind::DecodeSplitKv,
            ),
            (
                AttentionKind::FlashPrefill,
                8,
                8,
                2,
                AttentionKind::FlashPrefill,
            ),
        ];
        for (configured, tokens, length, ratio, expected) in cases {
            assert_eq!(
                attention_kind_for_tokens(configured, tokens, length, ratio),
                expected,
                "{configured:?} with {tokens} tokens and length {length}"
            );
        }
    }

    fn plan() -> Plan {
        Plan {
            kernels: KernelSelection::default(),
            fusions: Fusions {
                qkv: true,
                gate_up: true,
                add_rms_norm: true,
                qk_rope_cache: true,
                decode_norm: true,
            },
            attention: AttentionKind::Tiled,
        }
    }

    #[test]
    fn every_listed_value_can_be_applied_back() -> Result<(), PlanError> {
        let mut original = plan();
        original.apply_override("flash_decode.blocks=512:64,1024:128")?;
        let mut copy = plan();
        for (key, value) in original.entries() {
            copy.apply_override(&format!("{key}={value}"))?;
        }
        assert_eq!(copy, original, "listed entries must reproduce the plan");
        Ok(())
    }

    #[test]
    fn overrides_change_only_their_key() -> Result<(), PlanError> {
        let mut changed = plan();
        changed.apply_override("fusion.qkv=off")?;
        changed.apply_override("gemv.config=baseline")?;
        let mut expected = plan();
        expected.fusions.qkv = false;
        expected.kernels.decode_gemv = Some(DecodeGemvConfig::Baseline);
        assert_eq!(changed, expected, "an override must only change its key");
        Ok(())
    }

    #[test]
    fn invalid_overrides_are_rejected() {
        let mut target = plan();
        assert!(
            matches!(target.apply_override("fusion.qkv"), Err(PlanError::Syntax)),
            "a missing value must be rejected"
        );
        assert!(
            matches!(
                target.apply_override("fusion.unknown=on"),
                Err(PlanError::UnknownKey)
            ),
            "an unknown key must be rejected"
        );
        assert!(
            matches!(
                target.apply_override("fusion.qkv=maybe"),
                Err(PlanError::InvalidSwitch)
            ),
            "an invalid value must be rejected"
        );
    }
}
