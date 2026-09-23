use metal_infer_kernels::{AttentionKind, DecodeGemvConfig, FlashDecodeBlock, KernelSelection};
use thiserror::Error;

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

#[derive(Debug, Error)]
pub enum PlanError {
    #[error("plan override `{0}` must have the form key=value")]
    Syntax(String),
    #[error("unknown plan key `{0}`; use `--with list` to show the available keys")]
    UnknownKey(String),
    #[error("invalid value `{value}` for plan key `{key}`: expected {expected}")]
    InvalidValue {
        key: String,
        value: String,
        expected: &'static str,
    },
}

const SWITCH: &str = "on or off";
const ATTENTION: &str = "reference, tiled, flash-prefill, decode-split-kv, or flash-decode";
const GEMV_CONFIG: &str = "none, baseline, or tuned";
const BLOCKS: &str = "default, BLOCKxTHREADS, or LENGTH:BLOCKxTHREADS entries separated by commas";

impl Plan {
    pub fn apply_override(
        &mut self,
        assignment: &str,
    ) -> Result<(), PlanError> {
        let (key, value) = assignment
            .split_once('=')
            .ok_or_else(|| PlanError::Syntax(assignment.to_owned()))?;
        let key = key.trim();
        let value = value.trim();
        let kernels = &mut self.kernels;
        match key {
            "attention" => self.attention = parse_attention(key, value)?,
            "fusion.qkv" => self.fusions.qkv = parse_switch(key, value)?,
            "fusion.gate_up" => self.fusions.gate_up = parse_switch(key, value)?,
            "fusion.add_rms_norm" => self.fusions.add_rms_norm = parse_switch(key, value)?,
            "fusion.qk_rope_cache" => self.fusions.qk_rope_cache = parse_switch(key, value)?,
            "fusion.decode_norm" => self.fusions.decode_norm = parse_switch(key, value)?,
            "gemv.config" => kernels.decode_gemv = parse_gemv_config(key, value)?,
            "flash_decode.blocks" => kernels.flash_decode_blocks = parse_blocks(key, value)?,
            _ => return Err(PlanError::UnknownKey(key.to_owned())),
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

fn invalid(
    key: &str,
    value: &str,
    expected: &'static str,
) -> PlanError {
    PlanError::InvalidValue {
        key: key.to_owned(),
        value: value.to_owned(),
        expected,
    }
}

fn parse_switch(
    key: &str,
    value: &str,
) -> Result<bool, PlanError> {
    match value {
        "on" | "true" => Ok(true),
        "off" | "false" => Ok(false),
        _ => Err(invalid(key, value, SWITCH)),
    }
}

fn switch_name(enabled: bool) -> String {
    String::from(if enabled { "on" } else { "off" })
}

fn parse_attention(
    key: &str,
    value: &str,
) -> Result<AttentionKind, PlanError> {
    match value {
        "reference" => Ok(AttentionKind::Reference),
        "tiled" => Ok(AttentionKind::Tiled),
        "flash-prefill" => Ok(AttentionKind::FlashPrefill),
        "decode-split-kv" => Ok(AttentionKind::DecodeSplitKv),
        "flash-decode" => Ok(AttentionKind::FlashDecode),
        _ => Err(invalid(key, value, ATTENTION)),
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

fn parse_gemv_config(
    key: &str,
    value: &str,
) -> Result<Option<DecodeGemvConfig>, PlanError> {
    match value {
        "none" => Ok(None),
        "baseline" => Ok(Some(DecodeGemvConfig::Baseline)),
        "tuned" => Ok(Some(DecodeGemvConfig::Tuned)),
        _ => Err(invalid(key, value, GEMV_CONFIG)),
    }
}

fn parse_blocks(
    key: &str,
    value: &str,
) -> Result<Vec<FlashDecodeBlock>, PlanError> {
    if value == "default" {
        return Ok(Vec::new());
    }
    value
        .split(',')
        .map(|entry| {
            let (max_length, shape) = match entry.split_once(':') {
                Some((length, shape)) => (
                    length
                        .trim()
                        .parse()
                        .map_err(|_| invalid(key, value, BLOCKS))?,
                    shape,
                ),
                None => (usize::MAX, entry),
            };
            let (block, threads) = shape
                .trim()
                .split_once('x')
                .ok_or_else(|| invalid(key, value, BLOCKS))?;
            Ok(FlashDecodeBlock {
                max_length,
                block: block.parse().map_err(|_| invalid(key, value, BLOCKS))?,
                threads: threads.parse().map_err(|_| invalid(key, value, BLOCKS))?,
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
                format!("{}x{}", entry.block, entry.threads)
            } else {
                format!("{}:{}x{}", entry.max_length, entry.block, entry.threads)
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use metal_infer_kernels::{AttentionKind, DecodeGemvConfig, KernelSelection};

    use super::{Fusions, Plan, PlanError};

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
        original.apply_override("flash_decode.blocks=512:64x128,1024:128x256")?;
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
            matches!(
                target.apply_override("fusion.qkv"),
                Err(PlanError::Syntax(_))
            ),
            "a missing value must be rejected"
        );
        assert!(
            matches!(
                target.apply_override("fusion.unknown=on"),
                Err(PlanError::UnknownKey(_))
            ),
            "an unknown key must be rejected"
        );
        assert!(
            matches!(
                target.apply_override("fusion.qkv=maybe"),
                Err(PlanError::InvalidValue { .. })
            ),
            "an invalid value must be rejected"
        );
    }
}
