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
