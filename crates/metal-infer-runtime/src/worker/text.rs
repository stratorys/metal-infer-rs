const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Channels<'text> {
    pub reasoning: &'text str,
    pub content: &'text str,
}

pub fn truncate_stop<'text>(
    value: &'text str,
    stop_sequences: &[String],
) -> &'text str {
    stop_sequences
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| value.find(stop.as_str()))
        .min()
        .and_then(|index| value.get(..index))
        .unwrap_or(value)
}

pub fn safe_stream_boundary(
    value: &str,
    stop_sequences: &[impl AsRef<str>],
) -> usize {
    let withheld = stop_sequences
        .iter()
        .map(AsRef::as_ref)
        .filter(|stop| !stop.is_empty())
        .flat_map(|stop| {
            stop.char_indices()
                .skip(1)
                .filter_map(|(index, _)| stop.get(..index))
        })
        .filter(|prefix| value.ends_with(prefix))
        .map(str::len)
        .max()
        .unwrap_or(0);
    value.len().saturating_sub(withheld)
}

pub fn stream_channels(value: &str) -> Channels<'_> {
    if value.len() < THINK_OPEN.len() && THINK_OPEN.starts_with(value) {
        return Channels {
            reasoning: "",
            content: "",
        };
    }
    let Some(rest) = value.strip_prefix(THINK_OPEN) else {
        return Channels {
            reasoning: "",
            content: value,
        };
    };
    if let Some((reasoning, content)) = rest.split_once(THINK_CLOSE) {
        return Channels { reasoning, content };
    }
    let safe_end = safe_stream_boundary(rest, &[THINK_CLOSE]);
    Channels {
        reasoning: rest.get(..safe_end).unwrap_or_default(),
        content: "",
    }
}

pub fn final_channels(value: &str) -> Channels<'_> {
    let Some(rest) = value.strip_prefix(THINK_OPEN) else {
        return Channels {
            reasoning: "",
            content: value,
        };
    };
    rest.split_once(THINK_CLOSE).map_or(
        Channels {
            reasoning: rest,
            content: "",
        },
        |(reasoning, content)| Channels { reasoning, content },
    )
}

#[cfg(test)]
mod tests {
    use super::{Channels, final_channels, safe_stream_boundary, stream_channels, truncate_stop};

    fn stops(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn truncate_stop_cuts_at_the_earliest_stop() {
        assert_eq!(
            truncate_stop("one two three", &stops(&["three", "two"])),
            "one ",
            "the earliest stop sequence must win"
        );
    }

    #[test]
    fn truncate_stop_ignores_empty_stops() {
        assert_eq!(
            truncate_stop("hello", &stops(&[""])),
            "hello",
            "an empty stop sequence must not truncate"
        );
    }

    #[test]
    fn safe_stream_boundary_withholds_a_stop_prefix() {
        assert_eq!(
            safe_stream_boundary("hello\n\nUs", &stops(&["\n\nUser:"])),
            5,
            "a trailing stop prefix must be withheld"
        );
    }

    #[test]
    fn safe_stream_boundary_keeps_char_boundaries() {
        assert_eq!(
            safe_stream_boundary("café", &stops(&["éclair"])),
            3,
            "a multi-byte stop prefix must be withheld whole"
        );
    }

    #[test]
    fn safe_stream_boundary_keeps_text_without_stop_prefix() {
        assert_eq!(
            safe_stream_boundary("hello", &stops(&["world"])),
            5,
            "text without a stop prefix must be released"
        );
    }

    #[test]
    fn stream_channels_withholds_a_partial_open_tag() {
        assert_eq!(
            stream_channels("<thi"),
            Channels {
                reasoning: "",
                content: ""
            },
            "a partial <think> must not be released"
        );
    }

    #[test]
    fn stream_channels_withholds_a_partial_close_tag() {
        assert_eq!(
            stream_channels("<think>plan</thi"),
            Channels {
                reasoning: "plan",
                content: ""
            },
            "a partial </think> must not be released"
        );
    }

    #[test]
    fn stream_channels_splits_a_closed_block() {
        assert_eq!(
            stream_channels("<think>plan</think>answer"),
            Channels {
                reasoning: "plan",
                content: "answer"
            },
            "a closed block must split reasoning and content"
        );
    }

    #[test]
    fn stream_channels_passes_plain_text() {
        assert_eq!(
            stream_channels("answer"),
            Channels {
                reasoning: "",
                content: "answer"
            },
            "text without <think> is content"
        );
    }

    #[test]
    fn final_channels_releases_partial_tags() {
        assert_eq!(
            final_channels("<thi"),
            Channels {
                reasoning: "",
                content: "<thi"
            },
            "a partial <think> at the end is content"
        );
        assert_eq!(
            final_channels("<think>plan</thi"),
            Channels {
                reasoning: "plan</thi",
                content: ""
            },
            "an unclosed block at the end is reasoning"
        );
    }
}
