const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Channels<'text> {
    pub reasoning: &'text str,
    pub content: &'text str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThinkState {
    Undecided,
    Plain,
    Reasoning { close_search_from: usize },
    Closed { reasoning_end: usize },
}

pub fn truncate_stop<'text>(
    value: &'text str,
    stop_sequences: &[String],
) -> &'text str {
    earliest_stop(value, stop_sequences)
        .and_then(|index| value.get(..index))
        .unwrap_or(value)
}

pub fn find_stop_in_tail(
    value: &str,
    delta_bytes: usize,
    stop_sequences: &[String],
) -> Option<usize> {
    let stop_bytes_max = stop_sequences.iter().map(String::len).max().unwrap_or(0);
    let tail_bytes = delta_bytes.saturating_add(stop_bytes_max.saturating_sub(1));
    let window_start = floor_char_boundary(value, value.len().saturating_sub(tail_bytes));
    value
        .get(window_start..)
        .and_then(|window| earliest_stop(window, stop_sequences))
        .map(|index| window_start.saturating_add(index))
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

pub fn advance_think(
    state: ThinkState,
    value: &str,
) -> ThinkState {
    match state {
        ThinkState::Undecided => decide_think(value),
        ThinkState::Reasoning { close_search_from } => find_think_close(value, close_search_from),
        ThinkState::Plain | ThinkState::Closed { .. } => state,
    }
}

pub fn stream_channels(
    value: &str,
    state: ThinkState,
) -> Channels<'_> {
    match state {
        ThinkState::Undecided => Channels {
            reasoning: "",
            content: "",
        },
        ThinkState::Plain => Channels {
            reasoning: "",
            content: value,
        },
        ThinkState::Reasoning { .. } => {
            let rest = value.get(THINK_OPEN.len()..).unwrap_or_default();
            let safe_end = safe_stream_boundary(rest, &[THINK_CLOSE]);
            Channels {
                reasoning: rest.get(..safe_end).unwrap_or_default(),
                content: "",
            }
        }
        ThinkState::Closed { reasoning_end } => Channels {
            reasoning: value
                .get(THINK_OPEN.len()..reasoning_end)
                .unwrap_or_default(),
            content: value
                .get(reasoning_end.saturating_add(THINK_CLOSE.len())..)
                .unwrap_or_default(),
        },
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

fn earliest_stop(
    value: &str,
    stop_sequences: &[String],
) -> Option<usize> {
    stop_sequences
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| value.find(stop.as_str()))
        .min()
}

fn decide_think(value: &str) -> ThinkState {
    if value.len() < THINK_OPEN.len() && THINK_OPEN.starts_with(value) {
        ThinkState::Undecided
    } else if value.starts_with(THINK_OPEN) {
        find_think_close(value, THINK_OPEN.len())
    } else {
        ThinkState::Plain
    }
}

fn find_think_close(
    value: &str,
    search_from: usize,
) -> ThinkState {
    value
        .get(search_from..)
        .and_then(|tail| tail.find(THINK_CLOSE))
        .map_or_else(
            || ThinkState::Reasoning {
                close_search_from: floor_char_boundary(
                    value,
                    value.len().saturating_sub(THINK_CLOSE.len() - 1),
                )
                .max(THINK_OPEN.len()),
            },
            |index| ThinkState::Closed {
                reasoning_end: search_from.saturating_add(index),
            },
        )
}

fn floor_char_boundary(
    value: &str,
    index: usize,
) -> usize {
    (0..=index.min(value.len()))
        .rev()
        .find(|candidate| value.is_char_boundary(*candidate))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::{
        Channels, ThinkState, advance_think, final_channels, find_stop_in_tail,
        safe_stream_boundary, stream_channels, truncate_stop,
    };

    fn stops(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    fn streamed(value: &str) -> Channels<'_> {
        stream_channels(value, advance_think(ThinkState::Undecided, value))
    }

    fn think_steps(steps: &[&str]) -> ThinkState {
        steps.iter().fold(ThinkState::Undecided, |state, value| {
            advance_think(state, value)
        })
    }

    #[test]
    fn find_stop_in_tail_finds_a_stop_across_deltas() {
        assert_eq!(
            find_stop_in_tail("count 19, 20", 1, &stops(&["20"])),
            Some(10),
            "a stop straddling the previous text and the delta must be found"
        );
    }

    #[test]
    fn find_stop_in_tail_ignores_text_before_the_window() {
        assert_eq!(
            find_stop_in_tail("20 was checked, now 21", 2, &stops(&["20"])),
            None,
            "a stop before the window was already checked on an earlier step"
        );
    }

    #[test]
    fn find_stop_in_tail_picks_the_earliest_stop() {
        assert_eq!(
            find_stop_in_tail("prefix one two", 8, &stops(&["two", "one"])),
            Some(7),
            "the earliest stop in the window must win"
        );
    }

    #[test]
    fn find_stop_in_tail_handles_multi_byte_stops() {
        assert_eq!(
            find_stop_in_tail("un éclair", 3, &stops(&["éclair"])),
            Some(3),
            "a multi-byte stop must be found at its byte offset"
        );
    }

    #[test]
    fn find_stop_in_tail_starts_on_a_char_boundary() {
        assert_eq!(
            find_stop_in_tail("日本!", 1, &stops(&["本!", "!!!!!"])),
            Some(3),
            "a window starting inside a character must move back to its boundary"
        );
    }

    #[test]
    fn find_stop_in_tail_ignores_empty_stops() {
        assert_eq!(
            find_stop_in_tail("hello", 5, &stops(&[""])),
            None,
            "an empty stop sequence must not match"
        );
    }

    #[test]
    fn advance_think_finds_a_close_tag_split_across_steps() {
        let state = think_steps(&["<think>plan</thi", "<think>plan</think>ok"]);
        assert_eq!(
            state,
            ThinkState::Closed { reasoning_end: 11 },
            "a </think> split across two steps must be found"
        );
        assert_eq!(
            stream_channels("<think>plan</think>ok", state),
            Channels {
                reasoning: "plan",
                content: "ok"
            },
            "the closed block must split reasoning and content"
        );
    }

    #[test]
    fn advance_think_waits_for_a_split_open_tag() {
        assert_eq!(
            think_steps(&["<th"]),
            ThinkState::Undecided,
            "a partial <think> must keep the state undecided"
        );
        assert_eq!(
            think_steps(&["<th", "<think>"]),
            ThinkState::Reasoning {
                close_search_from: 7
            },
            "a completed <think> must start the reasoning"
        );
    }

    #[test]
    fn advance_think_keeps_an_unclosed_block_in_reasoning() {
        let value = "<think>still thinking";
        let state = think_steps(&["<think>still", value]);
        assert_eq!(
            state,
            ThinkState::Reasoning {
                close_search_from: 14
            },
            "an unclosed block must stay in reasoning"
        );
        assert_eq!(
            stream_channels(value, state),
            Channels {
                reasoning: "still thinking",
                content: ""
            },
            "an unclosed block must stream its reasoning"
        );
    }

    #[test]
    fn advance_think_passes_plain_text() {
        assert_eq!(
            think_steps(&["ans", "answer <think>"]),
            ThinkState::Plain,
            "text that does not start with <think> is plain"
        );
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
            streamed("<thi"),
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
            streamed("<think>plan</thi"),
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
            streamed("<think>plan</think>answer"),
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
            streamed("answer"),
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
