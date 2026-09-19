//! Low-level Apple Metal runtime and transformer primitives.

mod context;
mod error;
mod ops;
mod tensor;

pub use context::{CommandBatch, DispatchStats, MetalContext};
pub use error::CoreError;
pub use ops::{AttentionConfig, AttentionKind};
pub use tensor::{DType, Tensor};
