//! Low-level Apple Metal runtime and transformer primitives.

mod context;
mod error;
mod ops;
mod tensor;

pub use context::{
    CommandBatch, DispatchStats, KernelDispatchProfile, MatmulBackend, MetalContext,
};
pub use error::CoreError;
pub use ops::{AttentionConfig, AttentionKind, QkNormRopeCacheConfig};
pub use tensor::{DType, Tensor};
