//! Low-level Apple Metal runtime and transformer primitives.

mod context;
mod error;
mod gemv_dispatch;
mod ops;
mod tensor;

pub use context::{
    CommandBatch, DispatchStats, KernelDispatchProfile, MatmulBackend, MetalContext, PendingBatch,
};
pub use error::CoreError;
pub use gemv_dispatch::DecodeGemvConfig;
pub use ops::{AttentionConfig, AttentionKind, QkNormRopeCacheConfig};
pub use tensor::{DType, Tensor};
