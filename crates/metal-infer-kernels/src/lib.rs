mod error;
mod gpu;
mod kernels;
mod ops;
mod tuning;

pub use error::KernelError;
pub use gpu::{
    DType, DispatchStats, GpuError, KernelDispatchProfile, MetalContext, PendingBatch, Tensor,
};
pub use kernels::{KernelBatch, Kernels};
pub use ops::{AttentionConfig, AttentionKind, QkNormRopeCacheConfig};
pub use tuning::{DecodeGemvConfig, DeviceProfile, FlashDecodeBlock, KernelSelection};
