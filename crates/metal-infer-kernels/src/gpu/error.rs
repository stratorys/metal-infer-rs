use objc2::rc::Retained;
use objc2_foundation::NSError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("no Metal device is available")]
    NoDevice,
    #[error("cannot create the Metal command queue")]
    CommandQueueCreation,
    #[error("cannot create a Metal command buffer")]
    CommandBufferCreation,
    #[error("cannot create a Metal compute encoder")]
    ComputeEncoderCreation,
    #[error("cannot create a profiled Metal compute encoder")]
    ProfiledEncoderCreation,
    #[error("the Metal compute encoder is already finished")]
    EncoderFinished,
    #[error("cannot create a Metal buffer")]
    BufferCreation,
    #[error("cannot create a Metal scratch buffer")]
    ScratchBufferCreation,
    #[error("Metal shader compilation failed")]
    ShaderCompilation(#[source] Retained<NSError>),
    #[error("Metal kernel is missing from the shader library")]
    MissingKernel,
    #[error("Metal pipeline creation failed")]
    PipelineCreation(#[source] Retained<NSError>),
    #[error("Metal command failed")]
    Command(#[source] Retained<NSError>),
    #[error("Metal command failed without an error")]
    CommandWithoutError,
    #[error("tensor must be f16")]
    ExpectedF16,
    #[error("tensor must be u32")]
    ExpectedU32,
    #[error("host data length does not match the tensor")]
    DataLengthMismatch,
    #[error("tensor rank and dimensions must be non-zero")]
    EmptyShape,
    #[error("tensor element count overflows")]
    ElementCountOverflow,
    #[error("tensor byte length overflows")]
    ByteLengthOverflow,
    #[error("scratch allocation size overflows")]
    ScratchSizeOverflow,
    #[error("reshape element count overflows")]
    ReshapeOverflow,
    #[error("reshape changes the element count")]
    ReshapeMismatch,
    #[error("prefix requires a tensor of rank one or more")]
    PrefixRank,
    #[error("prefix length is outside the tensor capacity")]
    PrefixOutOfRange,
    #[error("row requires a matrix")]
    RowRank,
    #[error("row index is outside the matrix")]
    RowOutOfRange,
    #[error("slice requires a rank-one tensor")]
    SliceRank,
    #[error("slice range overflows")]
    SliceOverflow,
    #[error("slice range is outside the tensor")]
    SliceOutOfRange,
    #[error("too many dispatches in one batch for the timestamp buffer")]
    TooManyProfiledDispatches,
    #[error("GPU does not support counters at compute pass boundaries")]
    StageBoundaryCountersUnsupported,
    #[error("GPU exposes no counter sets")]
    NoCounterSets,
    #[error("GPU exposes no timestamp counter set")]
    NoTimestampCounterSet,
    #[error("cannot calibrate GPU timestamps")]
    TimestampCalibration,
    #[error("cannot create the GPU timestamp buffer")]
    TimestampBufferCreation(#[source] Retained<NSError>),
    #[error("batch has no GPU timestamp buffer")]
    MissingTimestampBuffer,
    #[error("GPU profiling was disabled before the batch completed")]
    ProfilingDisabled,
    #[error("cannot resolve GPU timestamps")]
    TimestampResolution,
    #[error("GPU timestamp buffer has an unexpected size")]
    TimestampBufferSize,
    #[error("GPU timestamp is missing")]
    MissingTimestamp,
    #[error("GPU timestamps of a kernel are invalid")]
    InvalidTimestamps,
}
