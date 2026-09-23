use objc2::rc::Retained;
use objc2_foundation::NSError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
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
    #[error("tensor shapes differ")]
    ShapeMismatch,
    #[error("tensor must be a matrix")]
    ExpectedMatrix,
    #[error("value does not fit in u32")]
    U32Overflow,
    #[error("dispatch size overflows")]
    DispatchOverflow,
    #[error("matmul inner dimensions differ")]
    MatmulInnerDimension,
    #[error("matmul2 inner dimensions differ")]
    Matmul2InnerDimension,
    #[error("matmul3 inner dimensions differ")]
    Matmul3InnerDimension,
    #[error("rms_norm_matmul3 requires one row and matching widths divisible by 256")]
    RmsNormMatmul3Shape,
    #[error("add_rms_norm_matmul2 requires one row and matching widths divisible by 256")]
    AddRmsNormMatmul2Shape,
    #[error("rms_norm requires a tensor of rank one or more")]
    RmsNormRank,
    #[error("rms_norm weight width differs from the input width")]
    RmsNormWeight,
    #[error("add_rms_norm requires a tensor of rank one or more")]
    AddRmsNormRank,
    #[error("add_rms_norm weight width differs from the input width")]
    AddRmsNormWeight,
    #[error("RoPE expects [tokens, heads, head_dim]")]
    RopeRank,
    #[error("RoPE head_dim must be even")]
    RopeOddHeadDim,
    #[error("query must have rank 3")]
    QueryRank,
    #[error("key must have rank 3")]
    KeyRank,
    #[error("value must have rank 3")]
    ValueRank,
    #[error("key cache must have rank 3")]
    KeyCacheRank,
    #[error("Q/K transform tensor shapes are incompatible")]
    QkTransformShape,
    #[error("Q/K transform has an invalid head_dim or cache offset")]
    QkTransformOffset,
    #[error("KV source must have rank 3")]
    KvSourceRank,
    #[error("KV cache must have rank 3")]
    KvCacheRank,
    #[error("KV source and cache shapes are incompatible")]
    KvShape,
    #[error("KV cache capacity exceeded")]
    KvCapacity,
    #[error("attention tensor shapes do not match the configuration")]
    AttentionShape,
    #[error("query heads must be divisible by KV heads")]
    AttentionHeadRatio,
    #[error("tiled attention requires a power-of-two head_dim of at most 256")]
    TiledAttentionHeadDim,
    #[error("flash decode block size must be 32, 64, 128, or 256")]
    FlashDecodeBlockSize,
    #[error("decode attention requires exactly one query token")]
    DecodeAttentionTokens,
    #[error("flash decode requires two query heads per KV head")]
    FlashDecodeHeadRatio,
    #[error("flash decode requires at least one available key")]
    FlashDecodeEmpty,
    #[error("flash decode partial width overflows")]
    FlashDecodePartialOverflow,
    #[error("copy_row source must be a matrix")]
    CopyRowSourceRank,
    #[error("copy_row destination must be a matrix")]
    CopyRowDestinationRank,
    #[error("copy_row shapes are incompatible")]
    CopyRowShape,
    #[error("tokens must be a rank-one u32 tensor")]
    TokensShape,
    #[error("argmax logits must be a nonempty vector")]
    ArgmaxLogitsShape,
    #[error("argmax output must be one u32")]
    ArgmaxOutputShape,
}
