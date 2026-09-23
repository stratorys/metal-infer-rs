use thiserror::Error;

use crate::gpu::GpuError;

#[derive(Debug, Error)]
pub enum KernelError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
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
