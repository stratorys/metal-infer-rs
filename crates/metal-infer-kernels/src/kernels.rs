use std::rc::Rc;

use metal_infer_runtime::{
    CommandBatch, CoreError, DType, DispatchStats, Library, MetalContext, PendingBatch, Tensor,
};
use objc2_metal::MTLSize;

use crate::tuning::Tuning;

const SHADERS: &str = concat!(
    include_str!("../metal/prelude.metal"),
    include_str!("../metal/elementwise.metal"),
    include_str!("../metal/gemm.metal"),
    include_str!("../metal/gemv.metal"),
    include_str!("../metal/fused.metal"),
    include_str!("../metal/norm.metal"),
    include_str!("../metal/sampling.metal"),
    include_str!("../metal/rope.metal"),
    include_str!("../metal/attention.metal"),
    include_str!("../metal/kv.metal"),
);

#[derive(Clone)]
pub struct Kernels {
    context: MetalContext,
    library: Rc<Library>,
    pub(crate) tuning: Tuning,
}

impl Kernels {
    pub fn new(context: &MetalContext) -> Result<Self, CoreError> {
        let library = Library::new(context, SHADERS)?;
        Ok(Self {
            context: context.clone(),
            library: Rc::new(library),
            tuning: Tuning::new(&context.device_name()),
        })
    }

    pub const fn context(&self) -> &MetalContext {
        &self.context
    }

    pub fn begin_batch(&self) -> Result<KernelBatch<'_>, CoreError> {
        Ok(KernelBatch {
            batch: self.context.begin_batch()?,
            kernels: self,
        })
    }
}

pub struct KernelBatch<'kernels> {
    pub(crate) batch: CommandBatch<'kernels>,
    pub(crate) kernels: &'kernels Kernels,
}

impl<'kernels> KernelBatch<'kernels> {
    pub fn commit(self) -> Result<PendingBatch<'kernels>, CoreError> {
        self.batch.commit()
    }

    pub fn finish(self) -> Result<DispatchStats, CoreError> {
        self.batch.finish()
    }

    pub(crate) fn empty(
        &self,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor, CoreError> {
        self.batch.empty(shape, dtype)
    }

    pub(crate) fn dispatch<T>(
        &mut self,
        kernel: &str,
        tensors: &[&Tensor],
        params: &T,
        grid: MTLSize,
        threadgroup: MTLSize,
    ) -> Result<(), CoreError> {
        let pipeline = self.kernels.library.pipeline(kernel)?;
        self.batch
            .dispatch(&pipeline, kernel, tensors, params, grid, threadgroup)
    }
}
