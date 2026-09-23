use std::rc::Rc;

use metal_infer_runtime::{
    CommandBatch, CoreError, DType, DispatchStats, Library, MetalContext, PendingBatch, Tensor,
};
use objc2_metal::MTLSize;

use crate::tuning::Tuning;

const PRELUDE: &str = include_str!("../metal/prelude.metal");
const PRELUDE_INCLUDE: &str = "#include \"prelude.metal\"\n";
const FAMILIES: [&str; 9] = [
    include_str!("../metal/elementwise.metal"),
    include_str!("../metal/gemm.metal"),
    include_str!("../metal/gemv.metal"),
    include_str!("../metal/fused.metal"),
    include_str!("../metal/norm.metal"),
    include_str!("../metal/sampling.metal"),
    include_str!("../metal/rope.metal"),
    include_str!("../metal/attention.metal"),
    include_str!("../metal/kv.metal"),
];

fn shader_source() -> String {
    let mut source = PRELUDE.to_owned();
    for family in FAMILIES {
        source.push_str(family.strip_prefix(PRELUDE_INCLUDE).unwrap_or(family));
    }
    source
}

#[derive(Clone)]
pub struct Kernels {
    context: MetalContext,
    library: Rc<Library>,
    pub(crate) tuning: Tuning,
}

impl Kernels {
    pub fn new(context: &MetalContext) -> Result<Self, CoreError> {
        let library = Library::new(context, &shader_source())?;
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

#[cfg(test)]
mod tests {
    use super::shader_source;

    #[test]
    fn shader_source_is_self_contained() {
        let source = shader_source();
        assert!(
            source.starts_with("#include <metal_simdgroup_matrix>\n"),
            "the prelude must open the shader source"
        );
        assert!(
            !source.contains("#include \""),
            "local includes must be resolved before compilation"
        );
        assert_eq!(
            source.matches("\nkernel void").count(),
            33,
            "every kernel must be part of the shader source"
        );
    }
}
