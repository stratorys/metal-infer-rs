use std::cell::RefCell;
use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary};

use crate::gpu::context::device;
use crate::gpu::{GpuError, MetalContext};

pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

pub struct Library {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: RefCell<HashMap<String, Pipeline>>,
}

impl Library {
    pub fn new(
        context: &MetalContext,
        source: &str,
    ) -> Result<Self, GpuError> {
        let metal_device = device(context);
        let source = NSString::from_str(source);
        let library = metal_device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(GpuError::ShaderCompilation)?;
        Ok(Self {
            device: metal_device.clone(),
            library,
            pipelines: RefCell::new(HashMap::new()),
        })
    }

    pub fn pipeline(
        &self,
        name: &str,
    ) -> Result<Pipeline, GpuError> {
        if let Some(pipeline) = self.pipelines.borrow().get(name) {
            return Ok(pipeline.clone());
        }
        let function_name = NSString::from_str(name);
        let function: Retained<ProtocolObject<dyn MTLFunction>> = self
            .library
            .newFunctionWithName(&function_name)
            .ok_or(GpuError::MissingKernel)?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(GpuError::PipelineCreation)?;
        self.pipelines
            .borrow_mut()
            .insert(name.to_owned(), pipeline.clone());
        Ok(pipeline)
    }
}
