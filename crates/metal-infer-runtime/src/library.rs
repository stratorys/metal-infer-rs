use std::cell::RefCell;
use std::collections::HashMap;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLDevice, MTLFunction, MTLLibrary};

use crate::{CoreError, MetalContext, Pipeline};

pub struct Library {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: RefCell<HashMap<String, Pipeline>>,
}

impl Library {
    pub fn new(
        context: &MetalContext,
        source: &str,
    ) -> Result<Self, CoreError> {
        let source = NSString::from_str(source);
        let library = context
            .device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(CoreError::Shader)?;
        Ok(Self {
            device: context.device.clone(),
            library,
            pipelines: RefCell::new(HashMap::new()),
        })
    }

    pub fn pipeline(
        &self,
        name: &str,
    ) -> Result<Pipeline, CoreError> {
        if let Some(pipeline) = self.pipelines.borrow().get(name) {
            return Ok(pipeline.clone());
        }
        let function_name = NSString::from_str(name);
        let function: Retained<ProtocolObject<dyn MTLFunction>> = self
            .library
            .newFunctionWithName(&function_name)
            .ok_or_else(|| CoreError::MissingKernel(name.to_owned()))?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(CoreError::Pipeline)?;
        self.pipelines
            .borrow_mut()
            .insert(name.to_owned(), pipeline.clone());
        Ok(pipeline)
    }
}
