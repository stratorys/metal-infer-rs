use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Duration;
use std::time::Instant;

use half::f16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLFunction, MTLLibrary, MTLResourceOptions,
};

use crate::{CoreError, DType, Tensor};

const SHADERS: &str = include_str!("kernels/transformer.metal");

pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

#[derive(Clone, Copy, Debug, Default)]
pub struct DispatchStats {
    pub gpu_time: Duration,
    pub wall_time: Duration,
}

pub struct CommandBatch<'context> {
    pub(crate) context: &'context MetalContext,
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
}

#[derive(Clone)]
pub struct MetalContext {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: RefCell<HashMap<String, Pipeline>>,
}

impl MetalContext {
    pub fn new() -> Result<Self, CoreError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(CoreError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or(CoreError::Resource("command queue"))?;
        let source = NSString::from_str(SHADERS);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(CoreError::Shader)?;
        Ok(Self {
            device,
            queue,
            library,
            pipelines: RefCell::new(HashMap::new()),
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub fn allocated_bytes(&self) -> usize {
        self.device.currentAllocatedSize()
    }

    pub fn begin_batch(&self) -> Result<CommandBatch<'_>, CoreError> {
        let command_buffer = self.command_buffer()?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(CoreError::Resource("compute encoder"))?;
        Ok(CommandBatch {
            context: self,
            command_buffer,
            encoder: Some(encoder),
        })
    }

    pub fn empty(
        &self,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor, CoreError> {
        let elements = checked_elements(shape)?;
        let byte_len = elements
            .checked_mul(dtype.size())
            .ok_or_else(|| CoreError::Shape("tensor byte length overflow".into()))?;
        let buffer = self
            .device
            .newBufferWithLength_options(byte_len.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or(CoreError::Resource("buffer"))?;
        Ok(Tensor::new(buffer, shape.to_vec(), dtype))
    }

    pub fn tensor_f16(
        &self,
        values: &[f32],
        shape: &[usize],
    ) -> Result<Tensor, CoreError> {
        let tensor = self.empty(shape, DType::F16)?;
        check_data_len(tensor.len(), values.len())?;
        let destination = tensor.buffer.contents().as_ptr().cast::<u16>();
        for (index, value) in values.iter().enumerate() {
            // SAFETY: destination has tensor.len() u16 slots and index is
            // bounded by values.
            unsafe {
                destination
                    .add(index)
                    .write(f16::from_f32(*value).to_bits())
            };
        }
        Ok(tensor)
    }

    pub fn tensor_f16_bits(
        &self,
        values: &[u16],
        shape: &[usize],
    ) -> Result<Tensor, CoreError> {
        let tensor = self.empty(shape, DType::F16)?;
        check_data_len(tensor.len(), values.len())?;
        // SAFETY: source and destination are valid, non-overlapping,
        // equal-length regions.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                tensor.buffer.contents().as_ptr().cast::<u16>(),
                values.len(),
            )
        };
        Ok(tensor)
    }

    pub fn tensor_f16_bytes(
        &self,
        bytes: &[u8],
        shape: &[usize],
    ) -> Result<Tensor, CoreError> {
        let tensor = self.empty(shape, DType::F16)?;
        if tensor.byte_len() != bytes.len() {
            return Err(CoreError::DataLength {
                expected: tensor.byte_len(),
                actual: bytes.len(),
            });
        }
        // SAFETY: both regions are valid for bytes.len() bytes and do not
        // overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                tensor.buffer.contents().as_ptr().cast::<u8>(),
                bytes.len(),
            )
        };
        Ok(tensor)
    }

    pub fn tensor_u32(
        &self,
        values: &[u32],
        shape: &[usize],
    ) -> Result<Tensor, CoreError> {
        let tensor = self.empty(shape, DType::U32)?;
        check_data_len(tensor.len(), values.len())?;
        // SAFETY: source and destination are valid, non-overlapping,
        // equal-length regions.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                tensor.buffer.contents().as_ptr().cast::<u32>(),
                values.len(),
            )
        };
        Ok(tensor)
    }

    pub(crate) fn pipeline(
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

    pub(crate) fn command_buffer(
        &self
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, CoreError> {
        self.queue
            .commandBuffer()
            .ok_or(CoreError::Resource("command buffer"))
    }

    pub(crate) unsafe fn bytes<T>(value: &T) -> (NonNull<c_void>, usize) {
        let pointer = NonNull::from(value).cast();
        (pointer, std::mem::size_of::<T>())
    }
}

impl CommandBatch<'_> {
    pub(crate) fn encoder(
        &self
    ) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>, CoreError> {
        self.encoder
            .as_deref()
            .ok_or(CoreError::Resource("finished compute encoder"))
    }

    pub fn finish(mut self) -> Result<DispatchStats, CoreError> {
        let encoder = self
            .encoder
            .take()
            .ok_or(CoreError::Resource("finished compute encoder"))?;
        encoder.endEncoding();
        let started = Instant::now();
        self.command_buffer.commit();
        self.command_buffer.waitUntilCompleted();
        let wall_time = started.elapsed();
        if self.command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(self
                .command_buffer
                .error()
                .map_or(CoreError::UnknownCommand, CoreError::Command));
        }
        let gpu_seconds =
            (self.command_buffer.GPUEndTime() - self.command_buffer.GPUStartTime()).max(0.0);
        Ok(DispatchStats {
            gpu_time: Duration::from_secs_f64(gpu_seconds),
            wall_time,
        })
    }
}

impl Drop for CommandBatch<'_> {
    fn drop(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
    }
}

fn checked_elements(shape: &[usize]) -> Result<usize, CoreError> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(CoreError::Shape(
            "rank and dimensions must be non-zero".into(),
        ));
    }
    shape.iter().try_fold(1usize, |length, dimension| {
        length
            .checked_mul(*dimension)
            .ok_or_else(|| CoreError::Shape("element count overflow".into()))
    })
}

fn check_data_len(
    expected: usize,
    actual: usize,
) -> Result<(), CoreError> {
    if expected == actual {
        Ok(())
    } else {
        Err(CoreError::DataLength { expected, actual })
    }
}
