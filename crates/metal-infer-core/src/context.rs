use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ops::Range;
use std::ptr::NonNull;
use std::rc::{Rc, Weak};
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
    scratch: Rc<RefCell<ScratchState>>,
}

const SCRATCH_ALIGNMENT: usize = 256;
const SCRATCH_CHUNK_BYTES: usize = 1024 * 1024;

struct ScratchChunk {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    free: Vec<Range<usize>>,
}

struct ScratchState {
    chunks: Vec<ScratchChunk>,
}

pub(crate) struct ScratchLease {
    state: Weak<RefCell<ScratchState>>,
    chunk: usize,
    range: Range<usize>,
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
            scratch: Rc::new(RefCell::new(ScratchState { chunks: Vec::new() })),
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
    pub(crate) fn empty(
        &self,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor, CoreError> {
        let elements = checked_elements(shape)?;
        let byte_len = elements
            .checked_mul(dtype.size())
            .ok_or_else(|| CoreError::Shape("tensor byte length overflow".into()))?;
        let allocation_len = align_up(byte_len.max(1), SCRATCH_ALIGNMENT)?;
        let mut state = self.scratch.borrow_mut();
        let mut selected = None;
        for (chunk_index, chunk) in state.chunks.iter_mut().enumerate() {
            if let Some(range_index) = chunk
                .free
                .iter()
                .position(|range| range.end - range.start >= allocation_len)
            {
                let range = chunk.free.remove(range_index);
                let allocation = range.start..range.start + allocation_len;
                if allocation.end < range.end {
                    chunk.free.push(allocation.end..range.end);
                }
                selected = Some((chunk_index, allocation, chunk.buffer.clone()));
                break;
            }
        }
        let (chunk_index, range, buffer) = if let Some(allocation) = selected {
            allocation
        } else {
            let chunk_len = allocation_len.max(SCRATCH_CHUNK_BYTES);
            let buffer = self
                .context
                .device
                .newBufferWithLength_options(chunk_len, MTLResourceOptions::StorageModeShared)
                .ok_or(CoreError::Resource("scratch buffer"))?;
            let chunk_index = state.chunks.len();
            state.chunks.push(ScratchChunk {
                buffer: buffer.clone(),
                free: if allocation_len < chunk_len {
                    vec![allocation_len..chunk_len]
                } else {
                    Vec::new()
                },
            });
            (chunk_index, 0..allocation_len, buffer)
        };
        drop(state);
        let lease = Rc::new(ScratchLease {
            state: Rc::downgrade(&self.scratch),
            chunk: chunk_index,
            range: range.clone(),
        });
        Ok(Tensor::new_scratch(
            buffer,
            range.start,
            shape.to_vec(),
            dtype,
            lease,
        ))
    }

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

impl Drop for ScratchLease {
    fn drop(&mut self) {
        let Some(state) = self.state.upgrade() else {
            return;
        };
        let mut state = state.borrow_mut();
        let Some(chunk) = state.chunks.get_mut(self.chunk) else {
            return;
        };
        chunk.free.push(self.range.clone());
        chunk.free.sort_unstable_by_key(|range| range.start);
        let mut merged: Vec<Range<usize>> = Vec::with_capacity(chunk.free.len());
        for range in chunk.free.drain(..) {
            if let Some(previous) = merged.last_mut()
                && previous.end == range.start
            {
                previous.end = range.end;
            } else {
                merged.push(range);
            }
        }
        chunk.free = merged;
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

fn align_up(
    value: usize,
    alignment: usize,
) -> Result<usize, CoreError> {
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded / alignment * alignment)
        .ok_or_else(|| CoreError::Shape("scratch allocation size overflow".into()))
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
