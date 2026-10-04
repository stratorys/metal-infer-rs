use std::cell::RefCell;
use std::rc::Rc;

use half::{bf16, f16};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLResourceOptions,
};

use crate::gpu::batch::CommandBatch;
use crate::gpu::profiling::{KernelDispatchProfile, Profiler};
use crate::gpu::scratch::{self, ScratchPool};
use crate::gpu::tensor::{from_buffer, metal_buffer};
use crate::gpu::{DType, GpuError, Tensor};

#[derive(Clone)]
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    device_name: String,
    profiler: Rc<Profiler>,
    scratch_pools: Rc<RefCell<Vec<ScratchPool>>>,
}

impl MetalContext {
    pub fn new() -> Result<Self, GpuError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(GpuError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or(GpuError::CommandQueueCreation)?;
        let device_name = device.name().to_string();
        Ok(Self {
            device,
            queue,
            device_name,
            profiler: Rc::new(Profiler::new()),
            scratch_pools: Rc::new(RefCell::new(Vec::new())),
        })
    }

    pub fn device_name(&self) -> String {
        self.device_name.clone()
    }

    pub fn allocated_bytes(&self) -> usize {
        self.device.currentAllocatedSize()
    }

    /// Enables timestamp sampling for subsequent batches. Each profiled dispatch
    /// uses a separate compute pass, so timings are diagnostic only.
    pub fn set_kernel_profiling(
        &self,
        enabled: bool,
    ) -> Result<(), GpuError> {
        self.profiler.set_enabled(&self.device, enabled)
    }

    pub fn take_kernel_profiles(&self) -> Vec<KernelDispatchProfile> {
        self.profiler.take()
    }

    pub fn empty(
        &self,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor, GpuError> {
        let elements = checked_elements(shape)?;
        let byte_len = elements
            .checked_mul(dtype.size())
            .ok_or_else(|| GpuError::ByteLengthOverflow)?;
        let buffer = self
            .device
            .newBufferWithLength_options(byte_len.max(1), MTLResourceOptions::StorageModeShared)
            .ok_or(GpuError::BufferCreation)?;
        Ok(from_buffer(buffer, shape.to_vec(), dtype))
    }

    pub fn tensor_f16(
        &self,
        values: &[f32],
        shape: &[usize],
    ) -> Result<Tensor, GpuError> {
        let tensor = self.empty(shape, DType::F16)?;
        check_data_len(tensor.len(), values.len())?;
        let destination = metal_buffer(&tensor).contents().as_ptr().cast::<u16>();
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
    ) -> Result<Tensor, GpuError> {
        let tensor = self.empty(shape, DType::F16)?;
        check_data_len(tensor.len(), values.len())?;
        // SAFETY: source and destination are valid, non-overlapping,
        // equal-length regions.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                metal_buffer(&tensor).contents().as_ptr().cast::<u16>(),
                values.len(),
            )
        };
        Ok(tensor)
    }

    pub fn tensor_f16_bytes(
        &self,
        bytes: &[u8],
        shape: &[usize],
    ) -> Result<Tensor, GpuError> {
        let tensor = self.empty(shape, DType::F16)?;
        if tensor.byte_len() != bytes.len() {
            return Err(GpuError::DataLengthMismatch);
        }
        // SAFETY: both regions are valid for bytes.len() bytes and do not
        // overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                metal_buffer(&tensor).contents().as_ptr().cast::<u8>(),
                bytes.len(),
            )
        };
        Ok(tensor)
    }

    /// Converts little-endian BF16 source data directly into an FP16 Metal
    /// buffer. This deliberately avoids allocating a second, tensor-sized
    /// host vector while loading large sharded checkpoints.
    pub fn tensor_bf16_as_f16_bytes(
        &self,
        bytes: &[u8],
        shape: &[usize],
    ) -> Result<Tensor, GpuError> {
        let tensor = self.empty(shape, DType::F16)?;
        if tensor.byte_len() != bytes.len() {
            return Err(GpuError::DataLengthMismatch);
        }
        let (values, remainder) = bytes.as_chunks::<2>();
        if !remainder.is_empty() {
            return Err(GpuError::DataLengthMismatch);
        }
        let destination = metal_buffer(&tensor).contents().as_ptr().cast::<u16>();
        for (index, [low, high]) in values.iter().enumerate() {
            let value = bf16::from_bits(u16::from_le_bytes([*low, *high]));
            // SAFETY: `values` contains exactly `tensor.len()` elements and
            // the Metal shared buffer has one u16 slot per FP16 element.
            unsafe {
                destination
                    .add(index)
                    .write(f16::from_f32(value.to_f32()).to_bits())
            };
        }
        Ok(tensor)
    }

    pub fn tensor_u32(
        &self,
        values: &[u32],
        shape: &[usize],
    ) -> Result<Tensor, GpuError> {
        let tensor = self.empty(shape, DType::U32)?;
        check_data_len(tensor.len(), values.len())?;
        // SAFETY: source and destination are valid, non-overlapping,
        // equal-length regions.
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                metal_buffer(&tensor).contents().as_ptr().cast::<u32>(),
                values.len(),
            )
        };
        Ok(tensor)
    }

    fn command_buffer(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, GpuError> {
        self.queue
            .commandBuffer()
            .ok_or(GpuError::CommandBufferCreation)
    }
}

pub fn device(context: &MetalContext) -> &Retained<ProtocolObject<dyn MTLDevice>> {
    &context.device
}

pub fn profiler(context: &MetalContext) -> &Profiler {
    &context.profiler
}

pub fn begin_batch(context: &MetalContext) -> Result<CommandBatch<'_>, GpuError> {
    let profile = context.profiler.new_batch_profile(&context.device)?;
    let command_buffer = context.command_buffer()?;
    CommandBatch::new(
        context,
        command_buffer,
        profile,
        scratch::acquire(&context.scratch_pools),
    )
}

pub fn checked_elements(shape: &[usize]) -> Result<usize, GpuError> {
    if shape.is_empty() || shape.contains(&0) {
        return Err(GpuError::EmptyShape);
    }
    shape.iter().try_fold(1usize, |length, dimension| {
        length
            .checked_mul(*dimension)
            .ok_or_else(|| GpuError::ElementCountOverflow)
    })
}

fn check_data_len(
    expected: usize,
    actual: usize,
) -> Result<(), GpuError> {
    if expected == actual {
        Ok(())
    } else {
        Err(GpuError::DataLengthMismatch)
    }
}
