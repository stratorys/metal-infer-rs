use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLComputeCommandEncoder,
    MTLComputePassDescriptor, MTLSize,
};

use crate::gpu::context::checked_elements;
use crate::gpu::library::Pipeline;
use crate::gpu::profiling::KernelBatchProfile;
use crate::gpu::scratch::{self, ScratchPool};
use crate::gpu::{DType, GpuError, MetalContext, Tensor};

#[derive(Clone, Copy, Debug, Default)]
pub struct DispatchStats {
    pub gpu_time: Duration,
    pub wall_time: Duration,
}

pub(crate) struct CommandBatch<'context> {
    context: &'context MetalContext,
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    scratch: ScratchPool,
    profile: Option<KernelBatchProfile>,
    committed: bool,
}

pub struct PendingBatch<'context> {
    context: &'context MetalContext,
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    scratch: ScratchPool,
    profile: Option<KernelBatchProfile>,
    started: Instant,
    completed: bool,
}

impl<'context> CommandBatch<'context> {
    pub(super) fn new(
        context: &'context MetalContext,
        command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        profile: Option<KernelBatchProfile>,
        scratch: ScratchPool,
    ) -> Result<Self, GpuError> {
        let encoder = if profile.is_some() {
            None
        } else {
            Some(
                command_buffer
                    .computeCommandEncoder()
                    .ok_or(GpuError::ComputeEncoderCreation)?,
            )
        };
        Ok(Self {
            context,
            command_buffer,
            encoder,
            scratch,
            profile,
            committed: false,
        })
    }

    pub(crate) fn dispatch<T>(
        &mut self,
        pipeline: &Pipeline,
        kernel: &str,
        tensors: &[&Tensor],
        params: &T,
        grid: MTLSize,
        threadgroup: MTLSize,
    ) -> Result<(), GpuError> {
        let profile_index = self
            .profile
            .as_mut()
            .map(|profile| profile.reserve(kernel))
            .transpose()?;
        let profiled_encoder = profile_index
            .map(|index| self.profiled_encoder(index))
            .transpose()?;
        let encoder = if let Some(encoder) = &profiled_encoder {
            encoder.as_ref()
        } else {
            self.encoder()?
        };
        encoder.setComputePipelineState(pipeline);
        for (index, tensor) in tensors.iter().enumerate() {
            // SAFETY: tensor resources remain alive through command completion
            // and each kernel's binding order is fixed by its safe
            // wrapper above.
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(tensor.buffer.as_ref()),
                    tensor.offset_bytes,
                    index,
                )
            };
        }
        let (pointer, length): (NonNull<c_void>, usize) = unsafe { bytes(params) };
        // SAFETY: Metal copies `length` bytes from a valid repr(C)/scalar value
        // while encoding.
        unsafe { encoder.setBytes_length_atIndex(pointer, length, tensors.len()) };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
        if let Some(encoder) = profiled_encoder {
            encoder.endEncoding();
        }
        Ok(())
    }

    fn end_compute_encoding(&mut self) -> Result<(), GpuError> {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        } else if self.profile.is_none() {
            return Err(GpuError::EncoderFinished);
        }
        Ok(())
    }

    fn profiled_encoder(
        &self,
        sample_index: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, GpuError> {
        let profile = self
            .profile
            .as_ref()
            .ok_or_else(|| GpuError::MissingTimestampBuffer)?;
        let descriptor = MTLComputePassDescriptor::computePassDescriptor();
        // SAFETY: Metal compute pass descriptors always have attachment slot 0.
        let attachment = unsafe {
            descriptor
                .sampleBufferAttachments()
                .objectAtIndexedSubscript(0)
        };
        attachment.setSampleBuffer(Some(&profile.buffer));
        // SAFETY: reserve() checked both indices against the sample buffer capacity.
        unsafe {
            attachment.setStartOfEncoderSampleIndex(sample_index);
            attachment.setEndOfEncoderSampleIndex(sample_index + 1)
        };
        self.command_buffer
            .computeCommandEncoderWithDescriptor(&descriptor)
            .ok_or(GpuError::ProfiledEncoderCreation)
    }

    pub(crate) fn empty(
        &self,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor, GpuError> {
        let elements = checked_elements(shape)?;
        let byte_len = elements
            .checked_mul(dtype.size())
            .ok_or_else(|| GpuError::ByteLengthOverflow)?;
        let allocation = scratch::allocate(&self.scratch, &self.context.device, byte_len)?;
        Ok(Tensor::new_scratch(
            allocation.buffer,
            allocation.offset_bytes,
            shape.to_vec(),
            dtype,
            allocation.lease,
        ))
    }

    fn encoder(&self) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>, GpuError> {
        self.encoder.as_deref().ok_or(GpuError::EncoderFinished)
    }

    pub(crate) fn commit(mut self) -> Result<PendingBatch<'context>, GpuError> {
        self.end_compute_encoding()?;
        let started = Instant::now();
        self.command_buffer.commit();
        self.committed = true;
        Ok(PendingBatch {
            context: self.context,
            command_buffer: self.command_buffer.clone(),
            scratch: self.scratch.clone(),
            profile: self.profile.take(),
            started,
            completed: false,
        })
    }

    pub(crate) fn finish(self) -> Result<DispatchStats, GpuError> {
        self.commit()?.wait()
    }
}

impl PendingBatch<'_> {
    pub fn wait(mut self) -> Result<DispatchStats, GpuError> {
        self.command_buffer.waitUntilCompleted();
        self.completed = true;
        scratch::release(&self.scratch);
        let wall_time = self.started.elapsed();
        if self.command_buffer.status() == MTLCommandBufferStatus::Error {
            return Err(self
                .command_buffer
                .error()
                .map_or(GpuError::CommandWithoutError, GpuError::Command));
        }
        let gpu_seconds =
            (self.command_buffer.GPUEndTime() - self.command_buffer.GPUStartTime()).max(0.0);
        if let Some(profile) = self.profile.take() {
            self.context.record_kernel_profiles(profile)?;
        }
        Ok(DispatchStats {
            gpu_time: Duration::from_secs_f64(gpu_seconds),
            wall_time,
        })
    }
}

impl Drop for PendingBatch<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.command_buffer.waitUntilCompleted();
            scratch::release(&self.scratch);
        }
    }
}

impl Drop for CommandBatch<'_> {
    fn drop(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        }
        if !self.committed {
            scratch::release(&self.scratch);
        }
    }
}

unsafe fn bytes<T>(value: &T) -> (NonNull<c_void>, usize) {
    let pointer = NonNull::from(value).cast();
    (pointer, std::mem::size_of::<T>())
}
