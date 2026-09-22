use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ops::Range;
use std::ptr::NonNull;
use std::rc::{Rc, Weak};
use std::time::Duration;
use std::time::Instant;

use half::{bf16, f16};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCommonCounterSetTimestamp, MTLComputeCommandEncoder, MTLComputePassDescriptor,
    MTLComputePipelineState, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLCreateSystemDefaultDevice, MTLDevice, MTLFunction,
    MTLLibrary, MTLResourceOptions, MTLStorageMode,
};

use crate::{CoreError, DType, Tensor};

const SHADERS: &str = include_str!("kernels/transformer.metal");
const PROFILE_SAMPLE_CAPACITY: usize = 2048;

pub type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

#[derive(Clone, Copy, Debug, Default)]
pub struct DispatchStats {
    pub gpu_time: Duration,
    pub wall_time: Duration,
}

#[derive(Clone, Debug)]
pub struct KernelDispatchProfile {
    pub kernel: String,
    pub gpu_time: Duration,
}

pub(crate) struct KernelBatchProfile {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    kernels: Vec<String>,
}

impl KernelBatchProfile {
    pub(crate) fn reserve(
        &mut self,
        kernel: &str,
    ) -> Result<usize, CoreError> {
        let index = self.kernels.len() * 2;
        if index + 2 > PROFILE_SAMPLE_CAPACITY {
            return Err(CoreError::Profiling(
                "too many dispatches in one command batch for the timestamp buffer".into(),
            ));
        }
        self.kernels.push(kernel.to_owned());
        Ok(index)
    }

    fn resolve(
        self,
        nanoseconds_per_tick: f64,
    ) -> Result<Vec<KernelDispatchProfile>, CoreError> {
        let count = self.kernels.len() * 2;
        if count == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: every index in the requested range was reserved and encoded.
        let resolved = unsafe {
            self.buffer.resolveCounterRange(NSRange {
                location: 0,
                length: count,
            })
        }
        .ok_or_else(|| CoreError::Profiling("could not resolve GPU timestamps".into()))?;
        // SAFETY: the resolved NSData is immutable and remains alive for this slice.
        let bytes = unsafe { resolved.as_bytes_unchecked() };
        if bytes.len() != count * size_of::<u64>() {
            return Err(CoreError::Profiling(
                "GPU timestamp buffer has an unexpected size".into(),
            ));
        }
        self.kernels
            .into_iter()
            .enumerate()
            .map(|(index, kernel)| {
                let read = |sample: usize| {
                    let offset = sample * size_of::<u64>();
                    u64::from_ne_bytes(bytes[offset..offset + 8].try_into().unwrap())
                };
                let start = read(index * 2);
                let end = read(index * 2 + 1);
                if start == 0 || end == 0 || start == u64::MAX || end == u64::MAX || end < start {
                    return Err(CoreError::Profiling(format!(
                        "invalid GPU timestamps for kernel {kernel}"
                    )));
                }
                Ok(KernelDispatchProfile {
                    kernel,
                    gpu_time: Duration::from_secs_f64(
                        (end - start) as f64 * nanoseconds_per_tick / 1.0e9,
                    ),
                })
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MatmulBackend {
    #[default]
    Auto,
    ReferenceMsl,
    NativeMsl,
    Mps,
}

impl MatmulBackend {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::ReferenceMsl => "reference-msl",
            Self::NativeMsl => "native-msl",
            Self::Mps => "mps",
        }
    }
}

pub struct CommandBatch<'context> {
    pub(crate) context: &'context MetalContext,
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    scratch: Rc<RefCell<ScratchState>>,
    pub(crate) profile: Option<KernelBatchProfile>,
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
    is_m4_pro: bool,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    pipelines: RefCell<HashMap<String, Pipeline>>,
    matmul_backend: Rc<Cell<MatmulBackend>>,
    profile_tick_nanoseconds: Rc<Cell<Option<f64>>>,
    kernel_profiles: Rc<RefCell<Vec<KernelDispatchProfile>>>,
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
        let is_m4_pro = device.name().to_string().contains("M4 Pro");
        Ok(Self {
            device,
            queue,
            is_m4_pro,
            library,
            pipelines: RefCell::new(HashMap::new()),
            matmul_backend: Rc::new(Cell::new(MatmulBackend::Auto)),
            profile_tick_nanoseconds: Rc::new(Cell::new(None)),
            kernel_profiles: Rc::new(RefCell::new(Vec::new())),
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    pub(crate) fn is_m4_pro(&self) -> bool {
        self.is_m4_pro
    }

    pub fn allocated_bytes(&self) -> usize {
        self.device.currentAllocatedSize()
    }

    pub fn set_matmul_backend(
        &self,
        backend: MatmulBackend,
    ) {
        self.matmul_backend.set(backend);
    }

    pub fn matmul_backend(&self) -> MatmulBackend {
        self.matmul_backend.get()
    }

    /// Enables timestamp sampling for subsequent batches. Each profiled dispatch
    /// uses a separate compute pass, so timings are diagnostic only.
    pub fn set_kernel_profiling(
        &self,
        enabled: bool,
    ) -> Result<(), CoreError> {
        if !enabled {
            self.profile_tick_nanoseconds.set(None);
            return Ok(());
        }
        if !self
            .device
            .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
        {
            return Err(CoreError::Profiling(
                "GPU does not support counters at compute pass boundaries".into(),
            ));
        }
        let sets = self
            .device
            .counterSets()
            .ok_or_else(|| CoreError::Profiling("GPU exposes no counter sets".into()))?;
        let has_timestamps = (0..sets.count()).any(|index| {
            let set = sets.objectAtIndex(index);
            set.name()
                .isEqualToString(unsafe { MTLCommonCounterSetTimestamp })
        });
        if !has_timestamps {
            return Err(CoreError::Profiling(
                "GPU exposes no timestamp counter set".into(),
            ));
        }
        let mut cpu_start = 0;
        let mut gpu_start = 0;
        let mut cpu_end = 0;
        let mut gpu_end = 0;
        // SAFETY: all four pointers refer to writable u64 values.
        unsafe {
            self.device.sampleTimestamps_gpuTimestamp(
                NonNull::from(&mut cpu_start),
                NonNull::from(&mut gpu_start),
            );
        }
        std::thread::sleep(Duration::from_millis(20));
        // SAFETY: both pointers refer to writable u64 values.
        unsafe {
            self.device.sampleTimestamps_gpuTimestamp(
                NonNull::from(&mut cpu_end),
                NonNull::from(&mut gpu_end),
            );
        }
        if cpu_end <= cpu_start || gpu_end <= gpu_start {
            return Err(CoreError::Profiling(
                "could not calibrate GPU timestamps".into(),
            ));
        }
        // Metal's CPU timestamps are already nanoseconds; GPU timestamps use
        // the device clock and need the ratio of the two sampled spans.
        let nanoseconds_per_tick = (cpu_end - cpu_start) as f64 / (gpu_end - gpu_start) as f64;
        self.kernel_profiles.borrow_mut().clear();
        self.profile_tick_nanoseconds
            .set(Some(nanoseconds_per_tick));
        Ok(())
    }

    pub fn take_kernel_profiles(&self) -> Vec<KernelDispatchProfile> {
        std::mem::take(&mut *self.kernel_profiles.borrow_mut())
    }

    fn new_batch_profile(&self) -> Result<Option<KernelBatchProfile>, CoreError> {
        if self.profile_tick_nanoseconds.get().is_none() {
            return Ok(None);
        }
        let sets = self
            .device
            .counterSets()
            .ok_or_else(|| CoreError::Profiling("GPU exposes no counter sets".into()))?;
        let counter_set = (0..sets.count())
            .map(|index| sets.objectAtIndex(index))
            .find(|set| {
                set.name()
                    .isEqualToString(unsafe { MTLCommonCounterSetTimestamp })
            })
            .ok_or_else(|| CoreError::Profiling("GPU exposes no timestamp counter set".into()))?;
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(&counter_set));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: the constant is below the Metal sample-buffer limit on Apple GPUs.
        unsafe { descriptor.setSampleCount(PROFILE_SAMPLE_CAPACITY) };
        let buffer = self
            .device
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .map_err(|error| {
                CoreError::Profiling(format!("could not create timestamp buffer: {error}"))
            })?;
        Ok(Some(KernelBatchProfile {
            buffer,
            kernels: Vec::new(),
        }))
    }

    pub fn begin_batch(&self) -> Result<CommandBatch<'_>, CoreError> {
        let profile = self.new_batch_profile()?;
        let command_buffer = self.command_buffer()?;
        let encoder = if profile.is_some() {
            None
        } else {
            Some(
                command_buffer
                    .computeCommandEncoder()
                    .ok_or(CoreError::Resource("compute encoder"))?,
            )
        };
        Ok(CommandBatch {
            context: self,
            command_buffer,
            encoder,
            scratch: Rc::new(RefCell::new(ScratchState { chunks: Vec::new() })),
            profile,
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

    /// Converts little-endian BF16 source data directly into an FP16 Metal
    /// buffer. This deliberately avoids allocating a second, tensor-sized
    /// host vector while loading large sharded checkpoints.
    pub fn tensor_bf16_as_f16_bytes(
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
        let (values, remainder) = bytes.as_chunks::<2>();
        if !remainder.is_empty() {
            return Err(CoreError::DataLength {
                expected: tensor.byte_len(),
                actual: bytes.len(),
            });
        }
        let destination = tensor.buffer.contents().as_ptr().cast::<u16>();
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
    pub(crate) fn end_compute_encoding(&mut self) -> Result<(), CoreError> {
        if let Some(encoder) = self.encoder.take() {
            encoder.endEncoding();
        } else if self.profile.is_none() {
            return Err(CoreError::Resource("finished compute encoder"));
        }
        Ok(())
    }

    pub(crate) fn resume_compute_encoding(&mut self) -> Result<(), CoreError> {
        if self.profile.is_some() {
            return Ok(());
        }
        let encoder = self
            .command_buffer
            .computeCommandEncoder()
            .ok_or(CoreError::Resource("compute encoder"))?;
        self.encoder = Some(encoder);
        Ok(())
    }

    pub(crate) fn profiled_encoder(
        &self,
        sample_index: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, CoreError> {
        let profile = self
            .profile
            .as_ref()
            .ok_or_else(|| CoreError::Profiling("batch has no timestamp buffer".into()))?;
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
            attachment.setEndOfEncoderSampleIndex(sample_index + 1);
        }
        self.command_buffer
            .computeCommandEncoderWithDescriptor(&descriptor)
            .ok_or(CoreError::Resource("profiled compute encoder"))
    }

    pub(crate) fn command_buffer_ref(&self) -> &ProtocolObject<dyn MTLCommandBuffer> {
        &self.command_buffer
    }

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
            let mut free = Vec::new();
            if allocation_len < chunk_len {
                free.push(allocation_len..chunk_len);
            }
            state.chunks.push(ScratchChunk {
                buffer: buffer.clone(),
                free,
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
        self.end_compute_encoding()?;
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
        if let Some(profile) = self.profile.take() {
            let nanoseconds_per_tick =
                self.context.profile_tick_nanoseconds.get().ok_or_else(|| {
                    CoreError::Profiling(
                        "GPU profiling was disabled before batch completion".into(),
                    )
                })?;
            self.context
                .kernel_profiles
                .borrow_mut()
                .extend(profile.resolve(nanoseconds_per_tick)?);
        }
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
