use std::ptr::NonNull;
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLCommonCounterSetTimestamp, MTLCounterSampleBuffer, MTLCounterSampleBufferDescriptor,
    MTLCounterSamplingPoint, MTLCounterSet, MTLDevice, MTLStorageMode,
};

use crate::gpu::{GpuError, MetalContext};

const PROFILE_SAMPLE_CAPACITY: usize = 2048;

#[derive(Clone, Debug)]
pub struct KernelDispatchProfile {
    pub kernel: String,
    pub gpu_time: Duration,
}

pub(super) struct KernelBatchProfile {
    pub(super) buffer: Retained<ProtocolObject<dyn MTLCounterSampleBuffer>>,
    kernels: Vec<String>,
}

impl KernelBatchProfile {
    pub(super) fn reserve(
        &mut self,
        kernel: &str,
    ) -> Result<usize, GpuError> {
        let index = self.kernels.len() * 2;
        if index + 2 > PROFILE_SAMPLE_CAPACITY {
            return Err(GpuError::TooManyProfiledDispatches);
        }
        self.kernels.push(kernel.to_owned());
        Ok(index)
    }

    pub(super) fn resolve(
        self,
        nanoseconds_per_tick: f64,
    ) -> Result<Vec<KernelDispatchProfile>, GpuError> {
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
        .ok_or_else(|| GpuError::TimestampResolution)?;
        // SAFETY: the resolved NSData is immutable and remains alive for this slice.
        let bytes = unsafe { resolved.as_bytes_unchecked() };
        if bytes.len() != count * size_of::<u64>() {
            return Err(GpuError::TimestampBufferSize);
        }
        self.kernels
            .into_iter()
            .enumerate()
            .map(|(index, kernel)| {
                let read = |sample: usize| -> Result<u64, GpuError> {
                    let offset = sample * size_of::<u64>();
                    let timestamp = bytes
                        .get(offset..offset + size_of::<u64>())
                        .and_then(|slice| slice.try_into().ok())
                        .ok_or_else(|| GpuError::MissingTimestamp)?;
                    Ok(u64::from_ne_bytes(timestamp))
                };
                let start = read(index * 2)?;
                let end = read(index * 2 + 1)?;
                if start == 0 || end == 0 || start == u64::MAX || end == u64::MAX || end < start {
                    return Err(GpuError::InvalidTimestamps);
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

impl MetalContext {
    /// Enables timestamp sampling for subsequent batches. Each profiled dispatch
    /// uses a separate compute pass, so timings are diagnostic only.
    pub fn set_kernel_profiling(
        &self,
        enabled: bool,
    ) -> Result<(), GpuError> {
        if !enabled {
            self.profile_tick_nanoseconds.set(None);
            return Ok(());
        }
        if !self
            .device
            .supportsCounterSampling(MTLCounterSamplingPoint::AtStageBoundary)
        {
            return Err(GpuError::StageBoundaryCountersUnsupported);
        }
        let sets = self
            .device
            .counterSets()
            .ok_or_else(|| GpuError::NoCounterSets)?;
        let has_timestamps = (0..sets.count()).any(|index| {
            let set = sets.objectAtIndex(index);
            set.name()
                .isEqualToString(unsafe { MTLCommonCounterSetTimestamp })
        });
        if !has_timestamps {
            return Err(GpuError::NoTimestampCounterSet);
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
            )
        };
        std::thread::sleep(Duration::from_millis(20));
        // SAFETY: both pointers refer to writable u64 values.
        unsafe {
            self.device.sampleTimestamps_gpuTimestamp(
                NonNull::from(&mut cpu_end),
                NonNull::from(&mut gpu_end),
            )
        };
        if cpu_end <= cpu_start || gpu_end <= gpu_start {
            return Err(GpuError::TimestampCalibration);
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

    pub(super) fn new_batch_profile(&self) -> Result<Option<KernelBatchProfile>, GpuError> {
        if self.profile_tick_nanoseconds.get().is_none() {
            return Ok(None);
        }
        let sets = self
            .device
            .counterSets()
            .ok_or_else(|| GpuError::NoCounterSets)?;
        let counter_set = (0..sets.count())
            .map(|index| sets.objectAtIndex(index))
            .find(|set| {
                set.name()
                    .isEqualToString(unsafe { MTLCommonCounterSetTimestamp })
            })
            .ok_or_else(|| GpuError::NoTimestampCounterSet)?;
        let descriptor = MTLCounterSampleBufferDescriptor::new();
        descriptor.setCounterSet(Some(&counter_set));
        descriptor.setStorageMode(MTLStorageMode::Shared);
        // SAFETY: the constant is below the Metal sample-buffer limit on Apple GPUs.
        unsafe { descriptor.setSampleCount(PROFILE_SAMPLE_CAPACITY) };
        let buffer = self
            .device
            .newCounterSampleBufferWithDescriptor_error(&descriptor)
            .map_err(GpuError::TimestampBufferCreation)?;
        Ok(Some(KernelBatchProfile {
            buffer,
            kernels: Vec::new(),
        }))
    }

    pub(super) fn record_kernel_profiles(
        &self,
        profile: KernelBatchProfile,
    ) -> Result<(), GpuError> {
        let nanoseconds_per_tick = self
            .profile_tick_nanoseconds
            .get()
            .ok_or_else(|| GpuError::ProfilingDisabled)?;
        self.kernel_profiles
            .borrow_mut()
            .extend(profile.resolve(nanoseconds_per_tick)?);
        Ok(())
    }
}
