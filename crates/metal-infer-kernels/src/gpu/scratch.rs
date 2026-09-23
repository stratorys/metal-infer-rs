use std::cell::RefCell;
use std::ops::Range;
use std::rc::{Rc, Weak};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use crate::gpu::GpuError;

const SCRATCH_ALIGNMENT: usize = 256;
const SCRATCH_CHUNK_BYTES: usize = 1024 * 1024;

pub(super) type ScratchPool = Rc<RefCell<ScratchState>>;

struct ScratchChunk {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    free: Vec<Range<usize>>,
}

pub(super) struct ScratchState {
    chunks: Vec<ScratchChunk>,
    busy: bool,
}

pub(crate) struct ScratchLease {
    state: Weak<RefCell<ScratchState>>,
    chunk: usize,
    range: Range<usize>,
}

pub(super) struct ScratchAllocation {
    pub(super) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(super) offset_bytes: usize,
    pub(super) lease: Rc<ScratchLease>,
}

pub(super) fn acquire(pools: &RefCell<Vec<ScratchPool>>) -> ScratchPool {
    let mut pools = pools.borrow_mut();
    if let Some(state) = pools.iter().find(|state| !state.borrow().busy) {
        let state = state.clone();
        state.borrow_mut().busy = true;
        state
    } else {
        let state = Rc::new(RefCell::new(ScratchState {
            chunks: Vec::new(),
            busy: true,
        }));
        pools.push(state.clone());
        state
    }
}

pub(super) fn release(pool: &ScratchPool) {
    pool.borrow_mut().busy = false;
}

pub(super) fn allocate(
    pool: &ScratchPool,
    device: &ProtocolObject<dyn MTLDevice>,
    byte_len: usize,
) -> Result<ScratchAllocation, GpuError> {
    let allocation_len = align_up(byte_len.max(1), SCRATCH_ALIGNMENT)?;
    let mut state = pool.borrow_mut();
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
        let buffer = device
            .newBufferWithLength_options(chunk_len, MTLResourceOptions::StorageModeShared)
            .ok_or(GpuError::ScratchBufferCreation)?;
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
        state: Rc::downgrade(pool),
        chunk: chunk_index,
        range: range.clone(),
    });
    Ok(ScratchAllocation {
        buffer,
        offset_bytes: range.start,
        lease,
    })
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

fn align_up(
    value: usize,
    alignment: usize,
) -> Result<usize, GpuError> {
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded / alignment * alignment)
        .ok_or_else(|| GpuError::ScratchSizeOverflow)
}
