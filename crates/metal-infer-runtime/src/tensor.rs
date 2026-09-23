use std::rc::Rc;

use half::f16;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

use crate::CoreError;
use crate::context::ScratchLease;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    F16,
    F32,
    U32,
}

impl DType {
    pub const fn size(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::F32 | Self::U32 => 4,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::U32 => "u32",
        }
    }
}

#[derive(Clone)]
pub struct Tensor {
    pub(crate) buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) offset_bytes: usize,
    shape: Vec<usize>,
    dtype: DType,
    scratch: Option<Rc<ScratchLease>>,
}

impl std::fmt::Debug for Tensor {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        formatter
            .debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("bytes", &self.byte_len())
            .finish()
    }
}

impl Tensor {
    pub(crate) fn new(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        shape: Vec<usize>,
        dtype: DType,
    ) -> Self {
        Self {
            buffer,
            offset_bytes: 0,
            shape,
            dtype,
            scratch: None,
        }
    }

    pub(crate) fn new_scratch(
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        offset_bytes: usize,
        shape: Vec<usize>,
        dtype: DType,
        scratch: Rc<ScratchLease>,
    ) -> Self {
        Self {
            buffer,
            offset_bytes,
            shape,
            dtype,
            scratch: Some(scratch),
        }
    }

    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    pub const fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn byte_len(&self) -> usize {
        self.len() * self.dtype.size()
    }

    pub fn reshape(
        &self,
        shape: &[usize],
    ) -> Result<Self, CoreError> {
        let length = shape.iter().try_fold(1usize, |value, dimension| {
            value
                .checked_mul(*dimension)
                .ok_or_else(|| CoreError::ReshapeOverflow)
        })?;
        if shape.is_empty() || shape.contains(&0) || length != self.len() {
            return Err(CoreError::ReshapeMismatch);
        }
        Ok(Self {
            buffer: self.buffer.clone(),
            offset_bytes: self.offset_bytes,
            shape: shape.to_vec(),
            dtype: self.dtype,
            scratch: self.scratch.clone(),
        })
    }

    pub fn prefix(
        &self,
        first_dimension: usize,
    ) -> Result<Self, CoreError> {
        let Some(capacity) = self.shape.first().copied() else {
            return Err(CoreError::PrefixRank);
        };
        if first_dimension == 0 || first_dimension > capacity {
            return Err(CoreError::PrefixOutOfRange);
        }
        let mut shape = self.shape.clone();
        let first = shape.first_mut().ok_or_else(|| CoreError::PrefixRank)?;
        *first = first_dimension;
        Ok(Self {
            buffer: self.buffer.clone(),
            offset_bytes: self.offset_bytes,
            shape,
            dtype: self.dtype,
            scratch: self.scratch.clone(),
        })
    }

    pub fn row(
        &self,
        row: usize,
    ) -> Result<Self, CoreError> {
        let [rows, width] = self.shape.as_slice() else {
            return Err(CoreError::RowRank);
        };
        if row >= *rows {
            return Err(CoreError::RowOutOfRange);
        }
        Ok(Self {
            buffer: self.buffer.clone(),
            offset_bytes: self.offset_bytes + row * *width * self.dtype.size(),
            shape: vec![1, *width],
            dtype: self.dtype,
            scratch: self.scratch.clone(),
        })
    }

    pub fn slice_1d(
        &self,
        start: usize,
        len: usize,
    ) -> Result<Self, CoreError> {
        let [capacity] = self.shape.as_slice() else {
            return Err(CoreError::SliceRank);
        };
        let end = start
            .checked_add(len)
            .ok_or_else(|| CoreError::SliceOverflow)?;
        if len == 0 || end > *capacity {
            return Err(CoreError::SliceOutOfRange);
        }
        Ok(Self {
            buffer: self.buffer.clone(),
            offset_bytes: self.offset_bytes + start * self.dtype.size(),
            shape: vec![len],
            dtype: self.dtype,
            scratch: self.scratch.clone(),
        })
    }

    pub fn with_f16_bits<T>(
        &self,
        read: impl FnOnce(&[u16]) -> T,
    ) -> Result<T, CoreError> {
        if self.dtype != DType::F16 {
            return Err(CoreError::ExpectedF16);
        }
        // SAFETY: the caller reads only after command completion; Tensor owns
        // the shared buffer and checked views remain within its allocation.
        let pointer =
            unsafe { self.buffer.contents().as_ptr().add(self.offset_bytes) }.cast::<u16>();
        let values = unsafe { std::slice::from_raw_parts(pointer, self.len()) };
        Ok(read(values))
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>, CoreError> {
        if self.dtype != DType::F16 {
            return Err(CoreError::ExpectedF16);
        }
        // SAFETY: offset_bytes was derived from a checked tensor view.
        let pointer =
            unsafe { self.buffer.contents().as_ptr().add(self.offset_bytes) }.cast::<u16>();
        // SAFETY: buffers owned by Tensor are allocated for exactly `len`
        // elements and callers only read them after the command buffer
        // has completed.
        let values = unsafe { std::slice::from_raw_parts(pointer, self.len()) };
        Ok(values
            .iter()
            .map(|bits| f16::from_bits(*bits).to_f32())
            .collect())
    }

    pub fn to_u32_vec(&self) -> Result<Vec<u32>, CoreError> {
        if self.dtype != DType::U32 {
            return Err(CoreError::ExpectedU32);
        }
        // SAFETY: offset_bytes was derived from a checked tensor view.
        let pointer =
            unsafe { self.buffer.contents().as_ptr().add(self.offset_bytes) }.cast::<u32>();
        // SAFETY: see `to_f32_vec`; u32 has the dtype's alignment and size.
        Ok(unsafe { std::slice::from_raw_parts(pointer, self.len()) }.to_vec())
    }
}
