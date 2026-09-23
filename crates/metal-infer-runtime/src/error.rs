use objc2::rc::Retained;
use objc2_foundation::NSError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CoreError {
    #[error("no Metal device is available")]
    NoDevice,
    #[error("could not create Metal resource: {0}")]
    Resource(&'static str),
    #[error("Metal shader compilation failed: {0}")]
    Shader(Retained<NSError>),
    #[error("Metal pipeline creation failed: {0}")]
    Pipeline(Retained<NSError>),
    #[error("Metal kernel `{0}` was not found in the shader library")]
    MissingKernel(String),
    #[error("Metal command failed: {0}")]
    Command(Retained<NSError>),
    #[error("Metal command failed without providing an NSError")]
    UnknownCommand,
    #[error("invalid tensor shape: {0}")]
    Shape(String),
    #[error("unsupported dtype: expected {expected}, got {actual}")]
    DType {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("host data has {actual} elements, expected {expected}")]
    DataLength { expected: usize, actual: usize },
    #[error("GPU kernel profiling failed: {0}")]
    Profiling(String),
}
