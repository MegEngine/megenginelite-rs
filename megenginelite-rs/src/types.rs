use crate::ffi::*;

pub type LiteResult<T> = std::result::Result<T, LiteError>;

/// A error type
#[derive(Debug)]
pub enum LiteError {
    /// A megenginelite error with a description
    MGELiteError(String),
    /// Dynamic library cannot be loaded
    LoadingFault,
    /// The version is not match
    VersionNotMatch(String),
}

/// A type to describe device
#[non_exhaustive]
pub struct DeviceType;

impl DeviceType {
    pub const CPU: LiteDeviceType = LiteDeviceType_LITE_CPU;
    pub const CUDA: LiteDeviceType = LiteDeviceType_LITE_CUDA;
    pub const NPU: LiteDeviceType = LiteDeviceType_LITE_NPU;
    pub const ATLAS: LiteDeviceType = LiteDeviceType_LITE_ATLAS;
    pub const DEFAULT: LiteDeviceType = LiteDeviceType_LITE_DEVICE_DEFAULT;
    pub const CAMBRICON: LiteDeviceType = LiteDeviceType_LITE_CAMBRICON;
}

/// A type to describe data
#[non_exhaustive]
pub struct DataType;

impl DataType {
    pub const F32: LiteDataType = LiteDataType_LITE_FLOAT;
    pub const F16: LiteDataType = LiteDataType_LITE_HALF;
    pub const I32: LiteDataType = LiteDataType_LITE_INT;
    pub const I16: LiteDataType = LiteDataType_LITE_INT16;
    pub const I8: LiteDataType = LiteDataType_LITE_INT8;
    pub const U32: LiteDataType = LiteDataType_LITE_UINT;
    pub const U8: LiteDataType = LiteDataType_LITE_UINT8;
    pub const U16: LiteDataType = LiteDataType_LITE_UINT16;
    pub const I64: LiteDataType = LiteDataType_LITE_INT64;

    pub fn width(ty: LiteDataType) -> usize {
        match ty {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::I32 => 4,
            Self::I16 => 2,
            Self::I8 => 1,
            Self::U8 => 1,
            Self::U32 => 4,
            Self::U16 => 2,
            Self::I64 => 8,
            _ => unreachable!(),
        }
    }
}

/// A type to describe fastrun strategy
#[non_exhaustive]
pub struct AlgoSelectStrategy;

impl AlgoSelectStrategy {
    pub const HEURISTIC: LiteAlgoSelectStrategy = LiteAlgoSelectStrategy_LITE_ALGO_HEURISTIC;
    pub const PROFILE: LiteAlgoSelectStrategy = LiteAlgoSelectStrategy_LITE_ALGO_PROFILE;
    pub const REPRODUCIBLE: LiteAlgoSelectStrategy = LiteAlgoSelectStrategy_LITE_ALGO_REPRODUCIBLE;
    pub const OPTIMIZED: LiteAlgoSelectStrategy = LiteAlgoSelectStrategy_LITE_ALGO_OPTIMIZED;
}

/// A type to describe network's input and output
#[non_exhaustive]
pub struct IOType;

impl IOType {
    pub const VALUE: LiteIOType = LiteIOType_LITE_IO_VALUE;
    pub const SHAPE: LiteIOType = LiteIOType_LITE_IO_SHAPE;
}
