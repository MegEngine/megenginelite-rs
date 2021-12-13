use megenginelite_sys::*;

#[non_exhaustive]
pub struct DeviceType;

impl DeviceType {
    pub const CPU: LiteDeviceType = LiteDeviceType_LITE_CPU;
    pub const CUDA: LiteDeviceType = LiteDeviceType_LITE_CUDA;
    pub const NPU: LiteDeviceType = LiteDeviceType_LITE_NPU;
    pub const ATLAS: LiteDeviceType = LiteDeviceType_LITE_ATLAS;
}

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
}

#[non_exhaustive]
pub struct AlgoSelectStrategy;

impl AlgoSelectStrategy {
    pub const HEURISTIC: LiteAlgoSelectStrategy = LiteAlgoSelectStrategy_LITE_ALGO_HEURISTIC;
    pub const PROFILE: LiteAlgoSelectStrategy = LiteAlgoSelectStrategy_LITE_ALGO_PROFILE;
    pub const REPRODUCIBLE: LiteAlgoSelectStrategy = LiteAlgoSelectStrategy_LITE_ALGO_REPRODUCIBLE;
    pub const OPTIMIZED: LiteAlgoSelectStrategy = LiteAlgoSelectStrategy_LITE_ALGO_OPTIMIZED;
}

#[non_exhaustive]
pub struct IOType;

impl IOType {
    pub const VALUE: LiteIOType = LiteIOType_LITE_IO_VALUE;
    pub const SHAPE: LiteIOType = LiteIOType_LITE_IO_SHAPE;
}
