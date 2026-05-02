pub mod activations;
pub mod conv2d;
pub mod linear;
pub mod ops;
pub mod weights;

/// Activation function types matching C libopus ACTIVATION_* constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Activation {
    Linear = 0,
    Sigmoid = 1,
    Tanh = 2,
    Relu = 3,
    Softmax = 4,
    Swish = 5,
    Exp = 6,
}

impl Activation {
    pub fn from_i32(v: i32) -> Self {
        match v {
            0 => Activation::Linear,
            1 => Activation::Sigmoid,
            2 => Activation::Tanh,
            3 => Activation::Relu,
            4 => Activation::Softmax,
            5 => Activation::Swish,
            6 => Activation::Exp,
            _ => panic!("unknown activation type: {v}"),
        }
    }
}

/// Generic sparse affine transformation layer.
/// Matches C `LinearLayer` from nnet.h.
#[derive(Debug, Clone)]
pub struct LinearLayer {
    pub bias: Option<Vec<f32>>,
    /// SU-bias: used with unsigned quantization path (x86 USE_SU_BIAS).
    /// Compensates for the 127 offset in the unsigned input quantization.
    pub subias: Option<Vec<f32>>,
    pub weights: Option<Vec<i8>>,
    pub float_weights: Option<Vec<f32>>,
    pub weights_idx: Option<Vec<i32>>,
    pub diag: Option<Vec<f32>>,
    pub scale: Option<Vec<f32>>,
    pub nb_inputs: usize,
    pub nb_outputs: usize,
}

impl LinearLayer {
    pub fn new(nb_inputs: usize, nb_outputs: usize) -> Self {
        LinearLayer {
            bias: None,
            subias: None,
            weights: None,
            float_weights: None,
            weights_idx: None,
            diag: None,
            scale: None,
            nb_inputs,
            nb_outputs,
        }
    }
}

/// 2D convolution layer.
/// Matches C `Conv2dLayer` from nnet.h.
#[derive(Debug, Clone)]
pub struct Conv2dLayer {
    pub bias: Option<Vec<f32>>,
    pub float_weights: Option<Vec<f32>>,
    pub in_channels: usize,
    pub out_channels: usize,
    pub ktime: usize,
    pub kheight: usize,
}

impl Conv2dLayer {
    pub fn new(in_channels: usize, out_channels: usize, ktime: usize, kheight: usize) -> Self {
        Conv2dLayer {
            bias: None,
            float_weights: None,
            in_channels,
            out_channels,
            ktime,
            kheight,
        }
    }
}

/// Weight type identifiers matching C WEIGHT_TYPE_* constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum WeightType {
    Float = 0,
    Int = 1,
    QWeight = 2,
    Int8 = 3,
}

/// A named weight array entry (matches C `WeightArray`).
#[derive(Debug, Clone)]
pub struct WeightArray {
    pub name: String,
    pub weight_type: WeightType,
    pub data: Vec<u8>,
}

pub const WEIGHT_BLOB_VERSION: i32 = 0;
pub const WEIGHT_BLOCK_SIZE: i32 = 64;
