/// NNUE quantization utilities.
/// Scaling constants and conversion functions for i16/i8 weight and activation representation.

/// Feature weight quantization scale (FP32 → i16)
pub const WEIGHT_SCALE: i32 = 64;

/// Scale for converting accumulator output to hidden layer input
pub const ACTIVATION_SCALE: i32 = 256;

/// Final output scale (evaluation in centipawn units)
pub const OUTPUT_SCALE: i32 = 16;

/// ClippedReLU: clamp val to \[0, max\]
#[inline]
pub fn clipped_relu(val: i16, max: i16) -> i16 {
    val.max(0).min(max)
}

/// SCReLU (f32): `clamp(val, 0, max)²`
#[inline]
pub fn screlu_f32(val: f32, max: f32) -> f32 {
    let clamped = val.max(0.0).min(max);
    clamped * clamped
}

/// SCReLU derivative (f32): `2 * clamp(val, 0, max)` (active region only)
#[inline]
pub fn screlu_grad_f32(val: f32, max: f32) -> f32 {
    if val > 0.0 && val < max {
        2.0 * val
    } else {
        0.0
    }
}

/// CReLU derivative (f32): 1 if 0 < val < max, else 0
#[inline]
pub fn crelu_grad_f32(val: f32, max: f32) -> f32 {
    if val > 0.0 && val < max {
        1.0
    } else {
        0.0
    }
}

/// Saturating i32 → i16 conversion
#[inline]
pub fn saturate_i16(val: i32) -> i16 {
    val.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}
