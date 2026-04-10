/// NNUE 양자화 유틸리티
/// 가중치와 활성화를 i16/i8로 표현하기 위한 스케일링 상수 및 변환 함수

/// 피처 가중치 양자화 스케일 (FP32 → i16)
pub const WEIGHT_SCALE: i32 = 64;

/// Accumulator 출력을 Hidden 레이어 입력으로 변환할 때의 스케일
pub const ACTIVATION_SCALE: i32 = 256;

/// 최종 출력 스케일 (평가값을 centipawn 단위로)
pub const OUTPUT_SCALE: i32 = 16;

/// ClippedReLU: val을 [0, max] 범위로 클램프
#[inline]
pub fn clipped_relu(val: i16, max: i16) -> i16 {
    val.max(0).min(max)
}

/// SCReLU (f32): clamp(val, 0, max)²
#[inline]
pub fn screlu_f32(val: f32, max: f32) -> f32 {
    let clamped = val.max(0.0).min(max);
    clamped * clamped
}

/// SCReLU 미분 (f32): 2 * clamp(val, 0, max)  (활성 구간에서만)
#[inline]
pub fn screlu_grad_f32(val: f32, max: f32) -> f32 {
    if val > 0.0 && val < max {
        2.0 * val
    } else {
        0.0
    }
}

/// CReLU 미분 (f32): 1 if 0 < val < max, else 0
#[inline]
pub fn crelu_grad_f32(val: f32, max: f32) -> f32 {
    if val > 0.0 && val < max {
        1.0
    } else {
        0.0
    }
}

/// i32 누적값을 i16으로 안전하게 변환 (saturating)
#[inline]
pub fn saturate_i16(val: i32) -> i16 {
    val.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}
