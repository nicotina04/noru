/// Scalar fallback implementations for SIMD operations.
/// These are the reference implementations that all SIMD paths must match.

/// Saturating i16 vector addition: acc[i] += w[i]
#[inline]
pub fn vec_add_i16(acc: &mut [i16], w: &[i16]) {
    debug_assert_eq!(acc.len(), w.len());
    for i in 0..acc.len() {
        acc[i] = acc[i].saturating_add(w[i]);
    }
}

/// Saturating i16 vector subtraction: acc[i] -= w[i]
#[inline]
pub fn vec_sub_i16(acc: &mut [i16], w: &[i16]) {
    debug_assert_eq!(acc.len(), w.len());
    for i in 0..acc.len() {
        acc[i] = acc[i].saturating_sub(w[i]);
    }
}

/// ClippedReLU: out[i] = clamp(inp[i], 0, 127)
#[inline]
pub fn vec_clipped_relu(out: &mut [i16], inp: &[i16]) {
    debug_assert_eq!(out.len(), inp.len());
    for i in 0..inp.len() {
        out[i] = inp[i].max(0).min(127);
    }
}

/// i16 dot product with i32 accumulation: sum(a[i] * b[i])
#[inline]
pub fn dot_i16_i32(a: &[i16], b: &[i16]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum: i32 = 0;
    for i in 0..a.len() {
        sum += a[i] as i32 * b[i] as i32;
    }
    sum
}

/// SCReLU squared dot product with i64 accumulation: sum(a[i]^2 * b[i])
#[inline]
pub fn dot_screlu_i64(a: &[i16], b: &[i16]) -> i64 {
    debug_assert_eq!(a.len(), b.len());
    let mut sum: i64 = 0;
    for i in 0..a.len() {
        sum += a[i] as i64 * a[i] as i64 * b[i] as i64;
    }
    sum
}
