/// SIMD-accelerated primitives for NNUE inference.
///
/// Platform dispatch:
/// - x86_64 with AVX2: 256-bit SIMD (16 × i16)
/// - aarch64 with NEON: 128-bit SIMD (8 × i16)
/// - Fallback: scalar implementation
pub mod scalar;

#[cfg(target_arch = "x86_64")]
mod avx2;

#[cfg(target_arch = "aarch64")]
mod neon;

/// Saturating i16 vector addition: `acc[i] += w[i]`
#[inline]
pub fn vec_add_i16(acc: &mut [i16], w: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::vec_add_i16(acc, w) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::vec_add_i16(acc, w) };
        return;
    }
    scalar::vec_add_i16(acc, w);
}

/// Saturating i16 vector subtraction: `acc[i] -= w[i]`
#[inline]
pub fn vec_sub_i16(acc: &mut [i16], w: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::vec_sub_i16(acc, w) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::vec_sub_i16(acc, w) };
        return;
    }
    scalar::vec_sub_i16(acc, w);
}

/// ClippedReLU: `out[i] = clamp(inp[i], 0, 127)`
#[inline]
pub fn vec_clipped_relu(out: &mut [i16], inp: &[i16]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { avx2::vec_clipped_relu(out, inp) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { neon::vec_clipped_relu(out, inp) };
        return;
    }
    scalar::vec_clipped_relu(out, inp);
}

/// i16 dot product with i32 accumulation
#[inline]
pub fn dot_i16_i32(a: &[i16], b: &[i16]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::dot_i16_i32(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::dot_i16_i32(a, b) };
    }
    scalar::dot_i16_i32(a, b)
}

/// SCReLU squared dot product with i64 accumulation
#[inline]
pub fn dot_screlu_i64(a: &[i16], b: &[i16]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { avx2::dot_screlu_i64(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { neon::dot_screlu_i64(a, b) };
    }
    scalar::dot_screlu_i64(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vec(len: usize, seed: u64) -> Vec<i16> {
        let mut state = if seed == 0 { 1u64 } else { seed };
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 201) as i16 - 100 // range [-100, 100]
            })
            .collect()
    }

    fn make_positive_vec(len: usize, seed: u64) -> Vec<i16> {
        let mut state = if seed == 0 { 1u64 } else { seed };
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % 128) as i16 // range [0, 127] (post-ClippedReLU)
            })
            .collect()
    }

    #[test]
    fn test_vec_add_matches_scalar() {
        for &len in &[16, 32, 512, 1024, 1027] {
            let w = make_test_vec(len, 1);
            let mut acc_scalar = make_test_vec(len, 2);
            let mut acc_dispatch = acc_scalar.clone();
            scalar::vec_add_i16(&mut acc_scalar, &w);
            vec_add_i16(&mut acc_dispatch, &w);
            assert_eq!(
                acc_scalar, acc_dispatch,
                "vec_add_i16 mismatch at len={len}"
            );
        }
    }

    #[test]
    fn test_vec_sub_matches_scalar() {
        for &len in &[16, 32, 512, 1024, 1027] {
            let w = make_test_vec(len, 3);
            let mut acc_scalar = make_test_vec(len, 4);
            let mut acc_dispatch = acc_scalar.clone();
            scalar::vec_sub_i16(&mut acc_scalar, &w);
            vec_sub_i16(&mut acc_dispatch, &w);
            assert_eq!(
                acc_scalar, acc_dispatch,
                "vec_sub_i16 mismatch at len={len}"
            );
        }
    }

    #[test]
    fn test_clipped_relu_matches_scalar() {
        for &len in &[16, 512, 1024, 1027] {
            let inp = make_test_vec(len, 5);
            let mut out_scalar = vec![0i16; len];
            let mut out_dispatch = vec![0i16; len];
            scalar::vec_clipped_relu(&mut out_scalar, &inp);
            vec_clipped_relu(&mut out_dispatch, &inp);
            assert_eq!(
                out_scalar, out_dispatch,
                "clipped_relu mismatch at len={len}"
            );
        }
    }

    #[test]
    fn test_dot_i16_i32_matches_scalar() {
        for &len in &[16, 512, 1024, 1027] {
            let a = make_positive_vec(len, 6);
            let b = make_test_vec(len, 7);
            let expected = scalar::dot_i16_i32(&a, &b);
            let actual = dot_i16_i32(&a, &b);
            assert_eq!(expected, actual, "dot_i16_i32 mismatch at len={len}");
        }
    }

    #[test]
    fn test_dot_screlu_i64_matches_scalar() {
        for &len in &[16, 512, 1024, 1027] {
            let a = make_positive_vec(len, 8);
            let b = make_test_vec(len, 9);
            let expected = scalar::dot_screlu_i64(&a, &b);
            let actual = dot_screlu_i64(&a, &b);
            assert_eq!(expected, actual, "dot_screlu_i64 mismatch at len={len}");
        }
    }

    #[test]
    fn test_saturation_boundary() {
        let mut acc = vec![i16::MAX - 1, i16::MIN + 1];
        let w = vec![10, -10];
        vec_add_i16(&mut acc, &w);
        assert_eq!(acc[0], i16::MAX); // saturated
        assert_eq!(acc[1], i16::MIN); // saturated
    }
}
