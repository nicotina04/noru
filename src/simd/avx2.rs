/// AVX2 SIMD implementations (x86_64, 256-bit = 16 × i16).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const I16_PER_VEC: usize = 16;

/// Saturating i16 vector addition.
#[target_feature(enable = "avx2")]
pub unsafe fn vec_add_i16(acc: &mut [i16], w: &[i16]) {
    let len = acc.len();
    let chunks = len / I16_PER_VEC;

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let a = _mm256_loadu_si256(acc.as_ptr().add(off) as *const __m256i);
        let b = _mm256_loadu_si256(w.as_ptr().add(off) as *const __m256i);
        let sum = _mm256_adds_epi16(a, b);
        _mm256_storeu_si256(acc.as_mut_ptr().add(off) as *mut __m256i, sum);
    }

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        acc[i] = acc[i].saturating_add(w[i]);
    }
}

/// Saturating i16 vector subtraction.
#[target_feature(enable = "avx2")]
pub unsafe fn vec_sub_i16(acc: &mut [i16], w: &[i16]) {
    let len = acc.len();
    let chunks = len / I16_PER_VEC;

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let a = _mm256_loadu_si256(acc.as_ptr().add(off) as *const __m256i);
        let b = _mm256_loadu_si256(w.as_ptr().add(off) as *const __m256i);
        let diff = _mm256_subs_epi16(a, b);
        _mm256_storeu_si256(acc.as_mut_ptr().add(off) as *mut __m256i, diff);
    }

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        acc[i] = acc[i].saturating_sub(w[i]);
    }
}

/// ClippedReLU: `out[i] = clamp(inp[i], 0, 127)`
#[target_feature(enable = "avx2")]
pub unsafe fn vec_clipped_relu(out: &mut [i16], inp: &[i16]) {
    let len = inp.len();
    let chunks = len / I16_PER_VEC;
    let zero = _mm256_setzero_si256();
    let max127 = _mm256_set1_epi16(127);

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let v = _mm256_loadu_si256(inp.as_ptr().add(off) as *const __m256i);
        let clamped = _mm256_min_epi16(_mm256_max_epi16(v, zero), max127);
        _mm256_storeu_si256(out.as_mut_ptr().add(off) as *mut __m256i, clamped);
    }

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        out[i] = inp[i].max(0).min(127);
    }
}

/// Horizontal sum of 8 × i32 in a __m256i → single i32.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    let hi128 = _mm256_extracti128_si256::<1>(v);
    let lo128 = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo128, hi128);
    let shuf1 = _mm_shuffle_epi32::<0b_00_01_10_11>(sum128);
    let sum64 = _mm_add_epi32(sum128, shuf1);
    let shuf2 = _mm_shufflelo_epi16::<0b_01_00_11_10>(sum64);
    let sum32 = _mm_add_epi32(sum64, shuf2);
    _mm_cvtsi128_si32(sum32)
}

/// i16 dot product → i32 using _mm256_madd_epi16.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_i16_i32(a: &[i16], b: &[i16]) -> i32 {
    let len = a.len();
    let chunks = len / I16_PER_VEC;
    let mut acc = _mm256_setzero_si256();

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let va = _mm256_loadu_si256(a.as_ptr().add(off) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(off) as *const __m256i);
        // madd: multiply 16 pairs of i16, sum adjacent pairs → 8 × i32
        let prod = _mm256_madd_epi16(va, vb);
        acc = _mm256_add_epi32(acc, prod);
    }

    let mut result = hsum_i32(acc);

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        result += a[i] as i32 * b[i] as i32;
    }
    result
}

/// SCReLU squared dot product → i64.
/// Computes `sum(a[i]² × b[i])` with i64 accumulation.
#[target_feature(enable = "avx2")]
pub unsafe fn dot_screlu_i64(a: &[i16], b: &[i16]) -> i64 {
    let len = a.len();
    let chunks = len / I16_PER_VEC;
    let mut acc_lo = _mm256_setzero_si256(); // 4 × i64
    let mut acc_hi = _mm256_setzero_si256(); // 4 × i64

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let va = _mm256_loadu_si256(a.as_ptr().add(off) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(off) as *const __m256i);

        // squared = a * a (i16, safe for [0, 127]: 127² = 16129 fits i16)
        let sq = _mm256_mullo_epi16(va, va);

        // madd(sq, b): pairs of (sq[i]*b[i]) summed → 8 × i32
        let prod32 = _mm256_madd_epi16(sq, vb);

        // Sign-extend 8 × i32 → 2 groups of 4 × i64
        let lo128 = _mm256_castsi256_si128(prod32);
        let hi128 = _mm256_extracti128_si256::<1>(prod32);
        let lo_64 = _mm256_cvtepi32_epi64(lo128);
        let hi_64 = _mm256_cvtepi32_epi64(hi128);

        acc_lo = _mm256_add_epi64(acc_lo, lo_64);
        acc_hi = _mm256_add_epi64(acc_hi, hi_64);
    }

    // Horizontal sum of 8 × i64
    let combined = _mm256_add_epi64(acc_lo, acc_hi); // 4 × i64
    let mut buf = [0i64; 4];
    _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, combined);
    let mut result = buf[0] + buf[1] + buf[2] + buf[3];

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        result += a[i] as i64 * a[i] as i64 * b[i] as i64;
    }
    result
}
