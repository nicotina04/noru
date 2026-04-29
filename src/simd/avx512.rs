/// AVX-512 SIMD implementations (x86_64, 512-bit = 32 × i16).
///
/// Requires `avx512f` (foundation) + `avx512bw` (byte/word ops — saturating
/// add/sub on i16, madd, min/max). Both are available on Skylake-X (2017+)
/// and all subsequent server / HEDT / Sapphire Rapids parts; unavailable on
/// most consumer Alder/Raptor Lake (P-cores nominally support, fused off).
/// Caller must check via `is_x86_feature_detected!` before invoking.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const I16_PER_VEC: usize = 32;

/// Saturating i16 vector addition.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn vec_add_i16(acc: &mut [i16], w: &[i16]) {
    let len = acc.len();
    let chunks = len / I16_PER_VEC;

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let a = _mm512_loadu_si512(acc.as_ptr().add(off) as *const __m512i);
        let b = _mm512_loadu_si512(w.as_ptr().add(off) as *const __m512i);
        let sum = _mm512_adds_epi16(a, b);
        _mm512_storeu_si512(acc.as_mut_ptr().add(off) as *mut __m512i, sum);
    }

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        acc[i] = acc[i].saturating_add(w[i]);
    }
}

/// Saturating i16 vector subtraction.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn vec_sub_i16(acc: &mut [i16], w: &[i16]) {
    let len = acc.len();
    let chunks = len / I16_PER_VEC;

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let a = _mm512_loadu_si512(acc.as_ptr().add(off) as *const __m512i);
        let b = _mm512_loadu_si512(w.as_ptr().add(off) as *const __m512i);
        let diff = _mm512_subs_epi16(a, b);
        _mm512_storeu_si512(acc.as_mut_ptr().add(off) as *mut __m512i, diff);
    }

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        acc[i] = acc[i].saturating_sub(w[i]);
    }
}

/// ClippedReLU: `out[i] = clamp(inp[i], 0, 127)`
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn vec_clipped_relu(out: &mut [i16], inp: &[i16]) {
    let len = inp.len();
    let chunks = len / I16_PER_VEC;
    let zero = _mm512_setzero_si512();
    let max127 = _mm512_set1_epi16(127);

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let v = _mm512_loadu_si512(inp.as_ptr().add(off) as *const __m512i);
        let clamped = _mm512_min_epi16(_mm512_max_epi16(v, zero), max127);
        _mm512_storeu_si512(out.as_mut_ptr().add(off) as *mut __m512i, clamped);
    }

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        out[i] = inp[i].max(0).min(127);
    }
}

/// i16 dot product → i32 using `_mm512_madd_epi16` + `_mm512_reduce_add_epi32`.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn dot_i16_i32(a: &[i16], b: &[i16]) -> i32 {
    let len = a.len();
    let chunks = len / I16_PER_VEC;
    let mut acc = _mm512_setzero_si512();

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let va = _mm512_loadu_si512(a.as_ptr().add(off) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(off) as *const __m512i);
        // madd: 32 i16 pairs → 16 i32 (sum of adjacent products)
        let prod = _mm512_madd_epi16(va, vb);
        acc = _mm512_add_epi32(acc, prod);
    }

    let mut result = _mm512_reduce_add_epi32(acc);

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        result += a[i] as i32 * b[i] as i32;
    }
    result
}

/// SCReLU squared dot product → i64.
/// Computes `sum(a[i]² × b[i])` with i64 accumulation.
#[target_feature(enable = "avx512f,avx512bw")]
pub unsafe fn dot_screlu_i64(a: &[i16], b: &[i16]) -> i64 {
    let len = a.len();
    let chunks = len / I16_PER_VEC;
    let mut acc_lo = _mm512_setzero_si512(); // 8 × i64
    let mut acc_hi = _mm512_setzero_si512(); // 8 × i64

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let va = _mm512_loadu_si512(a.as_ptr().add(off) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(off) as *const __m512i);

        // a is post-CReLU [0, 127], so a*a fits in i16 (127² = 16129 < 32767).
        let sq = _mm512_mullo_epi16(va, va);

        // madd(sq, b): 32 i16 pairs → 16 i32 (pair sums of products)
        let prod32 = _mm512_madd_epi16(sq, vb);

        // Sign-extend 16 × i32 → 2 groups of 8 × i64
        let lo256 = _mm512_castsi512_si256(prod32);
        let hi256 = _mm512_extracti64x4_epi64::<1>(prod32);
        let lo_64 = _mm512_cvtepi32_epi64(lo256);
        let hi_64 = _mm512_cvtepi32_epi64(hi256);

        acc_lo = _mm512_add_epi64(acc_lo, lo_64);
        acc_hi = _mm512_add_epi64(acc_hi, hi_64);
    }

    // Horizontal sum of 16 × i64 (combined = 8 × i64).
    let combined = _mm512_add_epi64(acc_lo, acc_hi);
    let mut result = _mm512_reduce_add_epi64(combined);

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        result += a[i] as i64 * a[i] as i64 * b[i] as i64;
    }
    result
}
