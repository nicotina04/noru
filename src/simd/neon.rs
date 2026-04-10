/// NEON SIMD implementations (aarch64, 128-bit = 8 × i16).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const I16_PER_VEC: usize = 8;

/// Saturating i16 vector addition.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_add_i16(acc: &mut [i16], w: &[i16]) {
    let len = acc.len();
    let chunks = len / I16_PER_VEC;

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let a = vld1q_s16(acc.as_ptr().add(off));
        let b = vld1q_s16(w.as_ptr().add(off));
        let sum = vqaddq_s16(a, b);
        vst1q_s16(acc.as_mut_ptr().add(off), sum);
    }

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        acc[i] = acc[i].saturating_add(w[i]);
    }
}

/// Saturating i16 vector subtraction.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_sub_i16(acc: &mut [i16], w: &[i16]) {
    let len = acc.len();
    let chunks = len / I16_PER_VEC;

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let a = vld1q_s16(acc.as_ptr().add(off));
        let b = vld1q_s16(w.as_ptr().add(off));
        let diff = vqsubq_s16(a, b);
        vst1q_s16(acc.as_mut_ptr().add(off), diff);
    }

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        acc[i] = acc[i].saturating_sub(w[i]);
    }
}

/// ClippedReLU: `out[i] = clamp(inp[i], 0, 127)`
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn vec_clipped_relu(out: &mut [i16], inp: &[i16]) {
    let len = inp.len();
    let chunks = len / I16_PER_VEC;
    let zero = vdupq_n_s16(0);
    let max127 = vdupq_n_s16(127);

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let v = vld1q_s16(inp.as_ptr().add(off));
        let clamped = vminq_s16(vmaxq_s16(v, zero), max127);
        vst1q_s16(out.as_mut_ptr().add(off), clamped);
    }

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        out[i] = inp[i].max(0).min(127);
    }
}

/// i16 dot product → i32 using multiply-accumulate.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_i16_i32(a: &[i16], b: &[i16]) -> i32 {
    let len = a.len();
    let chunks = len / I16_PER_VEC;
    let mut acc = vdupq_n_s32(0);

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let va = vld1q_s16(a.as_ptr().add(off));
        let vb = vld1q_s16(b.as_ptr().add(off));
        // Low 4 elements: i16 × i16 → i32, accumulate
        acc = vmlal_s16(acc, vget_low_s16(va), vget_low_s16(vb));
        // High 4 elements
        acc = vmlal_high_s16(acc, va, vb);
    }

    let mut result = vaddvq_s32(acc);

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        result += a[i] as i32 * b[i] as i32;
    }
    result
}

/// SCReLU squared dot product → i64.
/// Computes `sum(a[i]² × b[i])` with i64 accumulation.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_screlu_i64(a: &[i16], b: &[i16]) -> i64 {
    let len = a.len();
    let chunks = len / I16_PER_VEC;
    let mut acc64 = vdupq_n_s64(0);

    for c in 0..chunks {
        let off = c * I16_PER_VEC;
        let va = vld1q_s16(a.as_ptr().add(off));
        let vb = vld1q_s16(b.as_ptr().add(off));

        // Square: a * a → i32 (low 4 and high 4)
        let sq_lo = vmull_s16(vget_low_s16(va), vget_low_s16(va));
        let sq_hi = vmull_high_s16(va, va);

        // Widen weights to i32
        let wb_lo = vmovl_s16(vget_low_s16(vb));
        let wb_hi = vmovl_high_s16(vb);

        // sq * w in i32
        let prod_lo = vmulq_s32(sq_lo, wb_lo); // 4 × i32
        let prod_hi = vmulq_s32(sq_hi, wb_hi); // 4 × i32

        // Accumulate into i64 (pairwise widening add)
        acc64 = vaddq_s64(acc64, vpaddlq_s32(prod_lo));
        acc64 = vaddq_s64(acc64, vpaddlq_s32(prod_hi));
    }

    let mut result = vaddvq_s64(acc64);

    let tail = chunks * I16_PER_VEC;
    for i in tail..len {
        result += a[i] as i64 * a[i] as i64 * b[i] as i64;
    }
    result
}
