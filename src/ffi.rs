//! C ABI bindings for use from engines that embed noru (Unity, Godot, C#, C++).
//!
//! All functions are `extern "C"` and wrap their bodies in `catch_unwind` so
//! panics do not unwind across the FFI boundary. Errors are reported via
//! return codes; the most recent error message on the calling thread can be
//! retrieved with [`noru_last_error`].
//!
//! # Memory ownership
//!
//! Opaque handles (`NoruTrainer`, `NoruWeights`, `NoruAccumulator`) are created
//! by the library and must be released with their matching `*_free` function.
//! Byte buffers returned via `out_ptr`/`out_len` pairs must be released with
//! [`noru_free_bytes`]. All input slice pointers (features, byte buffers) are
//! read only and remain owned by the caller.
//!
//! # Pointer validity
//!
//! `noru_last_error` returns a pointer valid only until the next `noru_*` call
//! on the same thread — copy the string immediately on the caller side.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::slice;

use crate::config::{reclaim_leaked_hidden_sizes, Activation, OwnedNnueConfig};
use crate::network::{forward as nnue_forward, Accumulator, FeatureDelta, NnueWeights};
use crate::trainer::{
    AdamState, ForwardResult, Gradients, SimpleRng, TrainableWeights, TrainingSample,
};

// ---------- Error codes ----------

pub const NORU_OK: i32 = 0;
pub const NORU_ERR_NULL_PTR: i32 = -1;
pub const NORU_ERR_INVALID_ARG: i32 = -2;
pub const NORU_ERR_PANIC: i32 = -3;
pub const NORU_ERR_IO: i32 = -4;
pub const NORU_ERR_STATE: i32 = -5;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

fn set_last_error(msg: impl Into<String>) {
    let s = msg.into();
    let c = CString::new(s).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(c));
}

fn clear_last_error() {
    LAST_ERROR.with(|cell| *cell.borrow_mut() = None);
}

fn guard<F: FnOnce() -> i32>(f: F) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(code) => code,
        Err(_) => {
            set_last_error("panic in noru FFI call");
            NORU_ERR_PANIC
        }
    }
}

/// Returns a NUL-terminated UTF-8 error message for the most recent failing
/// call on this thread, or `NULL` if there has been no recent error.
///
/// The returned pointer is valid only until the next `noru_*` call on the
/// same thread. Copy it immediately if you need to retain it.
#[no_mangle]
pub extern "C" fn noru_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| match &*cell.borrow() {
        Some(s) => s.as_ptr(),
        None => ptr::null(),
    })
}

// ---------- Opaque handles ----------

/// Training handle: FP32 weights + Adam optimizer state + gradient buffer
/// plus cached forward/sample state for step-wise training.
pub struct NoruTrainer {
    weights: TrainableWeights,
    adam: AdamState,
    grad: Gradients,
    last_sample: Option<TrainingSample>,
    last_fwd: Option<ForwardResult>,
}

impl Drop for NoruTrainer {
    fn drop(&mut self) {
        unsafe { reclaim_leaked_hidden_sizes(self.weights.config.hidden_sizes) };
    }
}

/// Inference weights handle (i16 quantized).
pub struct NoruWeights {
    weights: NnueWeights,
}

impl Drop for NoruWeights {
    fn drop(&mut self) {
        unsafe { reclaim_leaked_hidden_sizes(self.weights.config.hidden_sizes) };
    }
}

/// Accumulator handle for incremental inference.
pub struct NoruAccumulator {
    acc: Accumulator,
}

// ---------- Helpers ----------

unsafe fn slice_from_raw_usize(ptr: *const u32, len: usize) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    let s = slice::from_raw_parts(ptr, len);
    s.iter().map(|&v| v as usize).collect()
}

fn activation_from_u8(v: u8) -> Result<Activation, &'static str> {
    match v {
        0 => Ok(Activation::CReLU),
        1 => Ok(Activation::SCReLU),
        _ => Err("unknown activation type"),
    }
}

fn build_feature_delta(added: &[usize], removed: &[usize]) -> Result<FeatureDelta, &'static str> {
    FeatureDelta::from_slices(added, removed)
        .map_err(|_| "feature delta exceeds 32 entries per side")
}

// ---------- Trainer lifecycle ----------

/// Create a new trainable network with Kaiming-initialized weights.
///
/// `hidden_sizes_ptr` must point to an array of `hidden_sizes_len` usize values.
/// `activation`: 0 = CReLU, 1 = SCReLU.
///
/// On success writes a non-null handle to `out_handle` and returns `NORU_OK`.
#[no_mangle]
pub unsafe extern "C" fn noru_trainer_new(
    feature_size: usize,
    accumulator_size: usize,
    hidden_sizes_ptr: *const usize,
    hidden_sizes_len: usize,
    activation: u8,
    seed: u64,
    out_handle: *mut *mut NoruTrainer,
) -> i32 {
    guard(|| {
        clear_last_error();
        if out_handle.is_null() {
            set_last_error("out_handle is null");
            return NORU_ERR_NULL_PTR;
        }
        if hidden_sizes_ptr.is_null() || hidden_sizes_len == 0 {
            set_last_error("hidden_sizes is empty or null");
            return NORU_ERR_INVALID_ARG;
        }
        if feature_size == 0 || accumulator_size == 0 {
            set_last_error("feature_size and accumulator_size must be non-zero");
            return NORU_ERR_INVALID_ARG;
        }
        let act = match activation_from_u8(activation) {
            Ok(a) => a,
            Err(e) => {
                set_last_error(e);
                return NORU_ERR_INVALID_ARG;
            }
        };
        let hidden = slice::from_raw_parts(hidden_sizes_ptr, hidden_sizes_len).to_vec();
        let owned = OwnedNnueConfig::new(feature_size, accumulator_size, hidden, act);
        let config = owned.leak();

        let mut rng = SimpleRng::new(seed);
        let weights = TrainableWeights::init_random(config, &mut rng);
        let adam = AdamState::new(config);
        let grad = Gradients::new(config);

        let handle = Box::new(NoruTrainer {
            weights,
            adam,
            grad,
            last_sample: None,
            last_fwd: None,
        });
        *out_handle = Box::into_raw(handle);
        NORU_OK
    })
}

/// Release a trainer handle. Safe to pass a null pointer (no-op).
#[no_mangle]
pub unsafe extern "C" fn noru_trainer_free(handle: *mut NoruTrainer) {
    if handle.is_null() {
        return;
    }
    drop(Box::from_raw(handle));
}

// ---------- Training step ----------

/// Run a forward pass and cache the intermediate state for a subsequent
/// backward call. Writes the pre-sigmoid output (aka "eval") to `out_eval`.
#[no_mangle]
pub unsafe extern "C" fn noru_trainer_forward(
    handle: *mut NoruTrainer,
    stm_ptr: *const u32,
    stm_len: usize,
    nstm_ptr: *const u32,
    nstm_len: usize,
    out_eval: *mut f32,
) -> i32 {
    guard(|| {
        clear_last_error();
        if handle.is_null() {
            set_last_error("handle is null");
            return NORU_ERR_NULL_PTR;
        }
        let trainer = &mut *handle;
        let stm = slice_from_raw_usize(stm_ptr, stm_len);
        let nstm = slice_from_raw_usize(nstm_ptr, nstm_len);

        let fwd = trainer.weights.forward(&stm, &nstm);
        if !out_eval.is_null() {
            *out_eval = fwd.output;
        }
        trainer.last_sample = Some(TrainingSample {
            stm_features: stm,
            nstm_features: nstm,
            target: 0.0,
        });
        trainer.last_fwd = Some(fwd);
        NORU_OK
    })
}

/// Backward pass with binary cross-entropy loss on `sigmoid(output)`.
///
/// Use this when the target is in the `[0, 1]` range (e.g. win probability
/// or a normalized evaluation). Gradient is `sigmoid(output) - target`.
/// Gradients accumulate into the internal buffer — call
/// [`noru_trainer_zero_grad`] between mini-batches.
#[no_mangle]
pub unsafe extern "C" fn noru_trainer_backward_bce(handle: *mut NoruTrainer, target: f32) -> i32 {
    guard(|| {
        clear_last_error();
        if handle.is_null() {
            set_last_error("handle is null");
            return NORU_ERR_NULL_PTR;
        }
        let trainer = &mut *handle;
        let sample = match trainer.last_sample.as_mut() {
            Some(s) => s,
            None => {
                set_last_error("backward_bce called before forward");
                return NORU_ERR_STATE;
            }
        };
        sample.target = target;
        let fwd = match trainer.last_fwd.as_ref() {
            Some(f) => f,
            None => {
                set_last_error("backward_bce called without forward result");
                return NORU_ERR_STATE;
            }
        };
        trainer.weights.backward_bce(sample, fwd, &mut trainer.grad);
        NORU_OK
    })
}

/// Backward pass with raw MSE: `(output - target)^2` against the raw
/// pre-sigmoid output.
///
/// Use this when the target is an unbounded real-valued eval
/// (e.g. centipawn score). **The target is not in `[0, 1]`.** For
/// probability-like targets use [`noru_trainer_backward_bce`] instead.
#[no_mangle]
pub unsafe extern "C" fn noru_trainer_backward_raw_mse(
    handle: *mut NoruTrainer,
    target: f32,
) -> i32 {
    guard(|| {
        clear_last_error();
        if handle.is_null() {
            set_last_error("handle is null");
            return NORU_ERR_NULL_PTR;
        }
        let trainer = &mut *handle;
        let sample = match trainer.last_sample.as_mut() {
            Some(s) => s,
            None => {
                set_last_error("backward_raw_mse called before forward");
                return NORU_ERR_STATE;
            }
        };
        sample.target = target;
        let fwd = match trainer.last_fwd.as_ref() {
            Some(f) => f,
            None => {
                set_last_error("backward_raw_mse called without forward result");
                return NORU_ERR_STATE;
            }
        };
        trainer
            .weights
            .backward_raw_mse(sample, fwd, &mut trainer.grad);
        NORU_OK
    })
}

/// Zero the internal gradient accumulator.
#[no_mangle]
pub unsafe extern "C" fn noru_trainer_zero_grad(handle: *mut NoruTrainer) -> i32 {
    guard(|| {
        clear_last_error();
        if handle.is_null() {
            set_last_error("handle is null");
            return NORU_ERR_NULL_PTR;
        }
        let trainer = &mut *handle;
        trainer.grad.zero();
        NORU_OK
    })
}

/// Apply one Adam optimizer step using the current gradient accumulator
/// scaled by `1.0 / batch_size`.
#[no_mangle]
pub unsafe extern "C" fn noru_trainer_adam_step(
    handle: *mut NoruTrainer,
    lr: f32,
    batch_size: f32,
) -> i32 {
    guard(|| {
        clear_last_error();
        if handle.is_null() {
            set_last_error("handle is null");
            return NORU_ERR_NULL_PTR;
        }
        if !(batch_size > 0.0) {
            set_last_error("batch_size must be positive");
            return NORU_ERR_INVALID_ARG;
        }
        let trainer = &mut *handle;
        trainer
            .weights
            .adam_update(&trainer.grad, &mut trainer.adam, lr, batch_size);
        NORU_OK
    })
}

// ---------- Serialization ----------

unsafe fn vec_into_out_buf(buf: Vec<u8>, out_ptr: *mut *mut u8, out_len: *mut usize) -> i32 {
    if out_ptr.is_null() || out_len.is_null() {
        set_last_error("output pointer(s) are null");
        return NORU_ERR_NULL_PTR;
    }
    let mut boxed = buf.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    let len = boxed.len();
    std::mem::forget(boxed);
    *out_ptr = ptr;
    *out_len = len;
    NORU_OK
}

/// Serialize the trainer's FP32 weights to a newly allocated byte buffer.
///
/// On success writes a pointer + length. Caller must release the buffer via
/// [`noru_free_bytes`].
#[no_mangle]
pub unsafe extern "C" fn noru_trainer_save_fp32(
    handle: *mut NoruTrainer,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        clear_last_error();
        if handle.is_null() {
            set_last_error("handle is null");
            return NORU_ERR_NULL_PTR;
        }
        let trainer = &*handle;
        let bytes = trainer.weights.save_to_bytes();
        vec_into_out_buf(bytes, out_ptr, out_len)
    })
}

/// Load an FP32 checkpoint previously produced by [`noru_trainer_save_fp32`].
/// Adam optimizer state is reinitialized to zero.
#[no_mangle]
pub unsafe extern "C" fn noru_trainer_load_fp32(
    data: *const u8,
    len: usize,
    out_handle: *mut *mut NoruTrainer,
) -> i32 {
    guard(|| {
        clear_last_error();
        if data.is_null() || out_handle.is_null() {
            set_last_error("null pointer argument");
            return NORU_ERR_NULL_PTR;
        }
        let slice = slice::from_raw_parts(data, len);
        let weights = match TrainableWeights::load_from_bytes(slice) {
            Ok(w) => w,
            Err(e) => {
                set_last_error(e);
                return NORU_ERR_IO;
            }
        };
        let adam = AdamState::new(weights.config);
        let grad = Gradients::new(weights.config);
        let handle = Box::new(NoruTrainer {
            weights,
            adam,
            grad,
            last_sample: None,
            last_fwd: None,
        });
        *out_handle = Box::into_raw(handle);
        NORU_OK
    })
}

/// Quantize the trainer's FP32 weights to i16 inference weights.
#[no_mangle]
pub unsafe extern "C" fn noru_trainer_quantize(
    handle: *mut NoruTrainer,
    out_weights: *mut *mut NoruWeights,
) -> i32 {
    guard(|| {
        clear_last_error();
        if handle.is_null() || out_weights.is_null() {
            set_last_error("null pointer argument");
            return NORU_ERR_NULL_PTR;
        }
        let trainer = &*handle;
        let quantized = trainer.weights.quantize();

        // `quantize()` reuses the trainer's leaked NnueConfig by value, so the
        // resulting NnueWeights shares the same `&'static hidden_sizes`. The
        // trainer handle still owns that allocation. To give the weights
        // handle its own allocation (so it can be freed independently), clone
        // the hidden_sizes into a fresh leaked slice.
        let owned = OwnedNnueConfig::new(
            quantized.config.feature_size,
            quantized.config.accumulator_size,
            quantized.config.hidden_sizes.to_vec(),
            quantized.config.activation,
        );
        let fresh_config = owned.leak();
        let mut weights = quantized;
        weights.config = fresh_config;

        let boxed = Box::new(NoruWeights { weights });
        *out_weights = Box::into_raw(boxed);
        NORU_OK
    })
}

// ---------- Inference weights ----------

/// Load inference weights from a NORU v2 binary blob.
#[no_mangle]
pub unsafe extern "C" fn noru_weights_load(
    data: *const u8,
    len: usize,
    out_weights: *mut *mut NoruWeights,
) -> i32 {
    guard(|| {
        clear_last_error();
        if data.is_null() || out_weights.is_null() {
            set_last_error("null pointer argument");
            return NORU_ERR_NULL_PTR;
        }
        let slice = slice::from_raw_parts(data, len);
        let weights = match NnueWeights::load_from_bytes(slice, None) {
            Ok(w) => w,
            Err(e) => {
                set_last_error(e);
                return NORU_ERR_IO;
            }
        };
        let boxed = Box::new(NoruWeights { weights });
        *out_weights = Box::into_raw(boxed);
        NORU_OK
    })
}

/// Serialize inference weights to a newly allocated NORU v2 binary blob.
#[no_mangle]
pub unsafe extern "C" fn noru_weights_save(
    handle: *mut NoruWeights,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    guard(|| {
        clear_last_error();
        if handle.is_null() {
            set_last_error("handle is null");
            return NORU_ERR_NULL_PTR;
        }
        let w = &*handle;
        let bytes = w.weights.save_to_bytes();
        vec_into_out_buf(bytes, out_ptr, out_len)
    })
}

/// Release an inference weights handle. Safe to pass a null pointer.
#[no_mangle]
pub unsafe extern "C" fn noru_weights_free(handle: *mut NoruWeights) {
    if handle.is_null() {
        return;
    }
    drop(Box::from_raw(handle));
}

/// Release a byte buffer previously returned via `out_ptr`/`out_len`.
#[no_mangle]
pub unsafe extern "C" fn noru_free_bytes(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let slice = slice::from_raw_parts_mut(ptr, len);
    drop(Box::from_raw(slice as *mut [u8]));
}

// ---------- Accumulator / inference forward ----------

/// Create an accumulator initialized from the given weights' feature bias.
#[no_mangle]
pub unsafe extern "C" fn noru_accumulator_new(
    weights: *mut NoruWeights,
    out_acc: *mut *mut NoruAccumulator,
) -> i32 {
    guard(|| {
        clear_last_error();
        if weights.is_null() || out_acc.is_null() {
            set_last_error("null pointer argument");
            return NORU_ERR_NULL_PTR;
        }
        let w = &*weights;
        let acc = Accumulator::new(&w.weights.feature_bias);
        let boxed = Box::new(NoruAccumulator { acc });
        *out_acc = Box::into_raw(boxed);
        NORU_OK
    })
}

/// Release an accumulator handle. Safe to pass a null pointer.
#[no_mangle]
pub unsafe extern "C" fn noru_accumulator_free(handle: *mut NoruAccumulator) {
    if handle.is_null() {
        return;
    }
    drop(Box::from_raw(handle));
}

/// Fully recompute the accumulator from STM/NSTM feature index lists.
#[no_mangle]
pub unsafe extern "C" fn noru_accumulator_refresh(
    acc: *mut NoruAccumulator,
    weights: *mut NoruWeights,
    stm_ptr: *const u32,
    stm_len: usize,
    nstm_ptr: *const u32,
    nstm_len: usize,
) -> i32 {
    guard(|| {
        clear_last_error();
        if acc.is_null() || weights.is_null() {
            set_last_error("null pointer argument");
            return NORU_ERR_NULL_PTR;
        }
        let acc_ref = &mut *acc;
        let w = &*weights;
        let stm = slice_from_raw_usize(stm_ptr, stm_len);
        let nstm = slice_from_raw_usize(nstm_ptr, nstm_len);
        acc_ref.acc.refresh(&w.weights, &stm, &nstm);
        NORU_OK
    })
}

/// Apply an incremental add/remove delta to the accumulator.
/// Up to 32 adds and 32 removes per side are supported per call.
#[no_mangle]
pub unsafe extern "C" fn noru_accumulator_update(
    acc: *mut NoruAccumulator,
    weights: *mut NoruWeights,
    stm_added_ptr: *const u32,
    stm_added_len: usize,
    stm_removed_ptr: *const u32,
    stm_removed_len: usize,
    nstm_added_ptr: *const u32,
    nstm_added_len: usize,
    nstm_removed_ptr: *const u32,
    nstm_removed_len: usize,
) -> i32 {
    guard(|| {
        clear_last_error();
        if acc.is_null() || weights.is_null() {
            set_last_error("null pointer argument");
            return NORU_ERR_NULL_PTR;
        }
        let acc_ref = &mut *acc;
        let w = &*weights;
        let stm_added = slice_from_raw_usize(stm_added_ptr, stm_added_len);
        let stm_removed = slice_from_raw_usize(stm_removed_ptr, stm_removed_len);
        let nstm_added = slice_from_raw_usize(nstm_added_ptr, nstm_added_len);
        let nstm_removed = slice_from_raw_usize(nstm_removed_ptr, nstm_removed_len);
        let stm_delta = match build_feature_delta(&stm_added, &stm_removed) {
            Ok(d) => d,
            Err(e) => {
                set_last_error(e);
                return NORU_ERR_INVALID_ARG;
            }
        };
        let nstm_delta = match build_feature_delta(&nstm_added, &nstm_removed) {
            Ok(d) => d,
            Err(e) => {
                set_last_error(e);
                return NORU_ERR_INVALID_ARG;
            }
        };
        acc_ref
            .acc
            .update_incremental(&w.weights, &stm_delta, &nstm_delta);
        NORU_OK
    })
}

/// Undo an incremental update previously applied to the accumulator.
///
/// Pass the exact same `added` / `removed` lists that were given to the
/// matching [`noru_accumulator_update`] call. This is the fastest way to
/// pop a branch in a tree search when you do not want to maintain a
/// cloned snapshot of the accumulator.
#[no_mangle]
pub unsafe extern "C" fn noru_accumulator_update_undo(
    acc: *mut NoruAccumulator,
    weights: *mut NoruWeights,
    stm_added_ptr: *const u32,
    stm_added_len: usize,
    stm_removed_ptr: *const u32,
    stm_removed_len: usize,
    nstm_added_ptr: *const u32,
    nstm_added_len: usize,
    nstm_removed_ptr: *const u32,
    nstm_removed_len: usize,
) -> i32 {
    guard(|| {
        clear_last_error();
        if acc.is_null() || weights.is_null() {
            set_last_error("null pointer argument");
            return NORU_ERR_NULL_PTR;
        }
        let acc_ref = &mut *acc;
        let w = &*weights;
        let stm_added = slice_from_raw_usize(stm_added_ptr, stm_added_len);
        let stm_removed = slice_from_raw_usize(stm_removed_ptr, stm_removed_len);
        let nstm_added = slice_from_raw_usize(nstm_added_ptr, nstm_added_len);
        let nstm_removed = slice_from_raw_usize(nstm_removed_ptr, nstm_removed_len);
        let stm_delta = match build_feature_delta(&stm_added, &stm_removed) {
            Ok(d) => d,
            Err(e) => {
                set_last_error(e);
                return NORU_ERR_INVALID_ARG;
            }
        };
        let nstm_delta = match build_feature_delta(&nstm_added, &nstm_removed) {
            Ok(d) => d,
            Err(e) => {
                set_last_error(e);
                return NORU_ERR_INVALID_ARG;
            }
        };
        acc_ref
            .acc
            .update_incremental_undo(&w.weights, &stm_delta, &nstm_delta);
        NORU_OK
    })
}

/// Deep-clone an accumulator into a new independently-owned handle.
///
/// The returned handle is not bound to any weights; it carries only the
/// raw state. Pair it with the same weights pointer you used for the
/// source accumulator when calling forward.
#[no_mangle]
pub unsafe extern "C" fn noru_accumulator_clone(
    acc: *mut NoruAccumulator,
    out_clone: *mut *mut NoruAccumulator,
) -> i32 {
    guard(|| {
        clear_last_error();
        if acc.is_null() || out_clone.is_null() {
            set_last_error("null pointer argument");
            return NORU_ERR_NULL_PTR;
        }
        let src = &*acc;
        let cloned = Box::new(NoruAccumulator {
            acc: src.acc.clone(),
        });
        *out_clone = Box::into_raw(cloned);
        NORU_OK
    })
}

/// Overwrite the state of `dst` with the state of `src`. Both handles must
/// have been created from accumulators of matching topology (same
/// accumulator size). Useful for reusing a single scratch accumulator
/// inside a search loop instead of allocating fresh clones per node.
#[no_mangle]
pub unsafe extern "C" fn noru_accumulator_copy_from(
    dst: *mut NoruAccumulator,
    src: *mut NoruAccumulator,
) -> i32 {
    guard(|| {
        clear_last_error();
        if dst.is_null() || src.is_null() {
            set_last_error("null pointer argument");
            return NORU_ERR_NULL_PTR;
        }
        let d = &mut *dst;
        let s = &*src;
        if d.acc.stm.len() != s.acc.stm.len() || d.acc.nstm.len() != s.acc.nstm.len() {
            set_last_error("accumulator size mismatch");
            return NORU_ERR_INVALID_ARG;
        }
        d.acc.stm.copy_from_slice(&s.acc.stm);
        d.acc.nstm.copy_from_slice(&s.acc.nstm);
        NORU_OK
    })
}

/// Swap the STM and NSTM perspectives of the accumulator.
#[no_mangle]
pub unsafe extern "C" fn noru_accumulator_swap(acc: *mut NoruAccumulator) -> i32 {
    guard(|| {
        clear_last_error();
        if acc.is_null() {
            set_last_error("null pointer argument");
            return NORU_ERR_NULL_PTR;
        }
        let acc_ref = &mut *acc;
        acc_ref.acc.swap();
        NORU_OK
    })
}

/// Full forward pass through the accumulator → hidden layers → output.
/// Writes the scaled integer evaluation to `out_eval`.
#[no_mangle]
pub unsafe extern "C" fn noru_accumulator_forward(
    acc: *mut NoruAccumulator,
    weights: *mut NoruWeights,
    out_eval: *mut i32,
) -> i32 {
    guard(|| {
        clear_last_error();
        if acc.is_null() || weights.is_null() || out_eval.is_null() {
            set_last_error("null pointer argument");
            return NORU_ERR_NULL_PTR;
        }
        let acc_ref = &*acc;
        let w = &*weights;
        *out_eval = nnue_forward(&acc_ref.acc, &w.weights);
        NORU_OK
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_hidden() -> Vec<usize> {
        vec![16, 8]
    }

    #[test]
    fn trainer_roundtrip_via_ffi() {
        unsafe {
            let hidden = small_hidden();
            let mut trainer: *mut NoruTrainer = ptr::null_mut();
            let rc = noru_trainer_new(
                64,
                32,
                hidden.as_ptr(),
                hidden.len(),
                0,
                1234,
                &mut trainer as *mut _,
            );
            assert_eq!(rc, NORU_OK);
            assert!(!trainer.is_null());

            let stm: [u32; 3] = [1, 5, 10];
            let nstm: [u32; 3] = [2, 7, 15];
            let mut eval: f32 = 0.0;
            let rc = noru_trainer_forward(
                trainer,
                stm.as_ptr(),
                stm.len(),
                nstm.as_ptr(),
                nstm.len(),
                &mut eval,
            );
            assert_eq!(rc, NORU_OK);

            assert_eq!(noru_trainer_zero_grad(trainer), NORU_OK);
            assert_eq!(noru_trainer_backward_bce(trainer, 0.5), NORU_OK);
            assert_eq!(noru_trainer_backward_raw_mse(trainer, 0.5), NORU_OK);
            assert_eq!(noru_trainer_adam_step(trainer, 1e-3, 1.0), NORU_OK);

            let mut save_ptr: *mut u8 = ptr::null_mut();
            let mut save_len: usize = 0;
            assert_eq!(
                noru_trainer_save_fp32(trainer, &mut save_ptr, &mut save_len),
                NORU_OK
            );
            assert!(!save_ptr.is_null() && save_len > 0);

            let mut trainer2: *mut NoruTrainer = ptr::null_mut();
            assert_eq!(
                noru_trainer_load_fp32(save_ptr, save_len, &mut trainer2),
                NORU_OK
            );
            assert!(!trainer2.is_null());

            noru_free_bytes(save_ptr, save_len);

            let mut weights_handle: *mut NoruWeights = ptr::null_mut();
            assert_eq!(noru_trainer_quantize(trainer, &mut weights_handle), NORU_OK);

            let mut acc_handle: *mut NoruAccumulator = ptr::null_mut();
            assert_eq!(
                noru_accumulator_new(weights_handle, &mut acc_handle),
                NORU_OK
            );
            assert_eq!(
                noru_accumulator_refresh(
                    acc_handle,
                    weights_handle,
                    stm.as_ptr(),
                    stm.len(),
                    nstm.as_ptr(),
                    nstm.len()
                ),
                NORU_OK
            );
            let mut int_eval: i32 = 0;
            assert_eq!(
                noru_accumulator_forward(acc_handle, weights_handle, &mut int_eval),
                NORU_OK
            );

            // Clone + copy_from + update_undo smoke test.
            let mut cloned_acc: *mut NoruAccumulator = ptr::null_mut();
            assert_eq!(noru_accumulator_clone(acc_handle, &mut cloned_acc), NORU_OK);
            assert!(!cloned_acc.is_null());

            let added: [u32; 1] = [3];
            assert_eq!(
                noru_accumulator_update(
                    acc_handle,
                    weights_handle,
                    added.as_ptr(),
                    added.len(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    0,
                    ptr::null(),
                    0
                ),
                NORU_OK
            );
            assert_eq!(
                noru_accumulator_update_undo(
                    acc_handle,
                    weights_handle,
                    added.as_ptr(),
                    added.len(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    0,
                    ptr::null(),
                    0
                ),
                NORU_OK
            );
            assert_eq!(noru_accumulator_copy_from(acc_handle, cloned_acc), NORU_OK);

            noru_accumulator_free(cloned_acc);
            noru_accumulator_free(acc_handle);
            noru_weights_free(weights_handle);
            noru_trainer_free(trainer2);
            noru_trainer_free(trainer);
        }
    }

    #[test]
    fn null_handle_is_an_error_not_a_crash() {
        unsafe {
            let rc = noru_trainer_zero_grad(ptr::null_mut());
            assert_eq!(rc, NORU_ERR_NULL_PTR);
            assert!(!noru_last_error().is_null());
        }
    }

    #[test]
    fn accumulator_update_rejects_oversized_delta() {
        unsafe {
            let hidden = [8usize];
            let mut trainer: *mut NoruTrainer = ptr::null_mut();
            assert_eq!(
                noru_trainer_new(32, 16, hidden.as_ptr(), hidden.len(), 0, 99, &mut trainer),
                NORU_OK
            );

            let mut weights_handle: *mut NoruWeights = ptr::null_mut();
            assert_eq!(noru_trainer_quantize(trainer, &mut weights_handle), NORU_OK);

            let mut acc_handle: *mut NoruAccumulator = ptr::null_mut();
            assert_eq!(
                noru_accumulator_new(weights_handle, &mut acc_handle),
                NORU_OK
            );

            let oversized: Vec<u32> = (0..=32u32).collect();
            let rc = noru_accumulator_update(
                acc_handle,
                weights_handle,
                oversized.as_ptr(),
                oversized.len(),
                ptr::null(),
                0,
                ptr::null(),
                0,
                ptr::null(),
                0,
            );
            assert_eq!(rc, NORU_ERR_INVALID_ARG);
            assert!(!noru_last_error().is_null());

            noru_accumulator_free(acc_handle);
            noru_weights_free(weights_handle);
            noru_trainer_free(trainer);
        }
    }
}
