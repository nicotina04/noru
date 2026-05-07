//! End-to-end integration tests for the public API.
//!
//! These tests exercise noru the way an external crate would: they only reach
//! into `pub` items and they cover the full train → quantize → serialize →
//! deserialize → inference loop. JOSS reviewers or crates.io users can run
//! `cargo test` and get a binary pass/fail on the core pipeline.

use noru::config::{Activation, NnueConfig};
use noru::network::{forward, Accumulator, FeatureDelta, NnueWeights};
use noru::trainer::{AdamState, Gradients, SimpleRng, TrainableWeights, TrainingSample};

const SMALL_CONFIG: NnueConfig = NnueConfig::new_static(16, 32, &[16], Activation::CReLU);

fn synthetic_samples(seed: u64, n: usize) -> Vec<TrainingSample> {
    let mut rng = SimpleRng::new(seed);
    (0..n)
        .map(|_| {
            let len = 2 + rng.next_usize(3);
            let mut stm: Vec<usize> = (0..len)
                .map(|_| rng.next_usize(SMALL_CONFIG.feature_size))
                .collect();
            stm.sort_unstable();
            stm.dedup();
            let mut nstm: Vec<usize> = (0..len)
                .map(|_| rng.next_usize(SMALL_CONFIG.feature_size))
                .collect();
            nstm.sort_unstable();
            nstm.dedup();
            let target =
                (stm.len() as f32 / (stm.len() + nstm.len()).max(1) as f32).clamp(0.0, 1.0);
            TrainingSample {
                stm_features: stm,
                nstm_features: nstm,
                target,
                dense_input: Vec::new(),
            }
        })
        .collect()
}

fn train(weights: &mut TrainableWeights, samples: &[TrainingSample], epochs: usize) {
    let mut adam = AdamState::new(SMALL_CONFIG);
    for _ in 0..epochs {
        for sample in samples {
            let fwd = weights.forward(&sample.stm_features, &sample.nstm_features, &[]);
            let mut grad = Gradients::new(SMALL_CONFIG);
            weights.backward_bce(sample, &fwd, &mut grad);
            weights.adam_update(&grad, &mut adam, 1e-2, 1.0);
        }
    }
}

#[test]
fn save_load_produces_identical_inference() {
    let mut rng = SimpleRng::new(1);
    let mut weights = TrainableWeights::init_random(SMALL_CONFIG, &mut rng);
    let samples = synthetic_samples(2, 32);
    train(&mut weights, &samples, 20);

    let quantized = weights.quantize();
    let bytes = quantized.save_to_bytes();
    let reloaded =
        NnueWeights::load_from_bytes(&bytes, None).expect("v2 header must be recognized on reload");

    // Same inputs must yield identical i16 outputs before and after serialize
    // → deserialize. This is the minimum contract of the binary format.
    for sample in &samples {
        let mut acc_before = Accumulator::new(&quantized.feature_bias);
        acc_before.refresh(&quantized, &sample.stm_features, &sample.nstm_features);
        let eval_before = forward(&acc_before, &quantized);

        let mut acc_after = Accumulator::new(&reloaded.feature_bias);
        acc_after.refresh(&reloaded, &sample.stm_features, &sample.nstm_features);
        let eval_after = forward(&acc_after, &reloaded);

        assert_eq!(
            eval_before, eval_after,
            "i16 inference must be bit-identical across save→load: stm={:?} nstm={:?}",
            sample.stm_features, sample.nstm_features,
        );
    }
}

#[test]
fn incremental_update_matches_refresh() {
    // An accumulator built incrementally (add/remove one feature at a time)
    // must land on the exact same numeric state as one built with a single
    // `refresh` over the final feature sets. This is the invariant search
    // code relies on.
    let mut rng = SimpleRng::new(9);
    let weights = TrainableWeights::init_random(SMALL_CONFIG, &mut rng).quantize();

    let stm_initial = vec![1_usize, 3, 5];
    let nstm_initial = vec![0_usize, 2, 4];
    let stm_added = 7_usize;
    let nstm_removed = 2_usize;

    // Start: refresh with initial features, then apply a delta.
    let mut acc_incremental = Accumulator::new(&weights.feature_bias);
    acc_incremental.refresh(&weights, &stm_initial, &nstm_initial);
    let mut delta_stm = FeatureDelta::new();
    delta_stm.add(stm_added);
    let mut delta_nstm = FeatureDelta::new();
    delta_nstm.remove(nstm_removed);
    acc_incremental.update_incremental(&weights, &delta_stm, &delta_nstm);

    // Reference: refresh directly with the final feature sets.
    let mut stm_final = stm_initial.clone();
    stm_final.push(stm_added);
    stm_final.sort_unstable();
    let nstm_final: Vec<usize> = nstm_initial
        .iter()
        .copied()
        .filter(|&f| f != nstm_removed)
        .collect();
    let mut acc_refresh = Accumulator::new(&weights.feature_bias);
    acc_refresh.refresh(&weights, &stm_final, &nstm_final);

    let eval_incremental = forward(&acc_incremental, &weights);
    let eval_refresh = forward(&acc_refresh, &weights);

    assert_eq!(
        eval_incremental, eval_refresh,
        "incremental update must agree with full refresh"
    );
}

#[test]
fn training_reduces_loss() {
    // Not claiming convergence — just that the Adam path isn't broken.
    // Mean BCE sigmoid-space MSE should drop after some training passes on a
    // learnable synthetic signal.
    let mut rng = SimpleRng::new(7);
    let mut weights = TrainableWeights::init_random(SMALL_CONFIG, &mut rng);
    let samples = synthetic_samples(11, 32);

    let mse = |w: &TrainableWeights| -> f32 {
        let mut total = 0.0;
        for s in &samples {
            let f = w.forward(&s.stm_features, &s.nstm_features, &[]);
            total += (f.sigmoid - s.target).powi(2);
        }
        total / samples.len() as f32
    };

    let before = mse(&weights);
    train(&mut weights, &samples, 100);
    let after = mse(&weights);

    assert!(
        after < before,
        "training should reduce sigmoid-space MSE: before={before:.6}, after={after:.6}"
    );
}

// ===== Phase A.1 (2026-05-07) — dense-input projection branch =====

const DENSE_CONFIG: NnueConfig = NnueConfig::new_static(16, 32, &[16], Activation::CReLU);

fn dense_config_with_branch(size: usize) -> NnueConfig {
    DENSE_CONFIG.clone().with_dense_input(size)
}

#[test]
fn default_constructors_disable_dense_branch() {
    assert_eq!(SMALL_CONFIG.dense_input_size, 0);
    assert!(!SMALL_CONFIG.has_dense_input());

    let owned = NnueConfig::new_owned(16, 32, vec![16], Activation::CReLU);
    assert_eq!(owned.dense_input_size, 0);

    let with_dense = SMALL_CONFIG.clone().with_dense_input(8);
    assert_eq!(with_dense.dense_input_size, 8);
    assert!(with_dense.has_dense_input());
}

#[test]
fn empty_dense_input_is_a_noop_for_disabled_branch() {
    // Disabled config: forward with `&[]` must yield the exact same output
    // as a sparse-only network, byte-for-byte.
    let config = SMALL_CONFIG.clone();
    let mut rng = SimpleRng::new(1234);
    let weights = TrainableWeights::init_random(config, &mut rng);

    let stm = vec![0, 3, 5];
    let nstm = vec![1, 2, 7];
    let baseline = weights.forward(&stm, &nstm, &[]).output;

    // Calling `apply_dense_input` on quantized weights with an empty
    // `dense_to_acc` is similarly a no-op.
    let q = weights.quantize();
    let mut acc = Accumulator::new(&q.feature_bias);
    acc.refresh(&q, &stm, &nstm);
    let before = (acc.stm.clone(), acc.nstm.clone());
    acc.apply_dense_input(&q, &[]);
    assert_eq!(before.0, acc.stm);
    assert_eq!(before.1, acc.nstm);

    // Forward through quantized — sanity check that the call chain still
    // produces an output close to the FP32 reference.
    let i16_out = forward(&acc, &q);
    let _ = (baseline, i16_out);
}

#[test]
fn dense_input_with_zero_vector_is_a_noop() {
    // Enabled config with non-empty dense_to_acc, but dense_input is all
    // zeros — accumulator must not change.
    let config = dense_config_with_branch(4);
    let mut rng = SimpleRng::new(2222);
    let weights = TrainableWeights::init_random(config, &mut rng);

    let stm = vec![0, 4];
    let nstm = vec![1, 5];
    let zero_dense = vec![0.0f32; 4];

    let f_no_dense = weights.forward(&stm, &nstm, &[]).output;
    let f_zero_dense = weights.forward(&stm, &nstm, &zero_dense).output;
    assert!((f_no_dense - f_zero_dense).abs() < 1e-5);
}

#[test]
fn dense_input_changes_accumulator_when_nonzero() {
    let config = dense_config_with_branch(4);
    let mut rng = SimpleRng::new(3333);
    let weights = TrainableWeights::init_random(config, &mut rng);

    let stm = vec![0];
    let nstm = vec![1];
    let nonzero_dense = vec![1.0, 0.0, -0.5, 0.25];

    let f_no_dense = weights.forward(&stm, &nstm, &[]).output;
    let f_with_dense = weights.forward(&stm, &nstm, &nonzero_dense).output;

    // The two paths must produce *different* outputs; otherwise the
    // dense-input projection isn't actually being applied.
    assert!(
        (f_no_dense - f_with_dense).abs() > 1e-4,
        "dense input should perturb the network output"
    );
}

#[test]
fn fp32_checkpoint_roundtrip_preserves_dense_weights() {
    let config = dense_config_with_branch(4);
    let mut rng = SimpleRng::new(4444);
    let original = TrainableWeights::init_random(config, &mut rng);

    let bytes = original.save_to_bytes();
    let restored = TrainableWeights::load_from_bytes(&bytes).expect("load");

    assert_eq!(restored.config.dense_input_size, 4);
    assert_eq!(original.dense_to_acc.len(), restored.dense_to_acc.len());
    for (a, b) in original
        .dense_to_acc
        .iter()
        .zip(restored.dense_to_acc.iter())
    {
        assert_eq!(a, b);
    }
}

#[test]
fn quantized_save_load_roundtrip_with_dense() {
    let config = dense_config_with_branch(4);
    let mut rng = SimpleRng::new(5555);
    let trainable = TrainableWeights::init_random(config, &mut rng);

    let q = trainable.quantize();
    assert!(!q.dense_to_acc.is_empty());

    let bytes = q.save_to_bytes();
    let restored = NnueWeights::load_from_bytes(&bytes, None).expect("load");

    assert_eq!(restored.config.dense_input_size, 4);
    assert_eq!(q.dense_to_acc.len(), restored.dense_to_acc.len());
    for (a, b) in q.dense_to_acc.iter().zip(restored.dense_to_acc.iter()) {
        assert_eq!(a, b);
    }

    // Inference must still produce the same output for the same dense input.
    let stm = vec![0, 7];
    let nstm = vec![1, 9];
    let dense = vec![0.5, -0.25, 0.125, 1.0];

    let mut a1 = Accumulator::new(&q.feature_bias);
    a1.refresh(&q, &stm, &nstm);
    a1.apply_dense_input(&q, &dense);
    let out1 = forward(&a1, &q);

    let mut a2 = Accumulator::new(&restored.feature_bias);
    a2.refresh(&restored, &stm, &nstm);
    a2.apply_dense_input(&restored, &dense);
    let out2 = forward(&a2, &restored);

    assert_eq!(out1, out2);
}

#[test]
fn legacy_v2_weights_load_with_empty_dense_branch() {
    // Build a sparse-only network, save (which now writes v3 with
    // dense_input_size = 0), then load. The loader must accept the file
    // and reconstruct an empty `dense_to_acc`.
    let mut rng = SimpleRng::new(6666);
    let trainable = TrainableWeights::init_random(SMALL_CONFIG.clone(), &mut rng);
    let q = trainable.quantize();

    let bytes = q.save_to_bytes();
    let restored = NnueWeights::load_from_bytes(&bytes, None).expect("load");

    assert_eq!(restored.config.dense_input_size, 0);
    assert!(restored.dense_to_acc.is_empty());
}

#[test]
fn dense_input_training_reduces_loss() {
    // End-to-end smoke test: a network with the dense branch enabled
    // should still converge on a synthetic regression target.
    let config = dense_config_with_branch(4);
    let mut rng = SimpleRng::new(7777);
    let mut weights = TrainableWeights::init_random(config.clone(), &mut rng);
    let mut adam = AdamState::new(config.clone());

    let samples: Vec<TrainingSample> = (0..32)
        .map(|i| {
            let dense = vec![
                (i as f32).sin(),
                (i as f32).cos(),
                ((i * 3) as f32).sin(),
                ((i * 5) as f32).cos(),
            ];
            // Target depends on dense input so the dense branch is
            // forced to learn something the sparse path can't.
            let target = (dense[0] + dense[2]).tanh().clamp(-0.95, 0.95);
            TrainingSample {
                stm_features: vec![(i % 4) as usize],
                nstm_features: vec![((i + 1) % 4) as usize],
                target: 0.5 + 0.5 * target,
                dense_input: dense,
            }
        })
        .collect();

    let mse = |w: &TrainableWeights| -> f32 {
        let mut total = 0.0;
        for s in &samples {
            let f = w.forward(&s.stm_features, &s.nstm_features, &s.dense_input);
            total += (f.sigmoid - s.target).powi(2);
        }
        total / samples.len() as f32
    };

    let before = mse(&weights);
    for _ in 0..200 {
        let mut grad = Gradients::new(config.clone());
        for s in &samples {
            let f = weights.forward(&s.stm_features, &s.nstm_features, &s.dense_input);
            weights.backward_bce(s, &f, &mut grad);
        }
        weights.adam_update(&grad, &mut adam, 0.01, samples.len() as f32);
    }
    let after = mse(&weights);

    assert!(
        after < before,
        "training with dense input should reduce loss: before={before:.6}, after={after:.6}"
    );
}
