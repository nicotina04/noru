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
            }
        })
        .collect()
}

fn train(weights: &mut TrainableWeights, samples: &[TrainingSample], epochs: usize) {
    let mut adam = AdamState::new(SMALL_CONFIG);
    for _ in 0..epochs {
        for sample in samples {
            let fwd = weights.forward(&sample.stm_features, &sample.nstm_features);
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
            let f = w.forward(&s.stm_features, &s.nstm_features);
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
