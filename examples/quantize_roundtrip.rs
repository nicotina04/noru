//! FP32 training → i16 quantization → save → load → inference consistency.
//!
//! Demonstrates the full deployment pipeline and reports the quantization
//! error so users can calibrate their tolerances. Uses the v2 binary format
//! with an embedded NORU header (no config argument needed on load).
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example quantize_roundtrip
//! ```

use noru::config::{Activation, NnueConfig};
use noru::network::{forward, Accumulator, NnueWeights};
use noru::trainer::{AdamState, Gradients, SimpleRng, TrainableWeights, TrainingSample};

const CONFIG: NnueConfig = NnueConfig {
    feature_size: 16,
    accumulator_size: 64,
    hidden_sizes: &[16],
    activation: Activation::CReLU,
};

fn main() {
    let mut rng = SimpleRng::new(7);
    let mut weights = TrainableWeights::init_random(CONFIG, &mut rng);
    let mut adam = AdamState::new(CONFIG);

    // Synthetic training so the network actually learns something non-trivial.
    let samples: Vec<TrainingSample> = (0..64)
        .map(|_| {
            let len = 2 + rng.next_usize(4);
            let mut stm: Vec<usize> = (0..len)
                .map(|_| rng.next_usize(CONFIG.feature_size))
                .collect();
            stm.sort_unstable();
            stm.dedup();
            let mut nstm: Vec<usize> = (0..len)
                .map(|_| rng.next_usize(CONFIG.feature_size))
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
        .collect();

    for _ in 0..500 {
        for sample in &samples {
            let fwd = weights.forward(&sample.stm_features, &sample.nstm_features);
            let mut grad = Gradients::new(CONFIG);
            weights.backward_bce(sample, &fwd, &mut grad);
            weights.adam_update(&grad, &mut adam, 1e-2, 1.0);
        }
    }

    // FP32 reference outputs (post-sigmoid, ∈ [0, 1]).
    let fp32_probs: Vec<f32> = samples
        .iter()
        .map(|s| weights.forward(&s.stm_features, &s.nstm_features).sigmoid)
        .collect();

    // Quantize → save → load (v2 auto-detected).
    let quantized = weights.quantize();
    let bytes = quantized.save_to_bytes();
    let reloaded = NnueWeights::load_from_bytes(&bytes, None).expect("v2 header should be present");

    // Run i16 inference on the reloaded weights. The two pipelines use
    // different internal scales, so we report sign agreement (binary decision
    // stability) and the raw i16 score distribution — direct prob comparison
    // would require knowing the quantization scale.
    let mut sign_agree = 0usize;
    let mut i16_min = i32::MAX;
    let mut i16_max = i32::MIN;
    for (sample, fp32) in samples.iter().zip(&fp32_probs) {
        let mut acc = Accumulator::new(&reloaded.feature_bias);
        acc.refresh(&reloaded, &sample.stm_features, &sample.nstm_features);
        let eval = forward(&acc, &reloaded);
        i16_min = i16_min.min(eval);
        i16_max = i16_max.max(eval);
        // FP32 prob > 0.5 ⇔ eval > 0 on the i16 pipeline (both are monotonic
        // in the network's raw output).
        let fp32_positive = *fp32 > 0.5;
        let i16_positive = eval > 0;
        if fp32_positive == i16_positive {
            sign_agree += 1;
        }
    }

    println!("Serialized model size : {} bytes", bytes.len());
    println!("Samples evaluated     : {}", samples.len());
    println!("i16 score range       : [{}, {}]", i16_min, i16_max);
    println!(
        "FP32↔i16 sign agreement: {}/{} ({:.1}%)",
        sign_agree,
        samples.len(),
        100.0 * sign_agree as f32 / samples.len() as f32,
    );
    println!("Round trip            : ok (v2 header auto-detected)");
}
