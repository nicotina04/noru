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
use noru::network::NnueWeights;
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

    // Quantize → save → load (v2 auto-detected), then audit FP32 vs i16 drift.
    let quantized = weights.quantize();
    let bytes = quantized.save_to_bytes();
    let reloaded = NnueWeights::load_from_bytes(&bytes, None).expect("v2 header should be present");
    let report = reloaded
        .audit_against_fp32(&weights, &samples)
        .expect("sample set is non-empty");

    println!("Serialized model size : {} bytes", report.model_bytes);
    println!("Samples evaluated     : {}", report.sample_count);
    println!(
        "FP32 raw range        : [{:.3}, {:.3}]",
        report.fp32_output_min, report.fp32_output_max
    );
    println!(
        "i16 score range       : [{}, {}]",
        report.i16_output_min, report.i16_output_max
    );
    println!(
        "Inferred output scale : {:.3}",
        report.inferred_output_scale
    );
    println!(
        "FP32↔i16 sign agreement: {}/{} ({:.1}%)",
        report.sign_agreement,
        report.sample_count,
        100.0 * report.sign_agreement_ratio,
    );
    println!(
        "Raw error (MAE/RMSE/max): {:.3} / {:.3} / {:.3}",
        report.raw_error.mean_abs, report.raw_error.rmse, report.raw_error.max_abs
    );
    println!(
        "Prob error (MAE/RMSE/max): {:.4} / {:.4} / {:.4}",
        report.probability_error.mean_abs,
        report.probability_error.rmse,
        report.probability_error.max_abs
    );
    println!("Round trip            : ok (v2 header auto-detected)");
}
