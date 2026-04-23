//! Minimal NNUE training + inference round trip.
//!
//! Trains a tiny network on a synthetic 4-feature XOR-style problem, then
//! quantizes the weights and runs inference through the i16 pipeline. Useful
//! as a smoke test / starting template.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example xor
//! ```

use noru::config::{Activation, NnueConfig};
use noru::network::{forward, Accumulator};
use noru::trainer::{AdamState, Gradients, SimpleRng, TrainableWeights, TrainingSample};

// 4 total features; a feature is "active" when its bit is set.
// Target mirrors an XOR-style signal over features {0,1}.
const CONFIG: NnueConfig = NnueConfig {
    feature_size: 4,
    accumulator_size: 16,
    hidden_sizes: &[8],
    activation: Activation::CReLU,
};

fn make_samples() -> Vec<TrainingSample> {
    // stm/nstm are mirrored here for simplicity; in a real game they'd differ.
    let cases = [
        (vec![0, 2], vec![1, 3], 0.0_f32), // 0 XOR 0 = 0
        (vec![0, 3], vec![1, 2], 1.0_f32), // 0 XOR 1 = 1
        (vec![1, 2], vec![0, 3], 1.0_f32), // 1 XOR 0 = 1
        (vec![1, 3], vec![0, 2], 0.0_f32), // 1 XOR 1 = 0
    ];
    cases
        .iter()
        .map(|(stm, nstm, target)| TrainingSample {
            stm_features: stm.clone(),
            nstm_features: nstm.clone(),
            target: *target,
        })
        .collect()
}

fn main() {
    let mut rng = SimpleRng::new(42);
    let mut weights = TrainableWeights::init_random(CONFIG, &mut rng);
    let mut adam = AdamState::new(CONFIG);
    let samples = make_samples();

    println!("Training {} samples for 2000 steps...", samples.len());
    for step in 0..2000 {
        let mut total_loss = 0.0_f32;
        for sample in &samples {
            let fwd = weights.forward(&sample.stm_features, &sample.nstm_features);
            let mut grad = Gradients::new(CONFIG);
            weights.backward_bce(sample, &fwd, &mut grad);
            weights.adam_update(&grad, &mut adam, 1e-2, 1.0);
            // BCE loss operates on fwd.sigmoid (∈ [0,1]); report its MSE for
            // a quick convergence signal.
            total_loss += (fwd.sigmoid - sample.target).powi(2);
        }
        if step % 500 == 0 {
            println!(
                "  step {step:>4}: mean MSE (sigmoid) = {:.6}",
                total_loss / samples.len() as f32
            );
        }
    }

    // Quantize to i16 for deployment.
    let inference = weights.quantize();

    println!("\nInference after quantization:");
    for sample in &samples {
        let mut acc = Accumulator::new(&inference.feature_bias);
        acc.refresh(&inference, &sample.stm_features, &sample.nstm_features);
        let raw = forward(&acc, &inference);
        println!(
            "  stm={:?} nstm={:?} target={:.1} i16_eval={raw}",
            sample.stm_features, sample.nstm_features, sample.target,
        );
    }
}
