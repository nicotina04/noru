//! Demonstrates multi-hidden-layer networks with SCReLU activation.
//!
//! The `NnueConfig::hidden_sizes` slice can be any length; noru stacks CReLU
//! between layers internally. SCReLU is applied on the first (accumulator)
//! layer only, matching the Stockfish pattern.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example multi_layer
//! ```

use noru::config::{Activation, NnueConfig};
use noru::network::{forward, Accumulator};
use noru::trainer::{AdamState, Gradients, SimpleRng, TrainableWeights, TrainingSample};

// Stockfish-ish topology, scaled down for a runnable demo.
const CONFIG: NnueConfig = NnueConfig {
    feature_size: 32,
    accumulator_size: 128,
    hidden_sizes: &[32, 16, 16],
    activation: Activation::SCReLU,
};

fn synthetic_sample(rng: &mut SimpleRng, idx: usize) -> TrainingSample {
    let stm_len = 4 + rng.next_usize(4);
    let nstm_len = 4 + rng.next_usize(4);
    let mut stm: Vec<usize> = (0..stm_len)
        .map(|_| rng.next_usize(CONFIG.feature_size))
        .collect();
    let mut nstm: Vec<usize> = (0..nstm_len)
        .map(|_| rng.next_usize(CONFIG.feature_size))
        .collect();
    stm.sort_unstable();
    stm.dedup();
    nstm.sort_unstable();
    nstm.dedup();

    // Deterministic target based on feature overlap — just to give the
    // trainer a learnable signal for demo purposes.
    let overlap = stm.iter().filter(|f| nstm.contains(f)).count();
    let target = (overlap as f32 / stm.len().max(1) as f32).clamp(0.0, 1.0);

    let _ = idx;
    TrainingSample {
        stm_features: stm,
        nstm_features: nstm,
        target,
    }
}

fn main() {
    let mut rng = SimpleRng::new(123);
    let mut weights = TrainableWeights::init_random(CONFIG, &mut rng);
    let mut adam = AdamState::new(CONFIG);

    let samples: Vec<TrainingSample> = (0..256).map(|i| synthetic_sample(&mut rng, i)).collect();

    println!(
        "Config: feature={}, acc={}, hidden={:?}, activation={:?}",
        CONFIG.feature_size, CONFIG.accumulator_size, CONFIG.hidden_sizes, CONFIG.activation,
    );
    println!("Training {} samples for 10 epochs...", samples.len());

    for epoch in 0..10 {
        let mut total = 0.0_f32;
        for sample in &samples {
            let fwd = weights.forward(&sample.stm_features, &sample.nstm_features);
            let mut grad = Gradients::new(CONFIG);
            weights.backward_bce(sample, &fwd, &mut grad);
            weights.adam_update(&grad, &mut adam, 1e-3, 1.0);
            total += (fwd.sigmoid - sample.target).powi(2);
        }
        println!(
            "  epoch {epoch:>2}: mean MSE = {:.6}",
            total / samples.len() as f32
        );
    }

    // Inference with i16 quantized weights.
    let inference = weights.quantize();
    let test = &samples[0];
    let mut acc = Accumulator::new(&inference.feature_bias);
    acc.refresh(&inference, &test.stm_features, &test.nstm_features);
    let eval = forward(&acc, &inference);
    println!(
        "\nFirst sample i16 eval: {eval} (target was {:.3})",
        test.target
    );
}
