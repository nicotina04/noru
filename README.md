![](.github/logo.png)

# NORU

**N**NUE **O**n **RU**st — Zero-dependency NNUE training & inference library in pure Rust.

## What is NNUE?

[NNUE](https://www.chessprogramming.org/NNUE) (Efficiently Updatable Neural Network) is a neural network architecture designed for fast evaluation in game engines. Originally developed for Shogi and adopted by Stockfish, NNUE enables real-time neural network inference through incremental accumulator updates.

## What is NORU?

NORU is a **game-agnostic** NNUE library that provides both training and inference in a single, dependency-free Rust crate. Configure your network dimensions at runtime via `NnueConfig` — no recompilation needed.

### Key Features

- **Training + Inference** — FP32 backpropagation with Adam optimizer, i16 quantized inference
- **Zero dependencies** — Pure Rust, no PyTorch, no CUDA, no C bindings
- **Game-agnostic** — Runtime-configurable network dimensions via `NnueConfig`
- **Incremental updates** — Efficient accumulator add/remove for search trees
- **Quantization** — Automatic FP32 → i16 conversion for deployment

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
noru = { git = "https://github.com/nicotina04/noru" }
```

### Training

```rust
use noru::config::NnueConfig;
use noru::trainer::{TrainableWeights, AdamState, Gradients, TrainingSample, SimpleRng};

// 1. Define your network dimensions
let config = NnueConfig {
    feature_size: 530,       // your game's feature count
    accumulator_size: 256,   // hidden accumulator neurons
    hidden_size: 64,         // hidden layer neurons
};

// 2. Initialize weights
let mut rng = SimpleRng::new(42);
let mut weights = TrainableWeights::init_random(config, &mut rng);
let mut adam = AdamState::new(config);

// 3. Train on samples
let sample = TrainingSample {
    stm_features: vec![0, 42, 100],   // active feature indices (side-to-move)
    nstm_features: vec![10, 50, 200], // active feature indices (opponent)
    target: 0.8,                       // evaluation target
};

let fwd = weights.forward(&sample.stm_features, &sample.nstm_features);
let mut grad = Gradients::new(config);
weights.backward(&sample, &fwd, &mut grad);  // BCE loss
weights.adam_update(&grad, &mut adam, 0.001, 1.0);

// 4. Quantize for deployment
let inference_weights = weights.quantize(); // FP32 → i16
```

### Inference

```rust
use noru::config::NnueConfig;
use noru::network::{NnueWeights, Accumulator, forward};

// Load quantized weights
let weights = NnueWeights::load_from_bytes(&model_bytes, config)?;

// Evaluate a position
let mut acc = Accumulator::new(&weights.feature_bias);
acc.refresh(&weights, &stm_features, &nstm_features);
let eval: i32 = forward(&acc, &weights);

// Incremental update (for search trees)
use noru::network::FeatureDelta;
let mut delta_stm = FeatureDelta::new();
delta_stm.add(new_feature);
delta_stm.remove(old_feature);
acc.update_incremental(&weights, &delta_stm, &delta_nstm);
```

## Architecture

```
Input (sparse features)
  ↓
Feature Transform: [feature_size] → [accumulator_size] (per perspective)
  ↓
ClippedReLU
  ↓
Concat: [accumulator_size × 2] (STM + NSTM perspectives)
  ↓
Hidden Layer: [accumulator_size × 2] → [hidden_size]
  ↓
ClippedReLU
  ↓
Output Layer: [hidden_size] → 1 (evaluation score)
```

All dimensions are configured at runtime:

```rust
let config = NnueConfig {
    feature_size: 530,      // depends on your game
    accumulator_size: 256,  // accuracy vs speed tradeoff
    hidden_size: 64,        // accuracy vs speed tradeoff
};
```

## API Reference

### `noru::config`

| Type | Description |
|------|-------------|
| `NnueConfig` | Network dimensions (feature_size, accumulator_size, hidden_size) |

### `noru::network` (Inference, i16)

| Type / Function | Description |
|-----------------|-------------|
| `NnueWeights` | Quantized i16 weights for inference |
| `NnueWeights::load_from_bytes()` | Load weights from binary file |
| `Accumulator` | Maintains per-perspective activation sums |
| `Accumulator::refresh()` | Full recomputation from feature list |
| `Accumulator::update_incremental()` | Efficient add/remove update |
| `Accumulator::swap()` | Swap STM/NSTM perspectives |
| `FeatureDelta` | Tracks added/removed features for incremental updates |
| `forward()` | Full forward pass: Accumulator → Hidden → Output |

### `noru::trainer` (Training, FP32)

| Type / Function | Description |
|-----------------|-------------|
| `TrainableWeights` | FP32 weights with training methods |
| `TrainableWeights::init_random()` | Kaiming initialization |
| `TrainableWeights::forward()` | FP32 forward pass with intermediate results |
| `TrainableWeights::backward()` | Backpropagation (BCE loss) |
| `TrainableWeights::backward_mse()` | Backpropagation (MSE loss) |
| `TrainableWeights::adam_update()` | Adam optimizer step |
| `TrainableWeights::quantize()` | FP32 → i16 for deployment |
| `AdamState` | Adam optimizer momentum/velocity state |
| `Gradients` | Gradient accumulation buffer |
| `TrainingSample` | Training data (features + target) |
| `SimpleRng` | Built-in xorshift64 RNG (no external dependency) |

### `noru::quant`

| Constant / Function | Description |
|---------------------|-------------|
| `WEIGHT_SCALE` (64) | FP32 → i16 quantization scale |
| `ACTIVATION_SCALE` (256) | Accumulator → Hidden scale |
| `OUTPUT_SCALE` (16) | Final output scale |
| `clipped_relu()` | ClippedReLU activation |
| `saturate_i16()` | Safe i32 → i16 conversion |

## Building

```bash
# Library
cargo build --release

# Run tests
cargo test

# Build as C-compatible shared library (for FFI / Unity / etc.)
# Add to Cargo.toml: [lib] crate-type = ["lib", "cdylib"]
cargo build --release
# Output: target/release/libnoru.so (Linux) / noru.dll (Windows) / libnoru.dylib (macOS)
```

## Design Decisions

- **No GPU** — Designed for real-time game AI on CPU. NNUE's strength is being fast enough for depth-4+ search on consumer hardware.
- **No external dependencies** — Even the RNG is built-in (xorshift64). This means `cargo add noru` just works, everywhere.
- **Vec\<T\> over fixed arrays** — All weights use heap-allocated vectors for runtime flexibility. Slight overhead vs compile-time arrays, but enables one binary for any game.
- **Sparse feature input** — Features are passed as active index lists, not dense vectors. This matches NNUE's design for board games where most features are inactive.

## License

Licensed under either of

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.

## Related Projects

- [Stockfish NNUE](https://github.com/official-stockfish/Stockfish) — The chess engine that popularized NNUE
- [bullet](https://github.com/jw1912/bullet) — GPU-accelerated NNUE training (Rust + CUDA)
- [Rapfi](https://github.com/dhbloo/rapfi) — Gomoku engine with advanced NNUE
