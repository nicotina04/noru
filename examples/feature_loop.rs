//! Minimal board-style feature extractor loop built on NORU.
//!
//! This example is intentionally more "application-shaped" than `xor.rs`: it
//! defines a tiny game state, extracts sparse STM/NSTM features from that
//! state, trains on a handcrafted target, and then runs both FP32 and i16
//! inference on one sampled position.
//!
//! Run with:
//!
//! ```sh
//! cargo run --release --example feature_loop
//! ```

use noru::config::{Activation, NnueConfig};
use noru::network::{forward, Accumulator};
use noru::trainer::{AdamState, Gradients, SimpleRng, TrainableWeights, TrainingSample};

const BOARD_SIZE: usize = 4;
const NUM_SQUARES: usize = BOARD_SIZE * BOARD_SIZE;
const PIECE_BUCKETS: usize = 3;
const ADJ_BUCKETS: usize = 3;
const CENTER_FLAG: usize = 1;

const SIDE_FEATURES: usize = NUM_SQUARES + PIECE_BUCKETS + ADJ_BUCKETS + CENTER_FLAG;
const TOTAL_FEATURES: usize = SIDE_FEATURES * 2;

const CONFIG: NnueConfig = NnueConfig::new_static(TOTAL_FEATURES, 32, &[16], Activation::CReLU);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Cell {
    Empty,
    Stm,
    Nstm,
}

#[derive(Clone)]
struct MiniBoard {
    cells: [Cell; NUM_SQUARES],
}

impl MiniBoard {
    fn random(rng: &mut SimpleRng) -> Self {
        let mut cells = [Cell::Empty; NUM_SQUARES];
        let stm_count = 2 + rng.next_usize(3);
        let nstm_count = 2 + rng.next_usize(3);

        let mut placed = 0;
        while placed < stm_count {
            let idx = rng.next_usize(NUM_SQUARES);
            if cells[idx] == Cell::Empty {
                cells[idx] = Cell::Stm;
                placed += 1;
            }
        }

        let mut placed = 0;
        while placed < nstm_count {
            let idx = rng.next_usize(NUM_SQUARES);
            if cells[idx] == Cell::Empty {
                cells[idx] = Cell::Nstm;
                placed += 1;
            }
        }

        Self { cells }
    }

    fn render(&self) -> String {
        let mut out = String::new();
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                let ch = match self.cells[r * BOARD_SIZE + c] {
                    Cell::Empty => '.',
                    Cell::Stm => 'X',
                    Cell::Nstm => 'O',
                };
                out.push(ch);
                if c + 1 != BOARD_SIZE {
                    out.push(' ');
                }
            }
            if r + 1 != BOARD_SIZE {
                out.push('\n');
            }
        }
        out
    }
}

fn side_features(board: &MiniBoard, side: Cell, base: usize) -> Vec<usize> {
    let mut features = Vec::with_capacity(12);
    let mut piece_count = 0usize;
    let mut adjacency = 0usize;
    let mut center_control = false;

    for idx in 0..NUM_SQUARES {
        if board.cells[idx] != side {
            continue;
        }
        piece_count += 1;
        features.push(base + idx);

        let row = idx / BOARD_SIZE;
        let col = idx % BOARD_SIZE;
        if (row == 1 || row == 2) && (col == 1 || col == 2) {
            center_control = true;
        }
        if col + 1 < BOARD_SIZE && board.cells[idx + 1] == side {
            adjacency += 1;
        }
        if row + 1 < BOARD_SIZE && board.cells[idx + BOARD_SIZE] == side {
            adjacency += 1;
        }
    }

    let piece_bucket = piece_count.saturating_sub(1).min(PIECE_BUCKETS - 1);
    features.push(base + NUM_SQUARES + piece_bucket);

    let adjacency_bucket = adjacency.min(ADJ_BUCKETS - 1);
    features.push(base + NUM_SQUARES + PIECE_BUCKETS + adjacency_bucket);

    if center_control {
        features.push(base + NUM_SQUARES + PIECE_BUCKETS + ADJ_BUCKETS);
    }

    features
}

fn extract_features(board: &MiniBoard) -> (Vec<usize>, Vec<usize>) {
    (
        side_features(board, Cell::Stm, 0),
        side_features(board, Cell::Nstm, SIDE_FEATURES),
    )
}

fn handcrafted_target(board: &MiniBoard) -> f32 {
    let (stm_features, nstm_features) = extract_features(board);

    let stm_piece_count = stm_features
        .iter()
        .filter(|&&idx| idx < NUM_SQUARES)
        .count() as f32;
    let nstm_piece_count = nstm_features
        .iter()
        .filter(|&&idx| idx >= SIDE_FEATURES && idx < SIDE_FEATURES + NUM_SQUARES)
        .count() as f32;

    let stm_adj = stm_features.iter().any(|&idx| {
        idx == NUM_SQUARES + PIECE_BUCKETS + 1 || idx == NUM_SQUARES + PIECE_BUCKETS + 2
    });
    let nstm_adj = nstm_features.iter().any(|&idx| {
        idx == SIDE_FEATURES + NUM_SQUARES + PIECE_BUCKETS + 1
            || idx == SIDE_FEATURES + NUM_SQUARES + PIECE_BUCKETS + 2
    });
    let stm_center = stm_features
        .iter()
        .any(|&idx| idx == NUM_SQUARES + PIECE_BUCKETS + ADJ_BUCKETS);
    let nstm_center = nstm_features
        .iter()
        .any(|&idx| idx == SIDE_FEATURES + NUM_SQUARES + PIECE_BUCKETS + ADJ_BUCKETS);

    let material = stm_piece_count - nstm_piece_count;
    let center = if stm_center { 0.25 } else { 0.0 } - if nstm_center { 0.25 } else { 0.0 };
    let adjacency = if stm_adj { 0.2 } else { 0.0 } - if nstm_adj { 0.2 } else { 0.0 };

    let raw = material * 0.45 + center + adjacency;
    1.0 / (1.0 + (-raw).exp())
}

fn make_samples(rng: &mut SimpleRng, count: usize) -> Vec<(MiniBoard, TrainingSample)> {
    (0..count)
        .map(|_| {
            let board = MiniBoard::random(rng);
            let (stm_features, nstm_features) = extract_features(&board);
            let sample = TrainingSample {
                stm_features,
                nstm_features,
                target: handcrafted_target(&board),
            };
            (board, sample)
        })
        .collect()
}

fn main() {
    let mut rng = SimpleRng::new(1234);
    let dataset = make_samples(&mut rng, 96);
    let mut weights = TrainableWeights::init_random(CONFIG, &mut rng);
    let mut adam = AdamState::new(CONFIG);

    for _ in 0..200 {
        for (_, sample) in &dataset {
            let fwd = weights.forward(&sample.stm_features, &sample.nstm_features);
            let mut grad = Gradients::new(CONFIG);
            weights.backward_bce(sample, &fwd, &mut grad);
            weights.adam_update(&grad, &mut adam, 5e-3, 1.0);
        }
    }

    let (board, sample) = &dataset[0];
    let fp32 = weights
        .forward(&sample.stm_features, &sample.nstm_features)
        .sigmoid;
    let quantized = weights.quantize();
    let mut acc = Accumulator::new(&quantized.feature_bias);
    acc.refresh(&quantized, &sample.stm_features, &sample.nstm_features);
    let i16_eval = forward(&acc, &quantized);

    println!("Mini board position:\n{}\n", board.render());
    println!("STM features: {:?}", sample.stm_features);
    println!("NSTM features: {:?}", sample.nstm_features);
    println!("Handcrafted target : {:.3}", sample.target);
    println!("FP32 prediction    : {:.3}", fp32);
    println!("i16 eval           : {}", i16_eval);
}
