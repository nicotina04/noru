/// NNUE forward pass and incremental accumulator update.
///
/// Network structure:
///   Input (sparse) → Accumulator → Hidden₁ → ... → Hiddenₙ → Output (1)
///
/// All dimensions are runtime-configurable via NnueConfig.
/// Hidden layer weights use output-major flat layout for SIMD-friendly access.

use crate::config::{Activation, NnueConfig};
use crate::quant::{clipped_relu, saturate_i16, ACTIVATION_SCALE, OUTPUT_SCALE};
use crate::simd;

pub const CLIPPED_RELU_MAX: i16 = 127;

// Binary format v2 magic number: "NORU" in little-endian
const MAGIC: u32 = 0x4E4F5255;
const FORMAT_VERSION: u32 = 2;

/// Feature delta for incremental accumulator update.
#[derive(Debug, Clone, Copy)]
pub struct FeatureDelta {
    pub added: [usize; 32],
    pub num_added: usize,
    pub removed: [usize; 32],
    pub num_removed: usize,
}

impl FeatureDelta {
    pub fn new() -> Self {
        Self {
            added: [0; 32],
            num_added: 0,
            removed: [0; 32],
            num_removed: 0,
        }
    }

    pub fn add(&mut self, index: usize) {
        if self.num_added < 32 {
            self.added[self.num_added] = index;
            self.num_added += 1;
        }
    }

    pub fn remove(&mut self, index: usize) {
        if self.num_removed < 32 {
            self.removed[self.num_removed] = index;
            self.num_removed += 1;
        }
    }
}

/// NNUE weights for i16 quantized inference.
///
/// Hidden layer weights are stored in output-major flat layout:
/// `hidden_weights[layer]` is a flat `Vec<i16>` of size `output_size * input_size`,
/// indexed as `[output_idx * input_size + input_idx]`.
pub struct NnueWeights {
    pub config: NnueConfig,
    /// Feature → Accumulator weights \[feature_size\]\[accumulator_size\]
    pub feature_weights: Vec<Vec<i16>>,
    /// Feature → Accumulator bias \[accumulator_size\]
    pub feature_bias: Vec<i16>,
    /// Hidden layer weights \[num_layers\] each flat \[output_size * input_size\], output-major
    pub hidden_weights: Vec<Vec<i16>>,
    /// Hidden layer biases \[num_layers\]\[output_size\]
    pub hidden_biases: Vec<Vec<i16>>,
    /// Hidden → Output weights \[last_hidden_size\]
    pub output_weights: Vec<i16>,
    /// Output bias
    pub output_bias: i16,
}

impl NnueWeights {
    /// Create zero-initialized weights.
    pub fn zeros(config: NnueConfig) -> Self {
        let num_layers = config.num_hidden_layers();

        let mut hidden_weights = Vec::with_capacity(num_layers);
        let mut hidden_biases = Vec::with_capacity(num_layers);

        for k in 0..num_layers {
            let in_size = config.layer_input_size(k);
            let out_size = config.hidden_sizes[k];
            hidden_weights.push(vec![0i16; out_size * in_size]);
            hidden_biases.push(vec![0i16; out_size]);
        }

        Self {
            config,
            feature_weights: vec![vec![0i16; config.accumulator_size]; config.feature_size],
            feature_bias: vec![0i16; config.accumulator_size],
            hidden_weights,
            hidden_biases,
            output_weights: vec![0i16; config.last_hidden_size()],
            output_bias: 0,
        }
    }

    /// Get a weight row for output neuron j in hidden layer k (contiguous slice).
    #[inline]
    fn hidden_row(&self, k: usize, j: usize) -> &[i16] {
        let in_size = self.config.layer_input_size(k);
        let start = j * in_size;
        &self.hidden_weights[k][start..start + in_size]
    }

    /// Save to v2 binary format.
    pub fn save_to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Header
        buf.extend_from_slice(&MAGIC.to_le_bytes());
        buf.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(self.config.feature_size as u32).to_le_bytes());
        buf.extend_from_slice(&(self.config.accumulator_size as u32).to_le_bytes());
        buf.extend_from_slice(&(self.config.num_hidden_layers() as u32).to_le_bytes());
        for &hs in self.config.hidden_sizes {
            buf.extend_from_slice(&(hs as u32).to_le_bytes());
        }
        let act_byte: u8 = match self.config.activation {
            Activation::CReLU => 0,
            Activation::SCReLU => 1,
        };
        buf.push(act_byte);

        let write_i16_slice = |buf: &mut Vec<u8>, data: &[i16]| {
            for &v in data {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        };

        // Feature weights
        for row in &self.feature_weights {
            write_i16_slice(&mut buf, row);
        }
        write_i16_slice(&mut buf, &self.feature_bias);

        // Hidden layers: write in input-major order (serialization format)
        // Internal layout is output-major, so transpose on save
        for k in 0..self.config.num_hidden_layers() {
            let in_size = self.config.layer_input_size(k);
            let out_size = self.config.hidden_sizes[k];
            for i in 0..in_size {
                for j in 0..out_size {
                    let val = self.hidden_weights[k][j * in_size + i];
                    buf.extend_from_slice(&val.to_le_bytes());
                }
            }
            write_i16_slice(&mut buf, &self.hidden_biases[k]);
        }

        // Output
        write_i16_slice(&mut buf, &self.output_weights);
        buf.extend_from_slice(&self.output_bias.to_le_bytes());

        buf
    }

    /// Load from binary (auto-detects v2 header, falls back to legacy).
    pub fn load_from_bytes(data: &[u8], legacy_config: Option<NnueConfig>) -> Result<Self, &'static str> {
        if data.len() >= 4 {
            let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            if magic == MAGIC {
                return Self::load_v2(data);
            }
        }
        let config = legacy_config.ok_or("legacy format requires config")?;
        Self::load_legacy(data, config)
    }

    fn load_v2(data: &[u8]) -> Result<Self, &'static str> {
        let mut cursor = 0;

        let read_u32 = |cursor: &mut usize| -> Result<u32, &'static str> {
            if *cursor + 4 > data.len() {
                return Err("unexpected EOF reading header");
            }
            let val = u32::from_le_bytes([
                data[*cursor],
                data[*cursor + 1],
                data[*cursor + 2],
                data[*cursor + 3],
            ]);
            *cursor += 4;
            Ok(val)
        };

        let magic = read_u32(&mut cursor)?;
        if magic != MAGIC {
            return Err("invalid magic number");
        }
        let version = read_u32(&mut cursor)?;
        if version != FORMAT_VERSION {
            return Err("unsupported format version");
        }

        let feature_size = read_u32(&mut cursor)? as usize;
        let accumulator_size = read_u32(&mut cursor)? as usize;
        let num_layers = read_u32(&mut cursor)? as usize;

        if num_layers == 0 || num_layers > 16 {
            return Err("invalid number of hidden layers");
        }

        let mut hidden_sizes_owned = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            hidden_sizes_owned.push(read_u32(&mut cursor)? as usize);
        }

        if cursor >= data.len() {
            return Err("unexpected EOF reading activation");
        }
        let activation = match data[cursor] {
            0 => Activation::CReLU,
            1 => Activation::SCReLU,
            _ => return Err("unknown activation type"),
        };
        cursor += 1;

        let hidden_sizes: &'static [usize] = hidden_sizes_owned.leak();

        let config = NnueConfig {
            feature_size,
            accumulator_size,
            hidden_sizes,
            activation,
        };

        let mut weights = Self::zeros(config);
        Self::read_weights_from(data, &mut cursor, &mut weights)?;
        Ok(weights)
    }

    fn load_legacy(data: &[u8], config: NnueConfig) -> Result<Self, &'static str> {
        let mut weights = Self::zeros(config);
        let mut cursor = 0;
        Self::read_weights_from(data, &mut cursor, &mut weights)?;
        Ok(weights)
    }

    fn read_weights_from(
        data: &[u8],
        cursor: &mut usize,
        weights: &mut Self,
    ) -> Result<(), &'static str> {
        let config = weights.config;
        let acc = config.accumulator_size;

        let read_i16 =
            |cursor: &mut usize, count: usize| -> Result<Vec<i16>, &'static str> {
                let byte_count = count * 2;
                if *cursor + byte_count > data.len() {
                    return Err("unexpected EOF in weight file");
                }
                let mut result = Vec::with_capacity(count);
                for i in 0..count {
                    let lo = data[*cursor + i * 2] as i16;
                    let hi = data[*cursor + i * 2 + 1] as i16;
                    result.push(lo | (hi << 8));
                }
                *cursor += byte_count;
                Ok(result)
            };

        // Feature weights
        for i in 0..config.feature_size {
            weights.feature_weights[i] = read_i16(cursor, acc)?;
        }
        weights.feature_bias = read_i16(cursor, acc)?;

        // Hidden layers: file is input-major [in][out], we store output-major
        for k in 0..config.num_hidden_layers() {
            let in_size = config.layer_input_size(k);
            let out_size = config.hidden_sizes[k];

            // Read input-major, transpose to output-major
            let mut flat = vec![0i16; out_size * in_size];
            for i in 0..in_size {
                let row = read_i16(cursor, out_size)?;
                for j in 0..out_size {
                    flat[j * in_size + i] = row[j];
                }
            }
            weights.hidden_weights[k] = flat;
            weights.hidden_biases[k] = read_i16(cursor, out_size)?;
        }

        // Output
        let last_hid = config.last_hidden_size();
        weights.output_weights = read_i16(cursor, last_hid)?;
        let obias = read_i16(cursor, 1)?;
        weights.output_bias = obias[0];

        Ok(())
    }
}

/// Accumulator: sum of active feature weights.
#[derive(Clone)]
pub struct Accumulator {
    /// STM (side to move) perspective
    pub stm: Vec<i16>,
    /// NSTM (non-side to move) perspective
    pub nstm: Vec<i16>,
}

impl Accumulator {
    pub fn new(bias: &[i16]) -> Self {
        Self {
            stm: bias.to_vec(),
            nstm: bias.to_vec(),
        }
    }

    /// Full recomputation.
    pub fn refresh(
        &mut self,
        weights: &NnueWeights,
        stm_features: &[usize],
        nstm_features: &[usize],
    ) {
        self.stm.copy_from_slice(&weights.feature_bias);
        self.nstm.copy_from_slice(&weights.feature_bias);

        for &feat in stm_features {
            simd::vec_add_i16(&mut self.stm, &weights.feature_weights[feat]);
        }
        for &feat in nstm_features {
            simd::vec_add_i16(&mut self.nstm, &weights.feature_weights[feat]);
        }
    }

    /// Incremental update.
    pub fn update_incremental(
        &mut self,
        weights: &NnueWeights,
        stm_delta: &FeatureDelta,
        nstm_delta: &FeatureDelta,
    ) {
        apply_delta(&mut self.stm, weights, stm_delta);
        apply_delta(&mut self.nstm, weights, nstm_delta);
    }

    /// Undo incremental update (reverse add/remove).
    pub fn update_incremental_undo(
        &mut self,
        weights: &NnueWeights,
        stm_delta: &FeatureDelta,
        nstm_delta: &FeatureDelta,
    ) {
        apply_delta_reversed(&mut self.stm, weights, stm_delta);
        apply_delta_reversed(&mut self.nstm, weights, nstm_delta);
    }

    /// Swap perspectives (STM ↔ NSTM).
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.stm, &mut self.nstm);
    }
}

fn apply_delta(acc: &mut [i16], weights: &NnueWeights, delta: &FeatureDelta) {
    for i in 0..delta.num_removed {
        simd::vec_sub_i16(acc, &weights.feature_weights[delta.removed[i]]);
    }
    for i in 0..delta.num_added {
        simd::vec_add_i16(acc, &weights.feature_weights[delta.added[i]]);
    }
}

fn apply_delta_reversed(acc: &mut [i16], weights: &NnueWeights, delta: &FeatureDelta) {
    for i in 0..delta.num_added {
        simd::vec_sub_i16(acc, &weights.feature_weights[delta.added[i]]);
    }
    for i in 0..delta.num_removed {
        simd::vec_add_i16(acc, &weights.feature_weights[delta.removed[i]]);
    }
}

/// NNUE forward pass: Accumulator → Hidden₁ → ... → Hiddenₙ → Output
pub fn forward(acc: &Accumulator, weights: &NnueWeights) -> i32 {
    let config = weights.config;
    let acc_size = config.accumulator_size;
    let use_screlu = config.activation == Activation::SCReLU;

    // 1. ClippedReLU on concatenated accumulator
    let mut prev = vec![0i16; acc_size * 2];
    simd::vec_clipped_relu(&mut prev[..acc_size], &acc.stm);
    simd::vec_clipped_relu(&mut prev[acc_size..], &acc.nstm);

    // 2. Hidden layers
    for k in 0..config.num_hidden_layers() {
        let out_size = config.hidden_sizes[k];
        let mut next = vec![0i16; out_size];

        if k == 0 && use_screlu {
            // SCReLU: clamped² × weight, i64 accumulation
            for j in 0..out_size {
                let w_row = weights.hidden_row(k, j);
                let mut sum: i64 =
                    weights.hidden_biases[k][j] as i64 * ACTIVATION_SCALE as i64;
                sum += simd::dot_screlu_i64(&prev, w_row);
                let scaled = (sum / CLIPPED_RELU_MAX as i64 / ACTIVATION_SCALE as i64) as i32;
                next[j] = clipped_relu(saturate_i16(scaled), CLIPPED_RELU_MAX);
            }
        } else {
            // CReLU: standard dot product
            for j in 0..out_size {
                let w_row = weights.hidden_row(k, j);
                let mut sum: i32 =
                    weights.hidden_biases[k][j] as i32 * ACTIVATION_SCALE;
                sum += simd::dot_i16_i32(&prev, w_row);
                next[j] = clipped_relu(saturate_i16(sum / ACTIVATION_SCALE), CLIPPED_RELU_MAX);
            }
        }

        prev = next;
    }

    // 3. Output layer
    let mut output: i32 = weights.output_bias as i32 * OUTPUT_SCALE;
    for j in 0..prev.len() {
        output += prev[j] as i32 * weights.output_weights[j] as i32;
    }

    output / OUTPUT_SCALE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> NnueConfig {
        NnueConfig {
            feature_size: 530,
            accumulator_size: 512,
            hidden_sizes: &[64],
            activation: Activation::CReLU,
        }
    }

    fn multi_layer_config() -> NnueConfig {
        NnueConfig {
            feature_size: 64,
            accumulator_size: 32,
            hidden_sizes: &[16, 8],
            activation: Activation::CReLU,
        }
    }

    #[test]
    fn test_zero_weights_eval_is_zero() {
        let config = test_config();
        let weights = NnueWeights::zeros(config);
        let acc = Accumulator::new(&weights.feature_bias);
        let eval = forward(&acc, &weights);
        assert_eq!(eval, 0);
    }

    #[test]
    fn test_accumulator_incremental_matches_refresh() {
        let config = test_config();
        let mut weights = NnueWeights::zeros(config);
        let acc_size = config.accumulator_size;

        for j in 0..acc_size {
            weights.feature_weights[0][j] = (j % 10) as i16;
            weights.feature_weights[1][j] = -((j % 7) as i16);
            weights.feature_weights[100][j] = ((j % 5) as i16) * 3;
        }

        let mut acc_refresh = Accumulator::new(&weights.feature_bias);
        acc_refresh.refresh(&weights, &[0, 100], &[1]);

        let mut acc_inc = Accumulator::new(&weights.feature_bias);
        let mut delta_stm = FeatureDelta::new();
        delta_stm.add(0);
        delta_stm.add(100);
        let mut delta_nstm = FeatureDelta::new();
        delta_nstm.add(1);
        acc_inc.update_incremental(&weights, &delta_stm, &delta_nstm);

        assert_eq!(acc_refresh.stm, acc_inc.stm);
        assert_eq!(acc_refresh.nstm, acc_inc.nstm);
    }

    #[test]
    fn test_multi_layer_zero_weights_eval_is_zero() {
        let config = multi_layer_config();
        let weights = NnueWeights::zeros(config);
        let acc = Accumulator::new(&weights.feature_bias);
        assert_eq!(forward(&acc, &weights), 0);
    }

    #[test]
    fn test_save_load_roundtrip() {
        let config = multi_layer_config();
        let mut weights = NnueWeights::zeros(config);
        // Set some weights via the flat layout
        weights.feature_weights[0][0] = 42;
        weights.hidden_weights[0][0] = 7; // output 0, input 0
        weights.hidden_weights[1][0] = -3; // output 0, input 0
        weights.output_weights[0] = 10;
        weights.output_bias = 5;

        let bytes = weights.save_to_bytes();
        let loaded = NnueWeights::load_from_bytes(&bytes, None).unwrap();

        assert_eq!(loaded.config, config);
        assert_eq!(loaded.feature_weights[0][0], 42);
        assert_eq!(loaded.hidden_weights[0][0], 7);
        assert_eq!(loaded.hidden_weights[1][0], -3);
        assert_eq!(loaded.output_weights[0], 10);
        assert_eq!(loaded.output_bias, 5);
    }

    #[test]
    fn test_screlu_forward_differs_from_crelu() {
        let crelu_config = NnueConfig {
            feature_size: 16,
            accumulator_size: 8,
            hidden_sizes: &[4],
            activation: Activation::CReLU,
        };
        let screlu_config = NnueConfig {
            feature_size: 16,
            accumulator_size: 8,
            hidden_sizes: &[4],
            activation: Activation::SCReLU,
        };

        let mut w_crelu = NnueWeights::zeros(crelu_config);
        let mut w_screlu = NnueWeights::zeros(screlu_config);

        // Set non-zero biases and weights
        for i in 0..8 {
            w_crelu.feature_bias[i] = 50;
            w_screlu.feature_bias[i] = 50;
        }
        // hidden_weights flat layout: [output_j * input_size + input_i]
        let in_size = 16; // acc_size * 2
        for j in 0..4 {
            for i in 0..in_size {
                w_crelu.hidden_weights[0][j * in_size + i] = 10;
                w_screlu.hidden_weights[0][j * in_size + i] = 10;
            }
        }
        for j in 0..4 {
            w_crelu.output_weights[j] = 10;
            w_screlu.output_weights[j] = 10;
        }

        let acc_c = Accumulator::new(&w_crelu.feature_bias);
        let acc_s = Accumulator::new(&w_screlu.feature_bias);

        let eval_crelu = forward(&acc_c, &w_crelu);
        let eval_screlu = forward(&acc_s, &w_screlu);

        assert_ne!(eval_crelu, eval_screlu);
    }
}
