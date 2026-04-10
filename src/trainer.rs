/// NNUE backpropagation trainer (game-agnostic).
///
/// Training is performed in FP32, quantized to i16 for inference.
/// All dimensions are runtime-configurable via NnueConfig.
/// Supports multi-hidden-layer networks and SCReLU activation.

use crate::config::{Activation, NnueConfig};
use crate::network::NnueWeights;
use crate::quant::{crelu_grad_f32, screlu_f32, screlu_grad_f32, WEIGHT_SCALE};

const CRELU_MAX: f32 = 1.0;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPSILON: f32 = 1e-8;

/// FP32 trainable weights.
#[derive(Clone)]
pub struct TrainableWeights {
    pub config: NnueConfig,
    pub ft_weight: Vec<Vec<f32>>,           // \[feature_size\]\[accumulator_size\]
    pub ft_bias: Vec<f32>,                  // \[accumulator_size\]
    pub hidden_weights: Vec<Vec<Vec<f32>>>, // \[num_layers\]\[input_size\]\[output_size\]
    pub hidden_biases: Vec<Vec<f32>>,       // \[num_layers\]\[output_size\]
    pub output_weight: Vec<f32>,            // \[last_hidden_size\]
    pub output_bias: f32,
}

/// Adam optimizer state.
#[derive(Clone)]
pub struct AdamState {
    ft_weight_m: Vec<Vec<f32>>,
    ft_weight_v: Vec<Vec<f32>>,
    ft_bias_m: Vec<f32>,
    ft_bias_v: Vec<f32>,
    hidden_weights_m: Vec<Vec<Vec<f32>>>,
    hidden_weights_v: Vec<Vec<Vec<f32>>>,
    hidden_biases_m: Vec<Vec<f32>>,
    hidden_biases_v: Vec<Vec<f32>>,
    output_weight_m: Vec<f32>,
    output_weight_v: Vec<f32>,
    output_bias_m: f32,
    output_bias_v: f32,
    pub step: u32,
}

impl AdamState {
    pub fn new(config: NnueConfig) -> Self {
        let acc = config.accumulator_size;
        let num_layers = config.num_hidden_layers();

        let mut hw_m = Vec::with_capacity(num_layers);
        let mut hw_v = Vec::with_capacity(num_layers);
        let mut hb_m = Vec::with_capacity(num_layers);
        let mut hb_v = Vec::with_capacity(num_layers);

        for k in 0..num_layers {
            let in_size = config.layer_input_size(k);
            let out_size = config.hidden_sizes[k];
            hw_m.push(vec![vec![0.0; out_size]; in_size]);
            hw_v.push(vec![vec![0.0; out_size]; in_size]);
            hb_m.push(vec![0.0; out_size]);
            hb_v.push(vec![0.0; out_size]);
        }

        Self {
            ft_weight_m: vec![vec![0.0; acc]; config.feature_size],
            ft_weight_v: vec![vec![0.0; acc]; config.feature_size],
            ft_bias_m: vec![0.0; acc],
            ft_bias_v: vec![0.0; acc],
            hidden_weights_m: hw_m,
            hidden_weights_v: hw_v,
            hidden_biases_m: hb_m,
            hidden_biases_v: hb_v,
            output_weight_m: vec![0.0; config.last_hidden_size()],
            output_weight_v: vec![0.0; config.last_hidden_size()],
            output_bias_m: 0.0,
            output_bias_v: 0.0,
            step: 0,
        }
    }
}

/// Gradient accumulation buffer.
pub struct Gradients {
    pub config: NnueConfig,
    pub ft_weight: Vec<Vec<f32>>,
    pub ft_bias: Vec<f32>,
    pub hidden_weights: Vec<Vec<Vec<f32>>>,
    pub hidden_biases: Vec<Vec<f32>>,
    pub output_weight: Vec<f32>,
    pub output_bias: f32,
}

impl Gradients {
    pub fn new(config: NnueConfig) -> Self {
        let acc = config.accumulator_size;
        let num_layers = config.num_hidden_layers();

        let mut hw = Vec::with_capacity(num_layers);
        let mut hb = Vec::with_capacity(num_layers);

        for k in 0..num_layers {
            let in_size = config.layer_input_size(k);
            let out_size = config.hidden_sizes[k];
            hw.push(vec![vec![0.0; out_size]; in_size]);
            hb.push(vec![0.0; out_size]);
        }

        Self {
            config,
            ft_weight: vec![vec![0.0; acc]; config.feature_size],
            ft_bias: vec![0.0; acc],
            hidden_weights: hw,
            hidden_biases: hb,
            output_weight: vec![0.0; config.last_hidden_size()],
            output_bias: 0.0,
        }
    }

    pub fn zero(&mut self) {
        for row in self.ft_weight.iter_mut() {
            for v in row.iter_mut() {
                *v = 0.0;
            }
        }
        self.ft_bias.iter_mut().for_each(|v| *v = 0.0);
        for layer in self.hidden_weights.iter_mut() {
            for row in layer.iter_mut() {
                for v in row.iter_mut() {
                    *v = 0.0;
                }
            }
        }
        for bias in self.hidden_biases.iter_mut() {
            bias.iter_mut().for_each(|v| *v = 0.0);
        }
        self.output_weight.iter_mut().for_each(|v| *v = 0.0);
        self.output_bias = 0.0;
    }
}

/// Training sample.
pub struct TrainingSample {
    pub stm_features: Vec<usize>,
    pub nstm_features: Vec<usize>,
    pub target: f32,
}

/// Forward pass intermediate results.
pub struct ForwardResult {
    pub acc_stm: Vec<f32>,
    pub acc_nstm: Vec<f32>,
    /// Post-activation result of concatenated accumulator
    pub acc_activated: Vec<f32>,
    /// Pre-activation values for each hidden layer
    pub hidden_raws: Vec<Vec<f32>>,
    /// Post-activation values for each hidden layer
    pub hidden_activations: Vec<Vec<f32>>,
    pub output: f32,
    pub sigmoid: f32,
}

impl TrainableWeights {
    /// Kaiming initialization.
    pub fn init_random(config: NnueConfig, rng: &mut SimpleRng) -> Self {
        let acc = config.accumulator_size;
        let num_layers = config.num_hidden_layers();

        let ft_scale = (2.0 / acc as f32).sqrt() * 0.1;

        let mut ft_weight = vec![vec![0.0; acc]; config.feature_size];
        let ft_bias = vec![0.0; acc];

        for row in ft_weight.iter_mut() {
            for v in row.iter_mut() {
                *v = rng.next_normal() * ft_scale;
            }
        }

        let mut hidden_weights = Vec::with_capacity(num_layers);
        let mut hidden_biases = Vec::with_capacity(num_layers);

        for k in 0..num_layers {
            let in_size = config.layer_input_size(k);
            let out_size = config.hidden_sizes[k];
            let scale = (2.0 / in_size as f32).sqrt();

            let mut layer_w = vec![vec![0.0f32; out_size]; in_size];
            for row in layer_w.iter_mut() {
                for v in row.iter_mut() {
                    *v = rng.next_normal() * scale;
                }
            }
            hidden_weights.push(layer_w);
            hidden_biases.push(vec![0.0; out_size]);
        }

        let last_hid = config.last_hidden_size();
        let output_scale = (1.0 / last_hid as f32).sqrt();
        let mut output_weight = vec![0.0f32; last_hid];
        for v in output_weight.iter_mut() {
            *v = rng.next_normal() * output_scale;
        }

        Self {
            config,
            ft_weight,
            ft_bias,
            hidden_weights,
            hidden_biases,
            output_weight,
            output_bias: 0.0,
        }
    }

    /// FP32 forward pass (for training).
    pub fn forward(&self, stm_features: &[usize], nstm_features: &[usize]) -> ForwardResult {
        let acc = self.config.accumulator_size;
        let num_layers = self.config.num_hidden_layers();
        let use_screlu = self.config.activation == Activation::SCReLU;

        // Accumulator
        let mut acc_stm = self.ft_bias.clone();
        for &feat in stm_features {
            for i in 0..acc {
                acc_stm[i] += self.ft_weight[feat][i];
            }
        }

        let mut acc_nstm = self.ft_bias.clone();
        for &feat in nstm_features {
            for i in 0..acc {
                acc_nstm[i] += self.ft_weight[feat][i];
            }
        }

        // Accumulator activation (CReLU or SCReLU)
        let mut acc_activated = vec![0.0f32; acc * 2];
        if use_screlu {
            for i in 0..acc {
                acc_activated[i] = screlu_f32(acc_stm[i], CRELU_MAX);
            }
            for i in 0..acc {
                acc_activated[acc + i] = screlu_f32(acc_nstm[i], CRELU_MAX);
            }
        } else {
            for i in 0..acc {
                acc_activated[i] = acc_stm[i].max(0.0).min(CRELU_MAX);
            }
            for i in 0..acc {
                acc_activated[acc + i] = acc_nstm[i].max(0.0).min(CRELU_MAX);
            }
        }

        // Hidden layers
        let mut hidden_raws = Vec::with_capacity(num_layers);
        let mut hidden_activations = Vec::with_capacity(num_layers);

        for k in 0..num_layers {
            let out_size = self.config.hidden_sizes[k];

            let mut raw = vec![0.0f32; out_size];
            for j in 0..out_size {
                let mut sum = self.hidden_biases[k][j];
                if k == 0 {
                    for i in 0..acc_activated.len() {
                        sum += acc_activated[i] * self.hidden_weights[k][i][j];
                    }
                } else {
                    let prev: &Vec<f32> = &hidden_activations[k - 1];
                    for i in 0..prev.len() {
                        sum += prev[i] * self.hidden_weights[k][i][j];
                    }
                }
                raw[j] = sum;
            }

            // Hidden layers always use CReLU
            let mut activated = vec![0.0f32; out_size];
            for j in 0..out_size {
                activated[j] = raw[j].max(0.0).min(CRELU_MAX);
            }

            hidden_raws.push(raw);
            hidden_activations.push(activated);
        }

        // Output
        let last_activated = &hidden_activations[num_layers - 1];
        let mut output = self.output_bias;
        for j in 0..last_activated.len() {
            output += last_activated[j] * self.output_weight[j];
        }

        let sigmoid = 1.0 / (1.0 + (-output).exp());

        ForwardResult {
            acc_stm,
            acc_nstm,
            acc_activated,
            hidden_raws,
            hidden_activations,
            output,
            sigmoid,
        }
    }

    /// Backpropagation (BCE loss: sigmoid - target).
    pub fn backward(&self, sample: &TrainingSample, fwd: &ForwardResult, grad: &mut Gradients) {
        let d_output = fwd.sigmoid - sample.target;
        self.backward_inner(d_output, sample, fwd, grad);
    }

    /// Backpropagation (MSE loss: output - target, linear output without sigmoid).
    pub fn backward_mse(&self, sample: &TrainingSample, fwd: &ForwardResult, grad: &mut Gradients) {
        let d_output = fwd.output - sample.target;
        self.backward_inner(d_output, sample, fwd, grad);
    }

    fn backward_inner(
        &self,
        d_output: f32,
        sample: &TrainingSample,
        fwd: &ForwardResult,
        grad: &mut Gradients,
    ) {
        let acc = self.config.accumulator_size;
        let num_layers = self.config.num_hidden_layers();
        let use_screlu = self.config.activation == Activation::SCReLU;

        // Output layer gradients
        let last_activated = &fwd.hidden_activations[num_layers - 1];
        grad.output_bias += d_output;
        for j in 0..last_activated.len() {
            grad.output_weight[j] += d_output * last_activated[j];
        }

        // Gradient to propagate to the last hidden layer
        let last_hid = self.config.last_hidden_size();
        let mut d_next = vec![0.0f32; last_hid];
        for j in 0..last_hid {
            d_next[j] = d_output * self.output_weight[j];
        }

        // Hidden layer backpropagation (reverse order)
        for k in (0..num_layers).rev() {
            let out_size = self.config.hidden_sizes[k];

            // Apply CReLU derivative (hidden layers always use CReLU)
            let mut d_pre = vec![0.0f32; out_size];
            for j in 0..out_size {
                d_pre[j] = d_next[j] * crelu_grad_f32(fwd.hidden_raws[k][j], CRELU_MAX);
            }

            // Input to this layer
            let input = if k == 0 {
                &fwd.acc_activated
            } else {
                &fwd.hidden_activations[k - 1]
            };
            let in_size = input.len();

            // Weight/bias gradients
            for j in 0..out_size {
                grad.hidden_biases[k][j] += d_pre[j];
            }
            for i in 0..in_size {
                for j in 0..out_size {
                    grad.hidden_weights[k][i][j] += d_pre[j] * input[i];
                }
            }

            // Propagate gradient to input
            if k > 0 {
                d_next = vec![0.0f32; in_size];
                for i in 0..in_size {
                    let mut sum = 0.0f32;
                    for j in 0..out_size {
                        sum += d_pre[j] * self.hidden_weights[k][i][j];
                    }
                    d_next[i] = sum;
                }
            } else {
                // k == 0: propagate to accumulator activation
                let mut d_acc_activated = vec![0.0f32; in_size];
                for i in 0..in_size {
                    let mut sum = 0.0f32;
                    for j in 0..out_size {
                        sum += d_pre[j] * self.hidden_weights[k][i][j];
                    }
                    d_acc_activated[i] = sum;
                }

                // Accumulator activation derivative
                let mut d_acc = vec![0.0f32; acc * 2];
                if use_screlu {
                    for i in 0..acc {
                        d_acc[i] = d_acc_activated[i]
                            * screlu_grad_f32(fwd.acc_stm[i], CRELU_MAX);
                    }
                    for i in 0..acc {
                        d_acc[acc + i] = d_acc_activated[acc + i]
                            * screlu_grad_f32(fwd.acc_nstm[i], CRELU_MAX);
                    }
                } else {
                    for i in 0..acc {
                        d_acc[i] = d_acc_activated[i]
                            * crelu_grad_f32(fwd.acc_stm[i], CRELU_MAX);
                    }
                    for i in 0..acc {
                        d_acc[acc + i] = d_acc_activated[acc + i]
                            * crelu_grad_f32(fwd.acc_nstm[i], CRELU_MAX);
                    }
                }

                // Feature layer gradients (sparse)
                for i in 0..acc {
                    grad.ft_bias[i] += d_acc[i] + d_acc[acc + i];
                }
                for &feat in &sample.stm_features {
                    for i in 0..acc {
                        grad.ft_weight[feat][i] += d_acc[i];
                    }
                }
                for &feat in &sample.nstm_features {
                    for i in 0..acc {
                        grad.ft_weight[feat][i] += d_acc[acc + i];
                    }
                }
            }
        }
    }

    /// Adam optimizer step.
    pub fn adam_update(
        &mut self,
        grad: &Gradients,
        state: &mut AdamState,
        lr: f32,
        batch_size: f32,
    ) {
        let acc = self.config.accumulator_size;
        let num_layers = self.config.num_hidden_layers();

        state.step += 1;
        let scale = 1.0 / batch_size;
        let bc1 = 1.0 - BETA1.powi(state.step as i32);
        let bc2 = 1.0 - BETA2.powi(state.step as i32);

        // Feature weights (sparse)
        for (feat, grad_row) in grad.ft_weight.iter().enumerate() {
            let any_nonzero = grad_row.iter().any(|&g| g != 0.0);
            if !any_nonzero {
                continue;
            }
            for i in 0..acc {
                let g = grad_row[i] * scale;
                adam_step(
                    &mut self.ft_weight[feat][i],
                    g,
                    &mut state.ft_weight_m[feat][i],
                    &mut state.ft_weight_v[feat][i],
                    lr,
                    bc1,
                    bc2,
                );
            }
        }

        // Feature bias
        for i in 0..acc {
            let g = grad.ft_bias[i] * scale;
            adam_step(
                &mut self.ft_bias[i],
                g,
                &mut state.ft_bias_m[i],
                &mut state.ft_bias_v[i],
                lr,
                bc1,
                bc2,
            );
        }

        // Hidden layers
        for k in 0..num_layers {
            let in_size = self.config.layer_input_size(k);
            let out_size = self.config.hidden_sizes[k];

            for i in 0..in_size {
                for j in 0..out_size {
                    let g = grad.hidden_weights[k][i][j] * scale;
                    adam_step(
                        &mut self.hidden_weights[k][i][j],
                        g,
                        &mut state.hidden_weights_m[k][i][j],
                        &mut state.hidden_weights_v[k][i][j],
                        lr,
                        bc1,
                        bc2,
                    );
                }
            }

            for j in 0..out_size {
                let g = grad.hidden_biases[k][j] * scale;
                adam_step(
                    &mut self.hidden_biases[k][j],
                    g,
                    &mut state.hidden_biases_m[k][j],
                    &mut state.hidden_biases_v[k][j],
                    lr,
                    bc1,
                    bc2,
                );
            }
        }

        // Output weights
        let last_hid = self.config.last_hidden_size();
        for j in 0..last_hid {
            let g = grad.output_weight[j] * scale;
            adam_step(
                &mut self.output_weight[j],
                g,
                &mut state.output_weight_m[j],
                &mut state.output_weight_v[j],
                lr,
                bc1,
                bc2,
            );
        }

        // Output bias
        {
            let g = grad.output_bias * scale;
            adam_step(
                &mut self.output_bias,
                g,
                &mut state.output_bias_m,
                &mut state.output_bias_v,
                lr,
                bc1,
                bc2,
            );
        }
    }

    /// FP32 → i16 quantization.
    pub fn quantize(&self) -> NnueWeights {
        let acc = self.config.accumulator_size;
        let num_layers = self.config.num_hidden_layers();
        let scale = WEIGHT_SCALE as f32;

        let mut weights = NnueWeights::zeros(self.config);

        // Feature weights
        for (feat, row) in self.ft_weight.iter().enumerate() {
            for i in 0..acc {
                weights.feature_weights[feat][i] = (row[i] * scale).round() as i16;
            }
        }
        for i in 0..acc {
            weights.feature_bias[i] = (self.ft_bias[i] * scale).round() as i16;
        }

        // Hidden layers (trainer is input-major, inference is output-major flat)
        for k in 0..num_layers {
            let in_size = self.config.layer_input_size(k);
            let out_size = self.config.hidden_sizes[k];
            for j in 0..out_size {
                for i in 0..in_size {
                    weights.hidden_weights[k][j * in_size + i] =
                        (self.hidden_weights[k][i][j] * scale).round() as i16;
                }
            }
            for j in 0..out_size {
                weights.hidden_biases[k][j] = (self.hidden_biases[k][j] * scale).round() as i16;
            }
        }

        // Output
        let last_hid = self.config.last_hidden_size();
        for j in 0..last_hid {
            weights.output_weights[j] = (self.output_weight[j] * scale).round() as i16;
        }
        weights.output_bias = (self.output_bias * scale).round() as i16;

        weights
    }
}

#[inline]
fn adam_step(param: &mut f32, grad: f32, m: &mut f32, v: &mut f32, lr: f32, bc1: f32, bc2: f32) {
    *m = BETA1 * *m + (1.0 - BETA1) * grad;
    *v = BETA2 * *v + (1.0 - BETA2) * grad * grad;
    let m_hat = *m / bc1;
    let v_hat = *v / bc2;
    *param -= lr * m_hat / (v_hat.sqrt() + EPSILON);
}

/// Simple xorshift64 random number generator.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-10);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
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

    fn screlu_config() -> NnueConfig {
        NnueConfig {
            feature_size: 64,
            accumulator_size: 32,
            hidden_sizes: &[16],
            activation: Activation::SCReLU,
        }
    }

    #[test]
    fn test_forward_backward_no_panic() {
        let config = test_config();
        let mut rng = SimpleRng::new(42);
        let weights = TrainableWeights::init_random(config, &mut rng);

        let sample = TrainingSample {
            stm_features: vec![0, 100, 300],
            nstm_features: vec![50, 200],
            target: 1.0,
        };

        let fwd = weights.forward(&sample.stm_features, &sample.nstm_features);
        assert!(fwd.sigmoid > 0.0 && fwd.sigmoid < 1.0);

        let mut grad = Gradients::new(config);
        weights.backward(&sample, &fwd, &mut grad);

        let has_nonzero = grad.output_weight.iter().any(|&g| g != 0.0);
        assert!(has_nonzero, "gradients should be non-zero");
    }

    #[test]
    fn test_training_reduces_loss() {
        let config = test_config();
        let mut rng = SimpleRng::new(123);
        let mut weights = TrainableWeights::init_random(config, &mut rng);
        let mut state = AdamState::new(config);

        let sample = TrainingSample {
            stm_features: vec![10, 20, 30],
            nstm_features: vec![40, 50],
            target: 1.0,
        };

        let fwd_before = weights.forward(&sample.stm_features, &sample.nstm_features);
        let loss_before = -sample.target * fwd_before.sigmoid.ln()
            - (1.0 - sample.target) * (1.0 - fwd_before.sigmoid).ln();

        for _ in 0..100 {
            let mut grad = Gradients::new(config);
            let fwd = weights.forward(&sample.stm_features, &sample.nstm_features);
            weights.backward(&sample, &fwd, &mut grad);
            weights.adam_update(&grad, &mut state, 0.01, 1.0);
        }

        let fwd_after = weights.forward(&sample.stm_features, &sample.nstm_features);
        let loss_after = -sample.target * fwd_after.sigmoid.ln()
            - (1.0 - sample.target) * (1.0 - fwd_after.sigmoid).ln();

        assert!(
            loss_after < loss_before,
            "loss should decrease: {loss_before} -> {loss_after}"
        );
    }

    #[test]
    fn test_quantize_roundtrip() {
        let config = test_config();
        let mut rng = SimpleRng::new(77);
        let weights = TrainableWeights::init_random(config, &mut rng);
        let quantized = weights.quantize();

        let has_nonzero = quantized
            .feature_weights
            .iter()
            .flat_map(|r| r.iter())
            .any(|&v| v != 0);
        assert!(has_nonzero, "quantized weights should have non-zero values");
    }

    #[test]
    fn test_multi_layer_forward_backward() {
        let config = multi_layer_config();
        let mut rng = SimpleRng::new(42);
        let weights = TrainableWeights::init_random(config, &mut rng);

        let sample = TrainingSample {
            stm_features: vec![0, 10, 20],
            nstm_features: vec![5, 15],
            target: 0.7,
        };

        let fwd = weights.forward(&sample.stm_features, &sample.nstm_features);
        assert!(fwd.sigmoid > 0.0 && fwd.sigmoid < 1.0);
        assert_eq!(fwd.hidden_raws.len(), 2);
        assert_eq!(fwd.hidden_activations.len(), 2);
        assert_eq!(fwd.hidden_raws[0].len(), 16);
        assert_eq!(fwd.hidden_raws[1].len(), 8);

        let mut grad = Gradients::new(config);
        weights.backward(&sample, &fwd, &mut grad);

        let has_nonzero = grad.output_weight.iter().any(|&g| g != 0.0);
        assert!(has_nonzero, "multi-layer gradients should be non-zero");
    }

    #[test]
    fn test_multi_layer_training_reduces_loss() {
        let config = multi_layer_config();
        let mut rng = SimpleRng::new(99);
        let mut weights = TrainableWeights::init_random(config, &mut rng);
        let mut state = AdamState::new(config);

        let sample = TrainingSample {
            stm_features: vec![1, 5, 10],
            nstm_features: vec![3, 7],
            target: 1.0,
        };

        let fwd_before = weights.forward(&sample.stm_features, &sample.nstm_features);
        let loss_before = -sample.target * fwd_before.sigmoid.ln()
            - (1.0 - sample.target) * (1.0 - fwd_before.sigmoid).ln();

        for _ in 0..200 {
            let mut grad = Gradients::new(config);
            let fwd = weights.forward(&sample.stm_features, &sample.nstm_features);
            weights.backward(&sample, &fwd, &mut grad);
            weights.adam_update(&grad, &mut state, 0.01, 1.0);
        }

        let fwd_after = weights.forward(&sample.stm_features, &sample.nstm_features);
        let loss_after = -sample.target * fwd_after.sigmoid.ln()
            - (1.0 - sample.target) * (1.0 - fwd_after.sigmoid).ln();

        assert!(
            loss_after < loss_before,
            "multi-layer loss should decrease: {loss_before} -> {loss_after}"
        );
    }

    #[test]
    fn test_screlu_forward_backward() {
        let config = screlu_config();
        let mut rng = SimpleRng::new(42);
        let weights = TrainableWeights::init_random(config, &mut rng);

        let sample = TrainingSample {
            stm_features: vec![0, 10, 20],
            nstm_features: vec![5, 15],
            target: 0.5,
        };

        let fwd = weights.forward(&sample.stm_features, &sample.nstm_features);
        assert!(fwd.sigmoid > 0.0 && fwd.sigmoid < 1.0);

        let mut grad = Gradients::new(config);
        weights.backward(&sample, &fwd, &mut grad);

        let has_nonzero = grad.output_weight.iter().any(|&g| g != 0.0);
        assert!(has_nonzero, "SCReLU gradients should be non-zero");
    }

    #[test]
    fn test_screlu_training_reduces_loss() {
        let config = screlu_config();
        let mut rng = SimpleRng::new(55);
        let mut weights = TrainableWeights::init_random(config, &mut rng);
        let mut state = AdamState::new(config);

        let sample = TrainingSample {
            stm_features: vec![1, 5, 10],
            nstm_features: vec![3, 7],
            target: 1.0,
        };

        let fwd_before = weights.forward(&sample.stm_features, &sample.nstm_features);
        let loss_before = -sample.target * fwd_before.sigmoid.ln()
            - (1.0 - sample.target) * (1.0 - fwd_before.sigmoid).ln();

        for _ in 0..200 {
            let mut grad = Gradients::new(config);
            let fwd = weights.forward(&sample.stm_features, &sample.nstm_features);
            weights.backward(&sample, &fwd, &mut grad);
            weights.adam_update(&grad, &mut state, 0.01, 1.0);
        }

        let fwd_after = weights.forward(&sample.stm_features, &sample.nstm_features);
        let loss_after = -sample.target * fwd_after.sigmoid.ln()
            - (1.0 - sample.target) * (1.0 - fwd_after.sigmoid).ln();

        assert!(
            loss_after < loss_before,
            "SCReLU loss should decrease: {loss_before} -> {loss_after}"
        );
    }

    #[test]
    fn test_multi_layer_quantize_roundtrip() {
        let config = multi_layer_config();
        let mut rng = SimpleRng::new(88);
        let weights = TrainableWeights::init_random(config, &mut rng);
        let quantized = weights.quantize();

        assert_eq!(quantized.hidden_weights.len(), 2);
        assert_eq!(quantized.hidden_biases.len(), 2);

        let has_nonzero = quantized.hidden_weights[1]
            .iter()
            .any(|&v| v != 0);
        assert!(has_nonzero, "second hidden layer should have non-zero quantized weights");
    }
}
