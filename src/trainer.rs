/// NNUE 역전파 트레이너 (범용)
///
/// 학습은 FP32로 수행, 추론 시 i16으로 양자화.
/// 모든 차원은 NnueConfig로 런타임 설정 가능.

use crate::config::NnueConfig;
use crate::network::NnueWeights;
use crate::quant::WEIGHT_SCALE;

const CRELU_MAX: f32 = 1.0;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPSILON: f32 = 1e-8;

/// 학습용 FP32 가중치
#[derive(Clone)]
pub struct TrainableWeights {
    pub config: NnueConfig,
    pub ft_weight: Vec<Vec<f32>>,    // [feature_size][accumulator_size]
    pub ft_bias: Vec<f32>,           // [accumulator_size]
    pub hidden_weight: Vec<Vec<f32>>,// [accumulator_size*2][hidden_size]
    pub hidden_bias: Vec<f32>,       // [hidden_size]
    pub output_weight: Vec<f32>,     // [hidden_size]
    pub output_bias: f32,
}

/// Adam 옵티마이저 상태
#[derive(Clone)]
pub struct AdamState {
    ft_weight_m: Vec<Vec<f32>>,
    ft_weight_v: Vec<Vec<f32>>,
    ft_bias_m: Vec<f32>,
    ft_bias_v: Vec<f32>,
    hidden_weight_m: Vec<Vec<f32>>,
    hidden_weight_v: Vec<Vec<f32>>,
    hidden_bias_m: Vec<f32>,
    hidden_bias_v: Vec<f32>,
    output_weight_m: Vec<f32>,
    output_weight_v: Vec<f32>,
    output_bias_m: f32,
    output_bias_v: f32,
    pub step: u32,
}

impl AdamState {
    pub fn new(config: NnueConfig) -> Self {
        let acc = config.accumulator_size;
        let hid = config.hidden_size;
        Self {
            ft_weight_m: vec![vec![0.0; acc]; config.feature_size],
            ft_weight_v: vec![vec![0.0; acc]; config.feature_size],
            ft_bias_m: vec![0.0; acc],
            ft_bias_v: vec![0.0; acc],
            hidden_weight_m: vec![vec![0.0; hid]; acc * 2],
            hidden_weight_v: vec![vec![0.0; hid]; acc * 2],
            hidden_bias_m: vec![0.0; hid],
            hidden_bias_v: vec![0.0; hid],
            output_weight_m: vec![0.0; hid],
            output_weight_v: vec![0.0; hid],
            output_bias_m: 0.0,
            output_bias_v: 0.0,
            step: 0,
        }
    }
}

/// 그래디언트 버퍼
pub struct Gradients {
    pub config: NnueConfig,
    pub ft_weight: Vec<Vec<f32>>,
    pub ft_bias: Vec<f32>,
    pub hidden_weight: Vec<Vec<f32>>,
    pub hidden_bias: Vec<f32>,
    pub output_weight: Vec<f32>,
    pub output_bias: f32,
}

impl Gradients {
    pub fn new(config: NnueConfig) -> Self {
        let acc = config.accumulator_size;
        let hid = config.hidden_size;
        Self {
            config,
            ft_weight: vec![vec![0.0; acc]; config.feature_size],
            ft_bias: vec![0.0; acc],
            hidden_weight: vec![vec![0.0; hid]; acc * 2],
            hidden_bias: vec![0.0; hid],
            output_weight: vec![0.0; hid],
            output_bias: 0.0,
        }
    }

    pub fn zero(&mut self) {
        let acc = self.config.accumulator_size;
        let hid = self.config.hidden_size;
        for row in self.ft_weight.iter_mut() {
            for v in row.iter_mut() {
                *v = 0.0;
            }
        }
        self.ft_bias.iter_mut().for_each(|v| *v = 0.0);
        for row in self.hidden_weight.iter_mut() {
            for v in row.iter_mut() {
                *v = 0.0;
            }
        }
        self.hidden_bias.iter_mut().for_each(|v| *v = 0.0);
        self.output_weight.iter_mut().for_each(|v| *v = 0.0);
        self.output_bias = 0.0;
        let _ = (acc, hid); // suppress unused
    }
}

/// 학습 샘플
pub struct TrainingSample {
    pub stm_features: Vec<usize>,
    pub nstm_features: Vec<usize>,
    pub target: f32,
}

/// 순전파 중간 결과
pub struct ForwardResult {
    pub acc_stm: Vec<f32>,
    pub acc_nstm: Vec<f32>,
    pub crelu: Vec<f32>,
    pub hidden_raw: Vec<f32>,
    pub hidden_crelu: Vec<f32>,
    pub output: f32,
    pub sigmoid: f32,
}

impl TrainableWeights {
    /// Kaiming 초기화
    pub fn init_random(config: NnueConfig, rng: &mut SimpleRng) -> Self {
        let acc = config.accumulator_size;
        let hid = config.hidden_size;
        let ft_scale = (2.0 / acc as f32).sqrt() * 0.1;
        let hidden_scale = (2.0 / hid as f32).sqrt();
        let output_scale = (1.0 / hid as f32).sqrt();

        let mut w = Self {
            config,
            ft_weight: vec![vec![0.0; acc]; config.feature_size],
            ft_bias: vec![0.0; acc],
            hidden_weight: vec![vec![0.0; hid]; acc * 2],
            hidden_bias: vec![0.0; hid],
            output_weight: vec![0.0; hid],
            output_bias: 0.0,
        };

        for row in w.ft_weight.iter_mut() {
            for v in row.iter_mut() {
                *v = rng.next_normal() * ft_scale;
            }
        }
        for row in w.hidden_weight.iter_mut() {
            for v in row.iter_mut() {
                *v = rng.next_normal() * hidden_scale;
            }
        }
        for v in w.output_weight.iter_mut() {
            *v = rng.next_normal() * output_scale;
        }

        w
    }

    /// 순전파 (학습용 FP32)
    pub fn forward(&self, stm_features: &[usize], nstm_features: &[usize]) -> ForwardResult {
        let acc = self.config.accumulator_size;
        let hid = self.config.hidden_size;

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

        // ClippedReLU
        let mut crelu = vec![0.0f32; acc * 2];
        for i in 0..acc {
            crelu[i] = acc_stm[i].max(0.0).min(CRELU_MAX);
        }
        for i in 0..acc {
            crelu[acc + i] = acc_nstm[i].max(0.0).min(CRELU_MAX);
        }

        // Hidden layer
        let mut hidden_raw = vec![0.0f32; hid];
        for j in 0..hid {
            let mut sum = self.hidden_bias[j];
            for i in 0..acc * 2 {
                sum += crelu[i] * self.hidden_weight[i][j];
            }
            hidden_raw[j] = sum;
        }

        let mut hidden_crelu = vec![0.0f32; hid];
        for j in 0..hid {
            hidden_crelu[j] = hidden_raw[j].max(0.0).min(CRELU_MAX);
        }

        // Output
        let mut output = self.output_bias;
        for j in 0..hid {
            output += hidden_crelu[j] * self.output_weight[j];
        }

        let sigmoid = 1.0 / (1.0 + (-output).exp());

        ForwardResult {
            acc_stm,
            acc_nstm,
            crelu,
            hidden_raw,
            hidden_crelu,
            output,
            sigmoid,
        }
    }

    /// 역전파 (BCE: sigmoid - target)
    pub fn backward(&self, sample: &TrainingSample, fwd: &ForwardResult, grad: &mut Gradients) {
        let d_output = fwd.sigmoid - sample.target;
        self.backward_inner(d_output, sample, fwd, grad);
    }

    /// 역전파 (MSE: output - target, sigmoid 없이 선형 출력)
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
        let hid = self.config.hidden_size;

        // Output layer gradients
        grad.output_bias += d_output;
        for j in 0..hid {
            grad.output_weight[j] += d_output * fwd.hidden_crelu[j];
        }

        // Hidden ClippedReLU backprop
        let mut d_hidden = vec![0.0f32; hid];
        for j in 0..hid {
            let d_crelu = d_output * self.output_weight[j];
            d_hidden[j] = if fwd.hidden_raw[j] > 0.0 && fwd.hidden_raw[j] < CRELU_MAX {
                d_crelu
            } else {
                0.0
            };
        }

        // Hidden layer gradients
        for j in 0..hid {
            grad.hidden_bias[j] += d_hidden[j];
        }
        for i in 0..acc * 2 {
            for j in 0..hid {
                grad.hidden_weight[i][j] += d_hidden[j] * fwd.crelu[i];
            }
        }

        // Accumulator ClippedReLU backprop
        let mut d_acc = vec![0.0f32; acc * 2];
        for i in 0..acc * 2 {
            let mut sum = 0.0f32;
            for j in 0..hid {
                sum += d_hidden[j] * self.hidden_weight[i][j];
            }
            let acc_val = if i < acc {
                fwd.acc_stm[i]
            } else {
                fwd.acc_nstm[i - acc]
            };
            d_acc[i] = if acc_val > 0.0 && acc_val < CRELU_MAX {
                sum
            } else {
                0.0
            };
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

    /// Adam 옵티마이저
    pub fn adam_update(
        &mut self,
        grad: &Gradients,
        state: &mut AdamState,
        lr: f32,
        batch_size: f32,
    ) {
        let acc = self.config.accumulator_size;
        let hid = self.config.hidden_size;

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

        // Hidden weights
        for i in 0..acc * 2 {
            for j in 0..hid {
                let g = grad.hidden_weight[i][j] * scale;
                adam_step(
                    &mut self.hidden_weight[i][j],
                    g,
                    &mut state.hidden_weight_m[i][j],
                    &mut state.hidden_weight_v[i][j],
                    lr,
                    bc1,
                    bc2,
                );
            }
        }

        // Hidden bias
        for j in 0..hid {
            let g = grad.hidden_bias[j] * scale;
            adam_step(
                &mut self.hidden_bias[j],
                g,
                &mut state.hidden_bias_m[j],
                &mut state.hidden_bias_v[j],
                lr,
                bc1,
                bc2,
            );
        }

        // Output weights
        for j in 0..hid {
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

    /// FP32 → i16 양자화
    pub fn quantize(&self) -> NnueWeights {
        let acc = self.config.accumulator_size;
        let hid = self.config.hidden_size;
        let scale = WEIGHT_SCALE as f32;

        let mut weights = NnueWeights::zeros(self.config);

        for (feat, row) in self.ft_weight.iter().enumerate() {
            for i in 0..acc {
                weights.feature_weights[feat][i] = (row[i] * scale).round() as i16;
            }
        }
        for i in 0..acc {
            weights.feature_bias[i] = (self.ft_bias[i] * scale).round() as i16;
        }
        for i in 0..acc * 2 {
            for j in 0..hid {
                weights.hidden_weights[i][j] =
                    (self.hidden_weight[i][j] * scale).round() as i16;
            }
        }
        for j in 0..hid {
            weights.hidden_bias[j] = (self.hidden_bias[j] * scale).round() as i16;
        }
        for j in 0..hid {
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

/// 간단한 난수 생성기 (xorshift64)
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
            hidden_size: 64,
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
}
