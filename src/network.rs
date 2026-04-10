/// NNUE 순전파 및 증분 업데이트 (범용)
///
/// 네트워크 구조:
///   Input (sparse) → Accumulator → Hidden₁ → ... → Hiddenₙ → Output (1)
///
/// 모든 차원은 NnueConfig로 런타임 설정 가능.

use crate::config::{Activation, NnueConfig};
use crate::quant::{clipped_relu, saturate_i16, ACTIVATION_SCALE, OUTPUT_SCALE};

pub const CLIPPED_RELU_MAX: i16 = 127;

// Binary format v2 magic number: "NORU" in little-endian
const MAGIC: u32 = 0x4E4F5255;
const FORMAT_VERSION: u32 = 2;

/// 증분 업데이트를 위한 피처 변경 정보
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

/// NNUE 가중치 (추론용 i16)
pub struct NnueWeights {
    pub config: NnueConfig,
    /// Feature → Accumulator 가중치 [feature_size][accumulator_size]
    pub feature_weights: Vec<Vec<i16>>,
    /// Feature → Accumulator 바이어스 [accumulator_size]
    pub feature_bias: Vec<i16>,
    /// Hidden 레이어 가중치 [num_layers][input_size][output_size]
    pub hidden_weights: Vec<Vec<Vec<i16>>>,
    /// Hidden 레이어 바이어스 [num_layers][output_size]
    pub hidden_biases: Vec<Vec<i16>>,
    /// Hidden → Output 가중치 [last_hidden_size]
    pub output_weights: Vec<i16>,
    /// Output 바이어스
    pub output_bias: i16,
}

impl NnueWeights {
    /// 빈 가중치 생성
    pub fn zeros(config: NnueConfig) -> Self {
        let num_layers = config.num_hidden_layers();

        let mut hidden_weights = Vec::with_capacity(num_layers);
        let mut hidden_biases = Vec::with_capacity(num_layers);

        for k in 0..num_layers {
            let in_size = config.layer_input_size(k);
            let out_size = config.hidden_sizes[k];
            hidden_weights.push(vec![vec![0i16; out_size]; in_size]);
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

    /// v2 바이너리 포맷으로 저장
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
        // Feature bias
        write_i16_slice(&mut buf, &self.feature_bias);

        // Hidden layers
        for k in 0..self.config.num_hidden_layers() {
            for row in &self.hidden_weights[k] {
                write_i16_slice(&mut buf, row);
            }
            write_i16_slice(&mut buf, &self.hidden_biases[k]);
        }

        // Output
        write_i16_slice(&mut buf, &self.output_weights);
        buf.extend_from_slice(&self.output_bias.to_le_bytes());

        buf
    }

    /// 바이너리에서 로드 (v2 헤더 자동 감지, 없으면 레거시)
    pub fn load_from_bytes(data: &[u8], legacy_config: Option<NnueConfig>) -> Result<Self, &'static str> {
        if data.len() >= 4 {
            let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            if magic == MAGIC {
                return Self::load_v2(data);
            }
        }

        // Legacy format: config 필수
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

        // hidden_sizes를 'static으로 만들기 위해 leak
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
        // Feature bias
        weights.feature_bias = read_i16(cursor, acc)?;

        // Hidden layers
        for k in 0..config.num_hidden_layers() {
            let in_size = config.layer_input_size(k);
            let out_size = config.hidden_sizes[k];
            for i in 0..in_size {
                weights.hidden_weights[k][i] = read_i16(cursor, out_size)?;
            }
            weights.hidden_biases[k] = read_i16(cursor, out_size)?;
        }

        // Output weights
        let last_hid = config.last_hidden_size();
        weights.output_weights = read_i16(cursor, last_hid)?;
        // Output bias
        let obias = read_i16(cursor, 1)?;
        weights.output_bias = obias[0];

        Ok(())
    }
}

/// Accumulator: 활성 피처 가중치의 합
#[derive(Clone)]
pub struct Accumulator {
    /// STM (현재 턴 플레이어) 관점
    pub stm: Vec<i16>,
    /// NSTM (상대 플레이어) 관점
    pub nstm: Vec<i16>,
}

impl Accumulator {
    pub fn new(bias: &[i16]) -> Self {
        Self {
            stm: bias.to_vec(),
            nstm: bias.to_vec(),
        }
    }

    /// 전체 재계산
    pub fn refresh(
        &mut self,
        weights: &NnueWeights,
        stm_features: &[usize],
        nstm_features: &[usize],
    ) {
        let acc = weights.config.accumulator_size;
        self.stm.copy_from_slice(&weights.feature_bias);
        self.nstm.copy_from_slice(&weights.feature_bias);

        for &feat in stm_features {
            let w = &weights.feature_weights[feat];
            for i in 0..acc {
                self.stm[i] = saturate_i16(self.stm[i] as i32 + w[i] as i32);
            }
        }

        for &feat in nstm_features {
            let w = &weights.feature_weights[feat];
            for i in 0..acc {
                self.nstm[i] = saturate_i16(self.nstm[i] as i32 + w[i] as i32);
            }
        }
    }

    /// 증분 업데이트
    pub fn update_incremental(
        &mut self,
        weights: &NnueWeights,
        stm_delta: &FeatureDelta,
        nstm_delta: &FeatureDelta,
    ) {
        apply_delta(&mut self.stm, weights, stm_delta);
        apply_delta(&mut self.nstm, weights, nstm_delta);
    }

    /// 증분 업데이트 취소 (Add/Remove 연산 반전)
    pub fn update_incremental_undo(
        &mut self,
        weights: &NnueWeights,
        stm_delta: &FeatureDelta,
        nstm_delta: &FeatureDelta,
    ) {
        apply_delta_reversed(&mut self.stm, weights, stm_delta);
        apply_delta_reversed(&mut self.nstm, weights, nstm_delta);
    }

    /// 관점 전환 (STM ↔ NSTM)
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.stm, &mut self.nstm);
    }
}

fn apply_delta(acc: &mut [i16], weights: &NnueWeights, delta: &FeatureDelta) {
    let size = acc.len();
    for i in 0..delta.num_removed {
        let w = &weights.feature_weights[delta.removed[i]];
        for j in 0..size {
            acc[j] = saturate_i16(acc[j] as i32 - w[j] as i32);
        }
    }
    for i in 0..delta.num_added {
        let w = &weights.feature_weights[delta.added[i]];
        for j in 0..size {
            acc[j] = saturate_i16(acc[j] as i32 + w[j] as i32);
        }
    }
}

fn apply_delta_reversed(acc: &mut [i16], weights: &NnueWeights, delta: &FeatureDelta) {
    let size = acc.len();
    for i in 0..delta.num_added {
        let w = &weights.feature_weights[delta.added[i]];
        for j in 0..size {
            acc[j] = saturate_i16(acc[j] as i32 - w[j] as i32);
        }
    }
    for i in 0..delta.num_removed {
        let w = &weights.feature_weights[delta.removed[i]];
        for j in 0..size {
            acc[j] = saturate_i16(acc[j] as i32 + w[j] as i32);
        }
    }
}

/// NNUE 순전파: Accumulator → Hidden₁ → ... → Hiddenₙ → Output
pub fn forward(acc: &Accumulator, weights: &NnueWeights) -> i32 {
    let config = weights.config;
    let acc_size = config.accumulator_size;
    let use_screlu = config.activation == Activation::SCReLU;

    // 1. Accumulator에 ClippedReLU 적용 (양쪽 관점 concat)
    let mut prev = vec![0i16; acc_size * 2];
    for i in 0..acc_size {
        prev[i] = clipped_relu(acc.stm[i], CLIPPED_RELU_MAX);
    }
    for i in 0..acc_size {
        prev[acc_size + i] = clipped_relu(acc.nstm[i], CLIPPED_RELU_MAX);
    }

    // 2. Hidden 레이어 순전파
    for k in 0..config.num_hidden_layers() {
        let out_size = config.hidden_sizes[k];
        let mut next = vec![0i16; out_size];

        if k == 0 && use_screlu {
            // SCReLU: clamped² * weight, i64 누적으로 오버플로 방지
            for j in 0..out_size {
                let mut sum: i64 =
                    weights.hidden_biases[k][j] as i64 * ACTIVATION_SCALE as i64;
                for i in 0..prev.len() {
                    sum += prev[i] as i64 * prev[i] as i64
                        * weights.hidden_weights[k][i][j] as i64;
                }
                // SCReLU 정규화: CLIPPED_RELU_MAX(127)로 나눠 스케일 복원
                let scaled = (sum / CLIPPED_RELU_MAX as i64 / ACTIVATION_SCALE as i64) as i32;
                next[j] = clipped_relu(saturate_i16(scaled), CLIPPED_RELU_MAX);
            }
        } else {
            // CReLU: 기존과 동일
            for j in 0..out_size {
                let mut sum: i32 =
                    weights.hidden_biases[k][j] as i32 * ACTIVATION_SCALE;
                for i in 0..prev.len() {
                    sum += prev[i] as i32 * weights.hidden_weights[k][i][j] as i32;
                }
                next[j] = clipped_relu(saturate_i16(sum / ACTIVATION_SCALE), CLIPPED_RELU_MAX);
            }
        }

        prev = next;
    }

    // 3. Output 레이어
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
        // 일부 가중치 설정
        weights.feature_weights[0][0] = 42;
        weights.hidden_weights[0][0][0] = 7;
        weights.hidden_weights[1][0][0] = -3;
        weights.output_weights[0] = 10;
        weights.output_bias = 5;

        let bytes = weights.save_to_bytes();
        let loaded = NnueWeights::load_from_bytes(&bytes, None).unwrap();

        assert_eq!(loaded.config, config);
        assert_eq!(loaded.feature_weights[0][0], 42);
        assert_eq!(loaded.hidden_weights[0][0][0], 7);
        assert_eq!(loaded.hidden_weights[1][0][0], -3);
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

        // 같은 가중치로 두 네트워크 생성
        let mut w_crelu = NnueWeights::zeros(crelu_config);
        let mut w_screlu = NnueWeights::zeros(screlu_config);

        // 비-제로 가중치 설정
        for i in 0..8 {
            w_crelu.feature_bias[i] = 50;
            w_screlu.feature_bias[i] = 50;
        }
        for i in 0..16 {
            for j in 0..4 {
                w_crelu.hidden_weights[0][i][j] = 10;
                w_screlu.hidden_weights[0][i][j] = 10;
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

        // SCReLU squaring이 다른 값을 생성해야 함
        assert_ne!(eval_crelu, eval_screlu);
    }
}
