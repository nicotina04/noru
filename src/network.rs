/// NNUE 순전파 및 증분 업데이트 (범용)
///
/// 네트워크 구조:
///   Input (sparse) → Accumulator → Hidden → Output (1)
///
/// 모든 차원은 NnueConfig로 런타임 설정 가능.

use crate::config::NnueConfig;
use crate::quant::{clipped_relu, saturate_i16, ACTIVATION_SCALE, OUTPUT_SCALE};

pub const CLIPPED_RELU_MAX: i16 = 127;

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
    /// Accumulator → Hidden 가중치 [accumulator_size*2][hidden_size]
    pub hidden_weights: Vec<Vec<i16>>,
    /// Hidden 바이어스 [hidden_size]
    pub hidden_bias: Vec<i16>,
    /// Hidden → Output 가중치 [hidden_size]
    pub output_weights: Vec<i16>,
    /// Output 바이어스
    pub output_bias: i16,
}

impl NnueWeights {
    /// 빈 가중치 생성
    pub fn zeros(config: NnueConfig) -> Self {
        let acc = config.accumulator_size;
        let hid = config.hidden_size;
        Self {
            config,
            feature_weights: vec![vec![0i16; acc]; config.feature_size],
            feature_bias: vec![0i16; acc],
            hidden_weights: vec![vec![0i16; hid]; acc * 2],
            hidden_bias: vec![0i16; hid],
            output_weights: vec![0i16; hid],
            output_bias: 0,
        }
    }

    /// .bin 파일에서 가중치 로드
    pub fn load_from_bytes(data: &[u8], config: NnueConfig) -> Result<Self, &'static str> {
        let mut weights = Self::zeros(config);
        let mut cursor = 0;
        let acc = config.accumulator_size;
        let hid = config.hidden_size;

        let read_i16 =
            |cursor: &mut usize, count: usize, data: &[u8]| -> Result<Vec<i16>, &'static str> {
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
            weights.feature_weights[i] = read_i16(&mut cursor, acc, data)?;
        }
        // Feature bias
        weights.feature_bias = read_i16(&mut cursor, acc, data)?;
        // Hidden weights
        for i in 0..acc * 2 {
            weights.hidden_weights[i] = read_i16(&mut cursor, hid, data)?;
        }
        // Hidden bias
        weights.hidden_bias = read_i16(&mut cursor, hid, data)?;
        // Output weights
        weights.output_weights = read_i16(&mut cursor, hid, data)?;
        // Output bias
        let obias = read_i16(&mut cursor, 1, data)?;
        weights.output_bias = obias[0];

        Ok(weights)
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
    // 추가했던 건 빼고
    for i in 0..delta.num_added {
        let w = &weights.feature_weights[delta.added[i]];
        for j in 0..size {
            acc[j] = saturate_i16(acc[j] as i32 - w[j] as i32);
        }
    }
    // 제거했던 건 다시 더함
    for i in 0..delta.num_removed {
        let w = &weights.feature_weights[delta.removed[i]];
        for j in 0..size {
            acc[j] = saturate_i16(acc[j] as i32 + w[j] as i32);
        }
    }
}

/// NNUE 순전파: Accumulator → Hidden → Output
pub fn forward(acc: &Accumulator, weights: &NnueWeights) -> i32 {
    let acc_size = weights.config.accumulator_size;
    let hid_size = weights.config.hidden_size;

    // 1. ClippedReLU on accumulator (양쪽 관점 concat)
    let mut clipped = vec![0i16; acc_size * 2];
    for i in 0..acc_size {
        clipped[i] = clipped_relu(acc.stm[i], CLIPPED_RELU_MAX);
    }
    for i in 0..acc_size {
        clipped[acc_size + i] = clipped_relu(acc.nstm[i], CLIPPED_RELU_MAX);
    }

    // 2. Hidden layer
    let mut hidden = vec![0i16; hid_size];
    for j in 0..hid_size {
        let mut sum: i32 = weights.hidden_bias[j] as i32 * ACTIVATION_SCALE;
        for i in 0..acc_size * 2 {
            sum += clipped[i] as i32 * weights.hidden_weights[i][j] as i32;
        }
        hidden[j] = clipped_relu(saturate_i16(sum / ACTIVATION_SCALE), CLIPPED_RELU_MAX);
    }

    // 3. Output layer
    let mut output: i32 = weights.output_bias as i32 * OUTPUT_SCALE;
    for j in 0..hid_size {
        output += hidden[j] as i32 * weights.output_weights[j] as i32;
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
            hidden_size: 64,
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

        // refresh
        let mut acc_refresh = Accumulator::new(&weights.feature_bias);
        acc_refresh.refresh(&weights, &[0, 100], &[1]);

        // incremental
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
}
