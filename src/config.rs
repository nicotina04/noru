/// NNUE 네트워크 런타임 설정
///
/// 게임에 따라 피처 크기, accumulator 크기, hidden 레이어 구성을 다르게 설정 가능.

/// 활성화 함수 종류
///
/// Accumulator 출력(첫 번째 레이어)에 적용되며, 이후 hidden 레이어는 항상 CReLU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Clipped ReLU: clamp(x, 0, max)
    CReLU,
    /// Squared Clipped ReLU: clamp(x, 0, max)²
    SCReLU,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueConfig {
    /// 입력 피처 총 크기
    pub feature_size: usize,
    /// Accumulator 뉴런 수 (각 관점별)
    pub accumulator_size: usize,
    /// Hidden 레이어 크기 배열 (예: &[64] 또는 &[256, 32, 32])
    pub hidden_sizes: &'static [usize],
    /// 활성화 함수 (accumulator 출력에만 적용)
    pub activation: Activation,
}

impl NnueConfig {
    /// STM + NSTM 합쳐진 accumulator 크기
    #[inline]
    pub fn concat_size(&self) -> usize {
        self.accumulator_size * 2
    }

    /// Hidden 레이어 수
    #[inline]
    pub fn num_hidden_layers(&self) -> usize {
        self.hidden_sizes.len()
    }

    /// 특정 hidden 레이어의 입력 크기
    #[inline]
    pub fn layer_input_size(&self, layer_idx: usize) -> usize {
        if layer_idx == 0 {
            self.concat_size()
        } else {
            self.hidden_sizes[layer_idx - 1]
        }
    }

    /// 마지막 hidden 레이어의 출력 크기 (= output 레이어의 입력 크기)
    #[inline]
    pub fn last_hidden_size(&self) -> usize {
        *self.hidden_sizes.last().expect("hidden_sizes must not be empty")
    }
}
