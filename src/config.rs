/// NNUE 네트워크 런타임 설정
///
/// 게임에 따라 피처 크기, accumulator 크기, hidden 크기를 다르게 설정 가능.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueConfig {
    /// 입력 피처 총 크기 (예: 오목 530, 헥스 전투 TBD)
    pub feature_size: usize,
    /// Accumulator 뉴런 수 (각 관점별)
    pub accumulator_size: usize,
    /// Hidden 레이어 뉴런 수
    pub hidden_size: usize,
}

impl NnueConfig {
    /// STM + NSTM 합쳐진 accumulator 크기
    #[inline]
    pub fn concat_size(&self) -> usize {
        self.accumulator_size * 2
    }
}
