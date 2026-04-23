//! Quantization audit utilities.
//!
//! These helpers compare FP32 training weights against i16 deployment weights
//! over a sample set and summarize the observed drift.

use crate::network::{forward, Accumulator, NnueWeights};
use crate::trainer::{TrainableWeights, TrainingSample};
use std::error::Error;
use std::fmt;

/// Feature lists used by quantization audit helpers.
pub trait FeatureSet {
    fn stm_features(&self) -> &[usize];
    fn nstm_features(&self) -> &[usize];
}

/// Borrowed feature lists for audit-only evaluation.
#[derive(Debug, Clone, Copy)]
pub struct AuditSample<'a> {
    pub stm_features: &'a [usize],
    pub nstm_features: &'a [usize],
}

impl FeatureSet for TrainingSample {
    fn stm_features(&self) -> &[usize] {
        &self.stm_features
    }

    fn nstm_features(&self) -> &[usize] {
        &self.nstm_features
    }
}

impl FeatureSet for AuditSample<'_> {
    fn stm_features(&self) -> &[usize] {
        self.stm_features
    }

    fn nstm_features(&self) -> &[usize] {
        self.nstm_features
    }
}

impl<'a> FeatureSet for (&'a [usize], &'a [usize]) {
    fn stm_features(&self) -> &[usize] {
        self.0
    }

    fn nstm_features(&self) -> &[usize] {
        self.1
    }
}

/// Aggregate absolute-error metrics.
#[derive(Debug, Clone, Copy, Default)]
pub struct ErrorSummary {
    pub mean_abs: f32,
    pub max_abs: f32,
    pub rmse: f32,
}

impl ErrorSummary {
    fn from_totals(total_abs: f32, max_abs: f32, total_sq: f32, sample_count: usize) -> Self {
        let n = sample_count as f32;
        Self {
            mean_abs: total_abs / n,
            max_abs,
            rmse: (total_sq / n).sqrt(),
        }
    }
}

/// Quantization drift summary across a sample set.
#[derive(Debug, Clone, Copy)]
pub struct QuantizationReport {
    pub sample_count: usize,
    pub sign_agreement: usize,
    pub sign_agreement_ratio: f32,
    /// Least-squares multiplier mapping FP32 raw outputs to the i16 score space.
    pub inferred_output_scale: f32,
    pub model_bytes: usize,
    pub fp32_output_min: f32,
    pub fp32_output_max: f32,
    pub i16_output_min: i32,
    pub i16_output_max: i32,
    pub raw_error: ErrorSummary,
    pub probability_error: ErrorSummary,
}

/// Errors returned by quantization audit helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationAuditError {
    EmptySampleSet,
}

impl fmt::Display for QuantizationAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySampleSet => write!(f, "quantization audit requires at least one sample"),
        }
    }
}

impl Error for QuantizationAuditError {}

/// Compare FP32 weights against a quantized model over a sample set.
pub fn audit_quantized_model<T: FeatureSet>(
    fp32: &TrainableWeights,
    quantized: &NnueWeights,
    positions: &[T],
) -> Result<QuantizationReport, QuantizationAuditError> {
    if positions.is_empty() {
        return Err(QuantizationAuditError::EmptySampleSet);
    }

    let sample_count = positions.len();
    let mut fp32_outputs = Vec::with_capacity(sample_count);
    let mut i16_outputs = Vec::with_capacity(sample_count);
    let mut sign_agreement = 0usize;
    let mut fp32_output_min = f32::INFINITY;
    let mut fp32_output_max = f32::NEG_INFINITY;
    let mut i16_output_min = i32::MAX;
    let mut i16_output_max = i32::MIN;

    for position in positions {
        let stm = position.stm_features();
        let nstm = position.nstm_features();

        let fp32_output = fp32.forward(stm, nstm).output;
        let mut acc = Accumulator::new(&quantized.feature_bias);
        acc.refresh(quantized, stm, nstm);
        let i16_output = forward(&acc, quantized);

        if (fp32_output > 0.0) == (i16_output > 0) {
            sign_agreement += 1;
        }

        fp32_output_min = fp32_output_min.min(fp32_output);
        fp32_output_max = fp32_output_max.max(fp32_output);
        i16_output_min = i16_output_min.min(i16_output);
        i16_output_max = i16_output_max.max(i16_output);
        fp32_outputs.push(fp32_output);
        i16_outputs.push(i16_output);
    }

    let mut scale_numerator = 0.0f32;
    let mut scale_denominator = 0.0f32;
    for (&fp32_output, &i16_output) in fp32_outputs.iter().zip(&i16_outputs) {
        scale_numerator += fp32_output * i16_output as f32;
        scale_denominator += fp32_output * fp32_output;
    }

    let mut inferred_output_scale = if scale_denominator > 1e-12 {
        scale_numerator / scale_denominator
    } else {
        1.0
    };
    if !inferred_output_scale.is_finite() || inferred_output_scale <= 0.0 {
        inferred_output_scale = 1.0;
    }

    let mut raw_total_abs = 0.0f32;
    let mut raw_max_abs = 0.0f32;
    let mut raw_total_sq = 0.0f32;
    let mut prob_total_abs = 0.0f32;
    let mut prob_max_abs = 0.0f32;
    let mut prob_total_sq = 0.0f32;
    let inverse_scale = 1.0 / inferred_output_scale;

    for (&fp32_output, &i16_output) in fp32_outputs.iter().zip(&i16_outputs) {
        let i16_output_f32 = i16_output as f32;
        let raw_delta = fp32_output * inferred_output_scale - i16_output_f32;
        let raw_abs = raw_delta.abs();
        raw_total_abs += raw_abs;
        raw_max_abs = raw_max_abs.max(raw_abs);
        raw_total_sq += raw_delta * raw_delta;

        let fp32_probability = sigmoid(fp32_output);
        let i16_probability = sigmoid(i16_output_f32 * inverse_scale);
        let prob_delta = fp32_probability - i16_probability;
        let prob_abs = prob_delta.abs();
        prob_total_abs += prob_abs;
        prob_max_abs = prob_max_abs.max(prob_abs);
        prob_total_sq += prob_delta * prob_delta;
    }

    Ok(QuantizationReport {
        sample_count,
        sign_agreement,
        sign_agreement_ratio: sign_agreement as f32 / sample_count as f32,
        inferred_output_scale,
        model_bytes: quantized.save_to_bytes().len(),
        fp32_output_min,
        fp32_output_max,
        i16_output_min,
        i16_output_max,
        raw_error: ErrorSummary::from_totals(
            raw_total_abs,
            raw_max_abs,
            raw_total_sq,
            sample_count,
        ),
        probability_error: ErrorSummary::from_totals(
            prob_total_abs,
            prob_max_abs,
            prob_total_sq,
            sample_count,
        ),
    })
}

impl TrainableWeights {
    /// Quantize these FP32 weights and audit the resulting deployment drift.
    pub fn audit_quantization<T: FeatureSet>(
        &self,
        positions: &[T],
    ) -> Result<QuantizationReport, QuantizationAuditError> {
        let quantized = self.quantize();
        audit_quantized_model(self, &quantized, positions)
    }
}

impl NnueWeights {
    /// Audit this quantized model against its FP32 source weights.
    pub fn audit_against_fp32<T: FeatureSet>(
        &self,
        fp32: &TrainableWeights,
        positions: &[T],
    ) -> Result<QuantizationReport, QuantizationAuditError> {
        audit_quantized_model(fp32, self, positions)
    }
}

fn sigmoid(raw: f32) -> f32 {
    if raw >= 0.0 {
        1.0 / (1.0 + (-raw).exp())
    } else {
        let exp = raw.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Activation, NnueConfig};

    fn tiny_config() -> NnueConfig {
        NnueConfig {
            feature_size: 2,
            accumulator_size: 1,
            hidden_sizes: &[1],
            activation: Activation::CReLU,
        }
    }

    fn tiny_weights() -> TrainableWeights {
        let config = tiny_config();
        TrainableWeights {
            config,
            ft_weight: vec![vec![1.0], vec![1.0]],
            ft_bias: vec![0.0],
            hidden_weights: vec![vec![vec![1.0], vec![-1.0]]],
            hidden_biases: vec![vec![0.0]],
            output_weight: vec![1.0],
            output_bias: -0.5,
        }
    }

    fn tiny_positions<'a>() -> [AuditSample<'a>; 4] {
        static EMPTY: [usize; 0] = [];
        static F0: [usize; 1] = [0];
        static F1: [usize; 1] = [1];
        [
            AuditSample {
                stm_features: &F0,
                nstm_features: &EMPTY,
            },
            AuditSample {
                stm_features: &EMPTY,
                nstm_features: &F0,
            },
            AuditSample {
                stm_features: &F0,
                nstm_features: &F1,
            },
            AuditSample {
                stm_features: &EMPTY,
                nstm_features: &EMPTY,
            },
        ]
    }

    #[test]
    fn audit_rejects_empty_sample_sets() {
        let weights = tiny_weights();
        let positions: [AuditSample<'_>; 0] = [];
        let err = weights.audit_quantization(&positions).unwrap_err();
        assert_eq!(err, QuantizationAuditError::EmptySampleSet);
    }

    #[test]
    fn audit_reports_sane_metrics() {
        let weights = tiny_weights();
        let positions = tiny_positions();

        let report = weights.audit_quantization(&positions).unwrap();

        assert_eq!(report.sample_count, positions.len());
        assert_eq!(report.sign_agreement, positions.len());
        assert_eq!(report.sign_agreement_ratio, 1.0);
        assert!(report.inferred_output_scale.is_finite());
        assert!(report.inferred_output_scale > 0.0);
        assert!(report.model_bytes > 0);
        assert!(report.fp32_output_min <= report.fp32_output_max);
        assert!(report.i16_output_min <= report.i16_output_max);
        assert!(report.raw_error.mean_abs.is_finite());
        assert!(report.raw_error.max_abs.is_finite());
        assert!(report.raw_error.rmse.is_finite());
        assert!(report.probability_error.mean_abs.is_finite());
        assert!(report.probability_error.max_abs.is_finite());
        assert!(report.probability_error.rmse.is_finite());
    }

    #[test]
    fn audit_reloaded_quantized_model_matches_in_memory_quantized_model() {
        let weights = tiny_weights();
        let positions = tiny_positions();
        let quantized = weights.quantize();
        let bytes = quantized.save_to_bytes();
        let reloaded = NnueWeights::load_from_bytes(&bytes, None).unwrap();

        let report_from_quantize = weights.audit_quantization(&positions).unwrap();
        let report_from_reload = reloaded.audit_against_fp32(&weights, &positions).unwrap();

        assert_eq!(
            report_from_reload.sample_count,
            report_from_quantize.sample_count
        );
        assert_eq!(
            report_from_reload.sign_agreement,
            report_from_quantize.sign_agreement
        );
        assert_eq!(report_from_reload.model_bytes, bytes.len());
        assert!(
            (report_from_reload.inferred_output_scale - report_from_quantize.inferred_output_scale)
                .abs()
                < 1e-6
        );
        assert!(
            (report_from_reload.raw_error.mean_abs - report_from_quantize.raw_error.mean_abs).abs()
                < 1e-6
        );
    }
}
