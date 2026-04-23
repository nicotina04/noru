/// NNUE network runtime configuration.
///
/// Configure feature size, accumulator size, and hidden layer layout per game.
use std::borrow::Cow;

/// Activation function type.
///
/// Applied to accumulator output (first layer); subsequent hidden layers always use CReLU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Clipped ReLU: clamp(x, 0, max)
    CReLU,
    /// Squared Clipped ReLU: clamp(x, 0, max)²
    SCReLU,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NnueConfig {
    /// Total input feature size
    pub feature_size: usize,
    /// Accumulator neurons per perspective
    pub accumulator_size: usize,
    /// Hidden layer sizes (e.g. `&[64]` or `vec![256, 32, 32]`)
    pub hidden_sizes: Cow<'static, [usize]>,
    /// Activation function (applied to accumulator output only)
    pub activation: Activation,
}

impl NnueConfig {
    /// Construct a config backed by a compile-time topology slice.
    pub const fn new_static(
        feature_size: usize,
        accumulator_size: usize,
        hidden_sizes: &'static [usize],
        activation: Activation,
    ) -> Self {
        Self {
            feature_size,
            accumulator_size,
            hidden_sizes: Cow::Borrowed(hidden_sizes),
            activation,
        }
    }

    /// Construct a config that owns its topology at runtime.
    pub fn new_owned(
        feature_size: usize,
        accumulator_size: usize,
        hidden_sizes: Vec<usize>,
        activation: Activation,
    ) -> Self {
        Self {
            feature_size,
            accumulator_size,
            hidden_sizes: Cow::Owned(hidden_sizes),
            activation,
        }
    }

    /// Combined accumulator size (STM + NSTM)
    #[inline]
    pub fn concat_size(&self) -> usize {
        self.accumulator_size * 2
    }

    /// Number of hidden layers
    #[inline]
    pub fn num_hidden_layers(&self) -> usize {
        self.hidden_sizes.len()
    }

    /// Input size for a specific hidden layer
    #[inline]
    pub fn layer_input_size(&self, layer_idx: usize) -> usize {
        if layer_idx == 0 {
            self.concat_size()
        } else {
            self.hidden_sizes[layer_idx - 1]
        }
    }

    /// Output size of the last hidden layer (= input size of the output layer)
    #[inline]
    pub fn last_hidden_size(&self) -> usize {
        *self
            .hidden_sizes
            .last()
            .expect("hidden_sizes must not be empty")
    }
}

/// Owned, runtime-constructible variant of [`NnueConfig`].
///
/// [`NnueConfig`] can either borrow a compile-time topology or own a runtime
/// one. When the topology must be built dynamically (FFI bindings, config
/// files, GUI editors), use `OwnedNnueConfig` and convert via
/// [`OwnedNnueConfig::into_config`] or `Into<NnueConfig>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedNnueConfig {
    pub feature_size: usize,
    pub accumulator_size: usize,
    pub hidden_sizes: Vec<usize>,
    pub activation: Activation,
}

impl OwnedNnueConfig {
    pub fn new(
        feature_size: usize,
        accumulator_size: usize,
        hidden_sizes: Vec<usize>,
        activation: Activation,
    ) -> Self {
        Self {
            feature_size,
            accumulator_size,
            hidden_sizes,
            activation,
        }
    }

    /// Consume `self` and produce a [`NnueConfig`] that owns `hidden_sizes`.
    pub fn into_config(self) -> NnueConfig {
        NnueConfig::new_owned(
            self.feature_size,
            self.accumulator_size,
            self.hidden_sizes,
            self.activation,
        )
    }
}

impl From<OwnedNnueConfig> for NnueConfig {
    fn from(value: OwnedNnueConfig) -> Self {
        value.into_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_config_into_config_preserves_fields() {
        let owned = OwnedNnueConfig::new(768, 256, vec![256, 32, 32], Activation::SCReLU);
        let config = owned.into_config();
        assert_eq!(config.feature_size, 768);
        assert_eq!(config.accumulator_size, 256);
        assert_eq!(config.hidden_sizes.as_ref(), &[256, 32, 32]);
        assert_eq!(config.activation, Activation::SCReLU);
        assert_eq!(config.concat_size(), 512);
        assert_eq!(config.num_hidden_layers(), 3);
        assert_eq!(config.last_hidden_size(), 32);
    }

    #[test]
    fn static_config_keeps_borrowed_topology() {
        let config = NnueConfig::new_static(530, 256, &[64], Activation::CReLU);
        assert!(matches!(config.hidden_sizes, Cow::Borrowed(_)));
        assert_eq!(config.hidden_sizes.as_ref(), &[64]);
    }
}
