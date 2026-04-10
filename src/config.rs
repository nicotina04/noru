/// NNUE network runtime configuration.
///
/// Configure feature size, accumulator size, and hidden layer layout per game.

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NnueConfig {
    /// Total input feature size
    pub feature_size: usize,
    /// Accumulator neurons per perspective
    pub accumulator_size: usize,
    /// Hidden layer sizes (e.g. `&[64]` or `&[256, 32, 32]`)
    pub hidden_sizes: &'static [usize],
    /// Activation function (applied to accumulator output only)
    pub activation: Activation,
}

impl NnueConfig {
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
        *self.hidden_sizes.last().expect("hidden_sizes must not be empty")
    }
}
