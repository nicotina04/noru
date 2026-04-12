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

/// Owned, runtime-constructible variant of [`NnueConfig`].
///
/// [`NnueConfig`] stores `hidden_sizes` as a `&'static [usize]` for zero-cost
/// copying, which is ideal when the topology is known at compile time. When
/// the topology must be built at runtime (FFI bindings, config files, GUI
/// editors), use `OwnedNnueConfig` and convert via [`OwnedNnueConfig::leak`].
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
        Self { feature_size, accumulator_size, hidden_sizes, activation }
    }

    /// Consume `self` and produce a [`NnueConfig`] by leaking `hidden_sizes`
    /// into a `&'static [usize]`.
    ///
    /// The leaked memory can be reclaimed later by passing the resulting
    /// `NnueConfig.hidden_sizes` to [`reclaim_leaked_hidden_sizes`] — pair
    /// every `leak` with exactly one `reclaim` to avoid an actual leak.
    pub fn leak(self) -> NnueConfig {
        let boxed: Box<[usize]> = self.hidden_sizes.into_boxed_slice();
        let static_ref: &'static [usize] = Box::leak(boxed);
        NnueConfig {
            feature_size: self.feature_size,
            accumulator_size: self.accumulator_size,
            hidden_sizes: static_ref,
            activation: self.activation,
        }
    }
}

/// Reclaim a `hidden_sizes` slice previously produced by
/// [`OwnedNnueConfig::leak`].
///
/// # Safety
///
/// `hidden_sizes` must be a slice that was produced by `OwnedNnueConfig::leak`
/// and has not already been reclaimed. Passing any other slice — including a
/// genuinely `'static` literal like `&[64]` — is undefined behavior.
pub unsafe fn reclaim_leaked_hidden_sizes(hidden_sizes: &'static [usize]) {
    let ptr = hidden_sizes as *const [usize] as *mut [usize];
    drop(Box::from_raw(ptr));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_config_leak_preserves_fields() {
        let owned = OwnedNnueConfig::new(768, 256, vec![256, 32, 32], Activation::SCReLU);
        let config = owned.leak();
        assert_eq!(config.feature_size, 768);
        assert_eq!(config.accumulator_size, 256);
        assert_eq!(config.hidden_sizes, &[256, 32, 32]);
        assert_eq!(config.activation, Activation::SCReLU);
        assert_eq!(config.concat_size(), 512);
        assert_eq!(config.num_hidden_layers(), 3);
        assert_eq!(config.last_hidden_size(), 32);
        unsafe { reclaim_leaked_hidden_sizes(config.hidden_sizes) };
    }
}
