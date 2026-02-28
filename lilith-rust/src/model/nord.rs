use burn::tensor::{Tensor, backend::Backend};
use burn::nn::{Linear, LinearConfig};
use crate::config::NordConfig;
use crate::model::{encoder::TemporalSpikeEncoder, lif::AssociativeLIF, feedforward::NordBlock};

/// NordModel – top‑level SNN model (stub implementation).
///
/// This struct assembles the encoder, input LIF neuron, a stack of `NordBlock`s,
/// a read‑out LIF, EMA smoothing, a layer‑norm weight, and a language‑model head.
/// The forward method currently returns a zero‑filled tensor of the expected
/// shape; the detailed computation can be added later.
pub struct NordModel<B: Backend> {
    cfg: NordConfig,
    encoder: TemporalSpikeEncoder<B>,
    input_lif: AssociativeLIF<B>,
    blocks: Vec<NordBlock<B>>,
    readout_lif: AssociativeLIF<B>,
    // EMA smoothing factor (scalar).
    ema_alpha: Tensor<B, 1>,
    // LayerNorm weight (per‑channel scaling).
    layer_norm_weight: Tensor<B, 1>,
    // Language‑model head (linear projection to vocab size).
    head: Linear<B>,
}

impl<B: Backend> NordModel<B> {
    /// Create a new model from the configuration.
    pub fn new(cfg: NordConfig) -> Self {
        // Encoder.
        let encoder = TemporalSpikeEncoder::new(cfg.clone());
        // Input LIF (persistent disabled for stub).
        let input_lif = AssociativeLIF::new(cfg.d_model, cfg.clone(), false);
        // Stack of NordBlocks.
        let mut blocks = Vec::new();
        for _ in 0..cfg.n_layers {
            blocks.push(NordBlock::new(cfg.clone()));
        }
        // Read‑out LIF.
        let readout_lif = AssociativeLIF::new(cfg.d_model, cfg.clone(), false);
        // EMA smoothing scalar (initialized to 0.9 as in Python config).
        let ema_alpha = Tensor::full([1], cfg.tau_mem);
        // LayerNorm weight (ones).
        let layer_norm_weight = Tensor::full([cfg.d_model], 1.0);
        // Language‑model head linear layer.
        let head_cfg = LinearConfig::new(cfg.d_model, cfg.vocab_size).with_bias(false);
        let head = Linear::new(head_cfg);

        Self {
            cfg,
            encoder,
            input_lif,
            blocks,
            readout_lif,
            ema_alpha,
            layer_norm_weight,
            head,
        }
    }

    /// Forward pass (stub).
    ///
    /// * `token_ids` – shape `[B, S]`.
    /// Returns a tensor of shape `[T+T_slow, B, S, vocab_size]` (currently zeros).
    pub fn forward(&mut self, token_ids: Tensor<B, 2>) -> Tensor<B, 4> {
        // Encode tokens to currents.
        let encoded = self.encoder.forward(token_ids);
        // Input LIF processing.
        let _ = self.input_lif.forward(encoded.clone(), 1.0);
        // Pass through each block.
        let mut x = encoded;
        for block in &self.blocks {
            x = block.forward(x);
        }
        // Read‑out LIF.
        let _ = self.readout_lif.forward(x.clone(), 1.0);
        // EMA smoothing (stub – no effect).
        let _ = self.ema_alpha.clone();
        // LayerNorm scaling (stub – no effect).
        let _ = self.layer_norm_weight.clone();
        // Project to vocab logits.
        let logits = self.head.forward(x);
        // Return zeros of expected shape (placeholder).
        Tensor::zeros(logits.shape())
    }
}
