use burn::tensor::{Tensor, backend::Backend};
use crate::config::NordConfig;

/// STDPEngine – reward‑modulated spike‑timing‑dependent plasticity (stub).
///
/// The full algorithm updates synaptic weights based on pre‑ and post‑spike
/// traces, modulated by a reward signal derived from the training loss.
/// This stub captures the configuration and provides the public API; the actual
/// weight‑update logic can be filled in later.
pub struct STDPEngine<B: Backend> {
    cfg: NordConfig,
    // STDP hyper‑parameters.
    a_plus: f32,
    a_minus: f32,
    tau_plus: f32,
    tau_minus: f32,
    w_max: f32,
    w_min: f32,
    reward_scale: f32,
    // Running EMA of the loss (baseline).
    loss_ema: f32,
    ema_decay: f32,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> STDPEngine<B> {
    /// Create a new engine from a configuration.
    pub fn new(cfg: NordConfig) -> Self {
        Self {
            cfg,
            a_plus: cfg.stdp_a_plus,
            a_minus: cfg.stdp_a_minus,
            tau_plus: cfg.stdp_tau_plus,
            tau_minus: cfg.stdp_tau_minus,
            w_max: cfg.stdp_w_max,
            w_min: cfg.stdp_w_min,
            reward_scale: cfg.stdp_reward_scale,
            loss_ema: 10.0, // initialise high, matches Python default
            ema_decay: 0.99,
            _marker: std::marker::PhantomData,
        }
    }

    /// Update the running loss EMA after a training step.
    pub fn update_reward(&mut self, current_loss: f32) {
        self.loss_ema = self.ema_decay * self.loss_ema + (1.0 - self.ema_decay) * current_loss;
    }

    /// Compute the reward signal from the current loss.
    ///
    /// reward = sigmoid(baseline - current) * reward_scale
    fn compute_reward(&self, current_loss: f32) -> f32 {
        let diff = self.loss_ema - current_loss;
        let sigmoid = 1.0 / (1.0 + (-diff).exp());
        sigmoid * self.reward_scale
    }

    /// Apply a (stub) STDP update to a weight tensor.
    ///
    /// * `weights` – synaptic weight matrix.
    /// * `pre_spikes` – pre‑synaptic spike tensor.
    /// * `post_spikes` – post‑synaptic spike tensor.
    /// * `current_loss` – loss for the current batch (used to compute reward).
    ///
    /// The real implementation would compute eligibility traces and adjust the
    /// weights based on `a_plus`, `a_minus`, `tau_plus`, `tau_minus`, and the reward.
    /// This stub simply returns the original weights unchanged.
    pub fn apply(
        &mut self,
        weights: Tensor<B, 2>,
        _pre_spikes: Tensor<B, 2>,
        _post_spikes: Tensor<B, 2>,
        current_loss: f32,
    ) -> Tensor<B, 2> {
        // Update EMA before computing reward.
        self.update_reward(current_loss);
        let _reward = self.compute_reward(current_loss);
        // TODO: implement actual STDP weight update.
        weights
    }
}
