use serde::{Deserialize, Serialize};

/// Configuration for the Nord SNN model, mirroring the Python `NordConfig` dataclass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NordConfig {
    // Tokenizer
    pub tokenizer_id: String,
    // Dimensions
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    // Multi‑Scale Temporal
    pub T: usize,
    pub T_slow: usize,
    pub persistent_mem: bool,
    // LIF neuron dynamics
    pub tau_mem: f32,
    pub tau_syn: f32,
    pub v_threshold: f32,
    pub v_reset: f32,
    pub refractory_t: usize,
    pub threshold_lr: f32,
    // Adaptive cascade
    pub n_clusters: usize,
    pub cascade_radius: usize,
    pub cascade_gain: f32,
    // Reward‑Modulated STDP
    pub stdp_a_plus: f32,
    pub stdp_a_minus: f32,
    pub stdp_tau_plus: f32,
    pub stdp_tau_minus: f32,
    pub stdp_w_max: f32,
    pub stdp_w_min: f32,
    pub stdp_reward_scale: f32,
    // Sparse resonance
    pub resonance_top_k: usize,
    // LeakyClamp
    pub clamp_floor: f32,
    // Surrogate gradient
    pub surrogate_alpha: f32,
    // Training hyper‑parameters
    pub batch_size: usize,
    pub grad_accum: usize,
    pub lr: f32,
    pub min_lr: f32,
    pub weight_decay: f32,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub save_every: usize,
    pub log_every: usize,
    pub max_grad_norm: f32,
    // Hardware
    pub dtype: String,
    pub device: String,
}

impl Default for NordConfig {
    fn default() -> Self {
        Self {
            tokenizer_id: "meta-llama/Llama-3.2-1B".to_string(),
            vocab_size: 128_256,
            d_model: 512,
            n_heads: 8,
            n_layers: 6,
            d_ff: 1024,
            max_seq_len: 1024,
            T: 8,
            T_slow: 2,
            persistent_mem: true,
            tau_mem: 0.9,
            tau_syn: 0.50,
            v_threshold: 0.25,
            v_reset: -0.1,
            refractory_t: 2,
            threshold_lr: 0.01,
            n_clusters: 64,
            cascade_radius: 3,
            cascade_gain: 0.8,
            stdp_a_plus: 0.005,
            stdp_a_minus: 0.005,
            stdp_tau_plus: 20.0,
            stdp_tau_minus: 20.0,
            stdp_w_max: 1.0,
            stdp_w_min: -0.3,
            stdp_reward_scale: 1.0,
            resonance_top_k: 64,
            clamp_floor: -0.1,
            surrogate_alpha: 4.0,
            batch_size: 4,
            grad_accum: 8,
            lr: 5e-4,
            min_lr: 1e-5,
            weight_decay: 0.01,
            warmup_steps: 500,
            max_steps: 100_000,
            save_every: 1000,
            log_every: 10,
            max_grad_norm: 1.0,
            dtype: "float16".to_string(),
            device: "cuda".to_string(),
        }
    }
}
