use burn::module::Module;
use burn::tensor::backend::Backend;
use crate::config::NordConfig;

/// Count the total number of trainable parameters in a model.
/// Returns the sum of `numel()` for each parameter tensor.
pub fn count_parameters<B: Backend, M: Module<B>>(model: &M) -> usize {
    // Burn's `state()` returns a hashmap of named tensors.
    model.state().values().map(|t| t.numel()).sum()
}

/// Estimate VRAM usage (in megabytes) for a given configuration.
/// This mirrors the heuristic used in the Python implementation.
pub fn estimate_vram<B: Backend>(cfg: &NordConfig) -> f32 {
    // Parameter memory: assume 4 bytes per float32 parameter.
    let param_bytes = (
        cfg.vocab_size * cfg.d_model + // embedding matrix
        cfg.n_layers * (
            // Linear layers in each block (Q,K,V,O) and feed‑forward
            4 * cfg.d_model * cfg.d_model + // Q,K,V,O projections
            2 * cfg.d_model * cfg.d_ff       // up/down FF
        )
    ) as f32 * 4.0;
    // Activation memory: rough estimate based on sequence length and model size.
    let activation_bytes = (cfg.max_seq_len * cfg.d_model * cfg.n_layers) as f32 * 4.0;
    (param_bytes + activation_bytes) / (1024.0 * 1024.0)
}
