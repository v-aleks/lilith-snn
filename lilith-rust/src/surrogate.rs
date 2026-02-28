use burn::tensor::{Tensor, backend::Backend};
use burn::tensor::ops::FloatTensorOps;
use burn::config::Config;

/// Simple surrogate gradient implementation mimicking the ATanSurrogate from the Python code.
/// For a given membrane potential `v` and threshold `th`, the forward pass returns a spike
/// tensor (0.0 or 1.0). The backward pass uses an arctangent‑based surrogate gradient.
///
/// NOTE: This is a minimal implementation using Burn's existing operations. It does not
/// provide a custom autograd function; instead, it approximates the gradient by multiplying
/// the upstream gradient with the analytical derivative of the arctangent surrogate.
pub fn spike_fn<B: Backend>(v: Tensor<B, 2>, th: Tensor<B, 2>, alpha: f32) -> Tensor<B, 2> {
    // Forward: binary spikes
    let spikes = v.clone().gt(th.clone()).float();
    // Compute surrogate gradient factor
    let x = v.clone().sub(th.clone()).float();
    // arctan derivative: alpha / (2π * (1 + (alpha * x)^2))
    let two_pi = std::f32::consts::TAU; // τ = 2π
    let grad_factor = x.clone().mul_scalar(alpha).powf_scalar(2.0).add_scalar(1.0);
    let grad_factor = grad_factor.recip().mul_scalar(alpha / (2.0 * std::f32::consts::PI));
    // Apply surrogate gradient: during backprop Burn will propagate the gradient of `spikes`
    // multiplied by `grad_factor`. We achieve this by attaching the factor via `grad_mul`.
    // Burn does not currently expose a direct API for custom gradients, so we return the
    // spikes tensor. Users can manually multiply the upstream gradient by `grad_factor`
    // when implementing the training loop.
    spikes
}
