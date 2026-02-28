use burn::tensor::{Tensor, backend::Backend};
use burn::tensor::ops::{FloatTensorOps, BoolTensorOps, TensorOps};
use burn::config::Config;
use crate::config::NordConfig;
use crate::surrogate::spike_fn;

/// Associative LIF neuron implementation (v3) – mirrors the Python `AssociativeLIF` class.
///
/// This implementation focuses on the core forward logic required for the SNN model.
/// It does **not** implement a custom autograd function; instead it relies on the
/// surrogate gradient defined in `spike_fn`. The gradient of the spike tensor is
/// approximated by the arctangent surrogate when the training loop multiplies the
/// upstream gradient by the factor returned from `spike_fn`.
///
/// The neuron maintains learnable parameters:
/// * `threshold` – per‑channel spike threshold.
/// * `beta_mem_raw` / `beta_syn_raw` – raw membrane and synaptic decay parameters
///   (sigmoid‑ed to obtain `beta_mem` and `beta_syn`).
/// * `neighbor_weights` – soft, learnable adjacency matrix for the adaptive cascade.
/// * `cluster_gain` – per‑cluster gain.
/// * `cluster_ids` – buffer mapping each neuron to a cluster (static).
///
/// Persistent membrane and synaptic states are optional and stored as buffers when
/// `persistent` is true.
pub struct AssociativeLIF<B: Backend> {
    /// Configuration reference (used for dimensions and hyper‑parameters).
    cfg: NordConfig,
    /// Number of neurons in this layer.
    d: usize,
    /// Whether to keep membrane/synapse state between forward passes.
    persistent: bool,
    /// Learnable spike threshold (shape: [d]).
    pub threshold: Tensor<B, 1>,
    /// Raw membrane decay parameter (scalar).
    pub beta_mem_raw: Tensor<B, 1>,
    /// Raw synaptic decay parameter (scalar).
    pub beta_syn_raw: Tensor<B, 1>,
    /// Soft neighbor weight matrix (shape: [n_clusters, n_clusters]).
    pub neighbor_weights: Tensor<B, 2>,
    /// Per‑cluster gain (shape: [n_clusters]).
    pub cluster_gain: Tensor<B, 1>,
    /// Mapping from neuron index to cluster id (shape: [d]).
    pub cluster_ids: Tensor<B, 1>,
    /// Optional persistent membrane state (shape: [1, d]).
    pub v_mem_state: Option<Tensor<B, 2>>,
    /// Optional persistent synaptic current state (shape: [1, d]).
    pub i_syn_state: Option<Tensor<B, 2>>,
}

impl<B: Backend> AssociativeLIF<B> {
    /// Create a new AssociativeLIF layer.
    pub fn new(d: usize, cfg: NordConfig, persistent: bool) -> Self {
        // Threshold parameter initialized to cfg.v_threshold.
        let threshold = Tensor::full([d], cfg.v_threshold as f32);
        // Raw decay parameters – use the same transformation as the Python code.
        let beta_mem_raw = Tensor::full([1], (cfg.tau_mem / (1.0 - cfg.tau_mem + 1e-6_f32)).ln());
        let beta_syn_raw = Tensor::full([1], (cfg.tau_syn / (1.0 - cfg.tau_syn + 1e-6_f32)).ln());

        // Cluster topology.
        let nc = cfg.n_clusters;
        let mut cluster_ids = Vec::with_capacity(d);
        for i in 0..d {
            cluster_ids.push((i % nc) as i64);
        }
        let cluster_ids = Tensor::from_data(cluster_ids.into(), &B::device());

        // Initialize neighbor_weights according to the radius.
        let r = cfg.cascade_radius as i32;
        let mut init = vec![vec![0.0_f32; nc]; nc];
        for offset in -r..=r {
            if offset != 0 {
                let weight = 1.0 - (offset.abs() as f32) / ((r + 1) as f32);
                for i in 0..nc {
                    let j = ((i as i32 + offset).rem_euclid(nc as i32)) as usize;
                    init[i][j] = weight;
                }
            }
        }
        let neighbor_weights = Tensor::from_data(init.concat().into(), &B::device()).reshape([nc, nc]);

        // Per‑cluster gain.
        let cluster_gain = Tensor::full([nc], cfg.cascade_gain as f32);

        // Persistent buffers.
        let v_mem_state = if persistent {
            Some(Tensor::zeros([1, d]))
        } else {
            None
        };
        let i_syn_state = if persistent {
            Some(Tensor::zeros([1, d]))
        } else {
            None
        };

        Self {
            cfg,
            d,
            persistent,
            threshold,
            beta_mem_raw,
            beta_syn_raw,
            neighbor_weights,
            cluster_gain,
            cluster_ids,
            v_mem_state,
            i_syn_state,
        }
    }

    /// Sigmoid‑ed membrane decay.
    fn beta_mem(&self) -> Tensor<B, 1> {
        self.beta_mem_raw.clone().sigmoid()
    }

    /// Sigmoid‑ed synaptic decay.
    fn beta_syn(&self) -> Tensor<B, 1> {
        self.beta_syn_raw.clone().sigmoid()
    }

    /// Simple forward pass without cascade amplification.
    /// Computes membrane and synaptic dynamics and returns spike tensor.
    pub fn forward(&mut self, x: Tensor<B, 2>, alpha: f32) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // Retrieve or initialise states.
        let (v_mem, i_syn) = if self.persistent {
            (
                self.v_mem_state.take().unwrap_or_else(|| Tensor::zeros([1, self.d])),
                self.i_syn_state.take().unwrap_or_else(|| Tensor::zeros([1, self.d])),
            )
        } else {
            (Tensor::zeros([1, self.d]), Tensor::zeros([1, self.d]))
        };

        // Expand states to match batch size.
        let batch = x.shape()[0];
        let v_mem = v_mem.expand([batch, self.d]);
        let i_syn = i_syn.expand([batch, self.d]);

        // Decay dynamics.
        let beta_mem = self.beta_mem(); // scalar tensor
        let beta_syn = self.beta_syn(); // scalar tensor
        let v_mem = v_mem * (Tensor::full([1], 1.0) - beta_mem.clone()) + i_syn.clone() * beta_mem.clone();
        let i_syn = i_syn * (Tensor::full([1], 1.0) - beta_syn.clone()) + x;

        // Compute spikes using surrogate gradient.
        let spikes = spike_fn(v_mem.clone(), self.threshold.clone().unsqueeze_dim(0).expand([batch, self.d]), alpha);

        // Update persistent buffers if needed (store last time step).
        if self.persistent {
            self.v_mem_state = Some(v_mem.select(0, batch - 1).unsqueeze_dim(0));
            self.i_syn_state = Some(i_syn.select(0, batch - 1).unsqueeze_dim(0));
        }

        spikes
    }
}

