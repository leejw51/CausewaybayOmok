//! What the search needs from a network, and a name for the batch it returns.

use crate::netspec::NetSpec;

/// Softmaxed policy plus a value in [-1, 1] for a batch of positions.
pub struct Prediction {
    pub batch: usize,
    pub action_size: usize,
    /// `batch * action_size`, row-major.
    pub policy: Vec<f32>,
    /// `batch` values, from the side-to-move's point of view.
    pub value: Vec<f32>,
}

impl Prediction {
    pub fn policy_row(&self, i: usize) -> &[f32] {
        &self.policy[i * self.action_size..(i + 1) * self.action_size]
    }
}

/// A policy/value network the search can call.
///
/// `planes` is a contiguous NCHW f32 buffer of `batch` positions as produced by
/// [`crate::encode::encode_batch`].
pub trait Evaluator {
    fn spec(&self) -> NetSpec;
    fn describe(&self) -> String;
    fn predict(&mut self, planes: &[f32], batch: usize) -> Result<Prediction, String>;
}
