//! Backend-independent description of the policy/value network.
//!
//! The parameter names are the PyTorch `state_dict` keys used by the trainer,
//! so a checkpoint written on any of its backends loads here unchanged.
//! Mirrors `omok/netspec.py`.

use crate::encode::NUM_PLANES;
use crate::npz::Weights;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NetSpec {
    pub board_size: usize,
    pub in_planes: usize,
    pub channels: usize,
    pub blocks: usize,
    pub policy_channels: usize,
    pub value_channels: usize,
    pub value_hidden: usize,
}

impl Default for NetSpec {
    fn default() -> Self {
        NetSpec {
            board_size: 15,
            in_planes: NUM_PLANES,
            channels: 96,
            blocks: 6,
            policy_channels: 4,
            value_channels: 2,
            value_hidden: 128,
        }
    }
}

impl NetSpec {
    pub fn action_size(&self) -> usize {
        self.board_size * self.board_size
    }

    /// Read a spec out of a checkpoint's sidecar JSON (`{"spec": {...}}`) or
    /// out of a bare spec object; missing fields keep their defaults.
    pub fn from_json(value: &serde_json::Value) -> Self {
        let spec = value.get("spec").unwrap_or(value);
        let mut out = NetSpec::default();
        let mut read = |key: &str, into: &mut usize| {
            if let Some(v) = spec.get(key).and_then(|v| v.as_u64()) {
                *into = v as usize;
            }
        };
        read("board_size", &mut out.board_size);
        read("in_planes", &mut out.in_planes);
        read("channels", &mut out.channels);
        read("blocks", &mut out.blocks);
        read("policy_channels", &mut out.policy_channels);
        read("value_channels", &mut out.value_channels);
        read("value_hidden", &mut out.value_hidden);
        out
    }

    /// Recover the spec from the weight shapes alone, so a `.npz` with no
    /// sidecar JSON still loads.
    pub fn from_weights(weights: &Weights) -> Result<Self, String> {
        let stem = weights
            .get("stem.conv.weight")
            .ok_or("weights have no stem.conv.weight; is this an Omok checkpoint?")?;
        if stem.shape.len() != 4 {
            return Err(format!("stem.conv.weight has shape {:?}, expected 4 axes", stem.shape));
        }
        let channels = stem.shape[0];
        let in_planes = stem.shape[1];
        let blocks = (0..).take_while(|i| weights.contains_key(&format!("blocks.{i}.conv1.weight"))).count();

        let policy_conv = weights.get("policy.conv.weight").ok_or("weights have no policy head")?;
        let value_conv = weights.get("value.conv.weight").ok_or("weights have no value head")?;
        let policy_channels = policy_conv.shape[0];
        let value_channels = value_conv.shape[0];

        let policy_fc = weights.get("policy.fc.weight").ok_or("weights have no policy.fc.weight")?;
        let action_size = policy_fc.shape[0];
        let board_size = (action_size as f64).sqrt().round() as usize;
        if board_size * board_size != action_size {
            return Err(format!("policy head has {action_size} outputs, which is not a square board"));
        }
        let value_hidden = weights
            .get("value.fc1.weight")
            .ok_or("weights have no value.fc1.weight")?
            .shape[0];

        Ok(NetSpec {
            board_size,
            in_planes,
            channels,
            blocks,
            policy_channels,
            value_channels,
            value_hidden,
        })
    }

    pub fn parameter_count(&self, weights: &Weights) -> usize {
        let _ = self;
        weights.values().map(|t| t.len()).sum()
    }

    pub fn describe(&self) -> String {
        format!(
            "{}x{} board, {} blocks x {} channels",
            self.board_size, self.board_size, self.blocks, self.channels
        )
    }
}
