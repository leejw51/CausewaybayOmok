//! The Omok policy/value network, running on Apple MLX.
//!
//! The architecture is the one in `omok/backends/mlx_backend.py`: a 3x3 stem,
//! a stack of residual blocks, then a policy head and a value head.  Nothing
//! here trains -- weights come from the Python trainer's `.npz` checkpoints,
//! and everything layout- or batch-norm-shaped is folded in `weights.rs` at
//! load time, so a forward pass is convolutions, matmuls and two activations.

pub mod weights;

use std::path::{Path, PathBuf};

use mlx_rs::error::Result as MlxResult;
use mlx_rs::nn::relu;
use mlx_rs::ops::{conv2d, softmax_axes, tanh};
use mlx_rs::Array;

use omok_core::net::{Evaluator, Prediction};
use omok_core::netspec::NetSpec;
use omok_core::npz::load_npz;

use weights::{conv_bn, linear, linear_from_feature_map, Conv, Linear};

fn to_array(data: &[f32], shape: &[i32]) -> Array {
    Array::from_slice(data, shape)
}

struct GpuConv {
    weight: Array,
    bias: Array,
    padding: i32,
}

impl GpuConv {
    fn new(conv: Conv) -> Self {
        let channels = conv.shape[0];
        GpuConv {
            weight: to_array(&conv.weight, &conv.shape),
            // [1, 1, 1, C] broadcasts across an NHWC feature map.
            bias: to_array(&conv.bias, &[1, 1, 1, channels]),
            padding: conv.padding,
        }
    }

    fn forward(&self, x: &Array) -> MlxResult<Array> {
        let y = conv2d(x, &self.weight, (1, 1), (self.padding, self.padding), (1, 1), 1)?;
        y.add(&self.bias)
    }
}

struct GpuLinear {
    weight: Array,
    bias: Array,
}

impl GpuLinear {
    fn new(l: Linear) -> Self {
        let out = l.shape[1];
        GpuLinear {
            weight: to_array(&l.weight, &l.shape),
            bias: to_array(&l.bias, &[1, out]),
        }
    }

    fn forward(&self, x: &Array) -> MlxResult<Array> {
        x.matmul(&self.weight)?.add(&self.bias)
    }
}

struct ResBlock {
    conv1: GpuConv,
    conv2: GpuConv,
}

impl ResBlock {
    fn forward(&self, x: &Array) -> MlxResult<Array> {
        let y = relu(&self.conv1.forward(x)?)?;
        let y = self.conv2.forward(&y)?;
        relu(&x.add(&y)?)
    }
}

/// A loaded network, ready to answer batches of positions.
pub struct MlxNet {
    spec: NetSpec,
    source: PathBuf,
    parameters: usize,
    stem: GpuConv,
    blocks: Vec<ResBlock>,
    policy_conv: GpuConv,
    policy_fc: GpuLinear,
    value_conv: GpuConv,
    value_fc1: GpuLinear,
    value_fc2: GpuLinear,
}

impl MlxNet {
    /// Load a `.npz` checkpoint written by the Python trainer.  A sidecar JSON
    /// (`best.json` next to `best.npz`) is used for the architecture when it is
    /// there; otherwise the shapes of the weights say what the network is.
    pub fn load(path: &Path) -> Result<Self, String> {
        let w = load_npz(path)?;

        let sidecar = path.with_extension("json");
        let spec = match std::fs::read_to_string(&sidecar)
            .ok()
            .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        {
            Some(meta) => NetSpec::from_json(&meta),
            None => NetSpec::from_weights(&w)?,
        };
        // Trust the weights over the sidecar: a stale JSON should not produce a
        // silently wrong network.
        let derived = NetSpec::from_weights(&w)?;
        if derived != spec {
            return Err(format!(
                "{} describes a {} network but its weights are a {} one",
                sidecar.display(),
                spec.describe(),
                derived.describe()
            ));
        }

        let n = spec.board_size;
        let mut blocks = Vec::with_capacity(spec.blocks);
        for i in 0..spec.blocks {
            blocks.push(ResBlock {
                conv1: GpuConv::new(conv_bn(
                    &w,
                    &format!("blocks.{i}.conv1.weight"),
                    &format!("blocks.{i}.bn1"),
                    1,
                )?),
                conv2: GpuConv::new(conv_bn(
                    &w,
                    &format!("blocks.{i}.conv2.weight"),
                    &format!("blocks.{i}.bn2"),
                    1,
                )?),
            });
        }

        Ok(MlxNet {
            parameters: w.values().map(|t| t.len()).sum(),
            spec,
            source: path.to_path_buf(),
            stem: GpuConv::new(conv_bn(&w, "stem.conv.weight", "stem.bn", 1)?),
            blocks,
            policy_conv: GpuConv::new(conv_bn(&w, "policy.conv.weight", "policy.bn", 0)?),
            policy_fc: GpuLinear::new(linear_from_feature_map(
                &w,
                "policy.fc",
                spec.policy_channels,
                n,
            )?),
            value_conv: GpuConv::new(conv_bn(&w, "value.conv.weight", "value.bn", 0)?),
            value_fc1: GpuLinear::new(linear_from_feature_map(
                &w,
                "value.fc1",
                spec.value_channels,
                n,
            )?),
            value_fc2: GpuLinear::new(linear(&w, "value.fc2")?),
        })
    }

    /// `planes` is NCHW as the encoder produces it; MLX convolves NHWC, so the
    /// one transpose the pipeline needs happens here, on the smallest tensor
    /// in the graph (5 planes in, 64+ channels everywhere after).
    fn to_nhwc(&self, planes: &[f32], batch: usize) -> Array {
        let n = self.spec.board_size;
        let c = self.spec.in_planes;
        let area = n * n;
        let mut nhwc = vec![0.0f32; batch * area * c];
        for b in 0..batch {
            let src = &planes[b * c * area..(b + 1) * c * area];
            let dst = &mut nhwc[b * area * c..(b + 1) * area * c];
            for ch in 0..c {
                let plane = &src[ch * area..(ch + 1) * area];
                for i in 0..area {
                    dst[i * c + ch] = plane[i];
                }
            }
        }
        to_array(&nhwc, &[batch as i32, n as i32, n as i32, c as i32])
    }

    fn forward(&self, x: &Array, batch: i32) -> MlxResult<(Array, Array)> {
        let mut y = relu(&self.stem.forward(x)?)?;
        for block in &self.blocks {
            y = block.forward(&y)?;
        }

        let n = self.spec.board_size as i32;
        let policy = relu(&self.policy_conv.forward(&y)?)?;
        let policy = policy.reshape(&[batch, n * n * self.spec.policy_channels as i32])?;
        let logits = self.policy_fc.forward(&policy)?;
        let policy = softmax_axes(&logits, &[-1], None)?;

        let value = relu(&self.value_conv.forward(&y)?)?;
        let value = value.reshape(&[batch, n * n * self.spec.value_channels as i32])?;
        let value = relu(&self.value_fc1.forward(&value)?)?;
        let value = tanh(&self.value_fc2.forward(&value)?)?.reshape(&[batch])?;

        Ok((policy, value))
    }

    pub fn source(&self) -> &Path {
        &self.source
    }
}

impl Evaluator for MlxNet {
    fn spec(&self) -> NetSpec {
        self.spec
    }

    fn describe(&self) -> String {
        format!(
            "mlx | {} | {} parameters | {}",
            self.spec.describe(),
            self.parameters,
            self.source.display()
        )
    }

    fn predict(&mut self, planes: &[f32], batch: usize) -> Result<Prediction, String> {
        let action_size = self.spec.action_size();
        if batch == 0 {
            return Ok(Prediction { batch, action_size, policy: Vec::new(), value: Vec::new() });
        }
        let expected = batch * self.spec.in_planes * action_size;
        if planes.len() != expected {
            return Err(format!(
                "expected {expected} floats for a batch of {batch}, got {}",
                planes.len()
            ));
        }
        let x = self.to_nhwc(planes, batch);
        let (policy, value) = self
            .forward(&x, batch as i32)
            .map_err(|e| format!("MLX forward pass failed: {e}"))?;
        mlx_rs::transforms::eval([&policy, &value])
            .map_err(|e| format!("MLX evaluation failed: {e}"))?;
        Ok(Prediction {
            batch,
            action_size,
            policy: policy.as_slice::<f32>().to_vec(),
            value: value.as_slice::<f32>().to_vec(),
        })
    }
}
