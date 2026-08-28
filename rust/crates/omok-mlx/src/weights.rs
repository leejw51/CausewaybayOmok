//! Turn a trainer checkpoint into the tensors MLX actually wants.
//!
//! Two layout differences and one algebraic simplification are handled here,
//! once at load time, so the forward pass is nothing but convolutions and
//! matmuls:
//!
//! * PyTorch convolution weights are `OIHW`; MLX wants `OHWI`.
//! * The heads flatten their feature map in NCHW order before the fully
//!   connected layer, but MLX hands us NHWC.  Permuting the *columns* of the
//!   weight matrix gives the same result from a straight NHWC flatten, so no
//!   transpose is needed at run time.
//! * Batch-norm is fixed at inference, so each `conv -> bn` pair folds into a
//!   single convolution with a bias.

use omok_core::npz::{Tensor, Weights};

/// PyTorch's `BatchNorm2d` default, matched by MLX's `nn.BatchNorm`.
const BN_EPS: f32 = 1e-5;

pub struct Conv {
    /// `[out, kh, kw, in]`, the layout `mlx_rs::ops::conv2d` expects.
    pub weight: Vec<f32>,
    pub shape: [i32; 4],
    /// Folded batch-norm bias, one per output channel.
    pub bias: Vec<f32>,
    pub padding: i32,
}

pub struct Linear {
    /// `[in, out]`, already column-permuted for an NHWC flatten where relevant.
    pub weight: Vec<f32>,
    pub shape: [i32; 2],
    pub bias: Vec<f32>,
}

fn get<'a>(w: &'a Weights, key: &str) -> Result<&'a Tensor, String> {
    w.get(key).ok_or_else(|| format!("checkpoint is missing {key}"))
}

/// `scale`/`shift` such that `bn(x) == x * scale + shift` at inference.
fn bn_affine(w: &Weights, prefix: &str, channels: usize) -> Result<(Vec<f32>, Vec<f32>), String> {
    let gamma = get(w, &format!("{prefix}.weight"))?;
    let beta = get(w, &format!("{prefix}.bias"))?;
    let mean = get(w, &format!("{prefix}.running_mean"))?;
    let var = get(w, &format!("{prefix}.running_var"))?;
    for (name, t) in [("weight", gamma), ("bias", beta), ("running_mean", mean), ("running_var", var)] {
        if t.len() != channels {
            return Err(format!(
                "{prefix}.{name} has {} entries, expected {channels}", t.len()
            ));
        }
    }
    let mut scale = Vec::with_capacity(channels);
    let mut shift = Vec::with_capacity(channels);
    for c in 0..channels {
        let s = gamma.data[c] / (var.data[c] + BN_EPS).sqrt();
        scale.push(s);
        shift.push(beta.data[c] - mean.data[c] * s);
    }
    Ok((scale, shift))
}

/// Load `conv_key` (OIHW, no bias) folded with the batch-norm at `bn_prefix`.
pub fn conv_bn(w: &Weights, conv_key: &str, bn_prefix: &str, padding: i32) -> Result<Conv, String> {
    let conv = get(w, conv_key)?;
    if conv.shape.len() != 4 {
        return Err(format!("{conv_key} has shape {:?}, expected 4 axes", conv.shape));
    }
    let (o, i, kh, kw) = (conv.shape[0], conv.shape[1], conv.shape[2], conv.shape[3]);
    let (scale, bias) = bn_affine(w, bn_prefix, o)?;

    // OIHW -> OHWI, scaling each output channel by the batch-norm factor.
    let mut weight = vec![0.0f32; o * kh * kw * i];
    for oc in 0..o {
        let s = scale[oc];
        for ic in 0..i {
            for y in 0..kh {
                for x in 0..kw {
                    let from = ((oc * i + ic) * kh + y) * kw + x;
                    let to = ((oc * kh + y) * kw + x) * i + ic;
                    weight[to] = conv.data[from] * s;
                }
            }
        }
    }
    Ok(Conv {
        weight,
        shape: [o as i32, kh as i32, kw as i32, i as i32],
        bias,
        padding,
    })
}

/// Load a fully connected layer whose input is a feature map flattened in
/// NCHW order, and re-index it so an NHWC flatten of `channels x n x n` works.
pub fn linear_from_feature_map(
    w: &Weights,
    prefix: &str,
    channels: usize,
    n: usize,
) -> Result<Linear, String> {
    let weight = get(w, &format!("{prefix}.weight"))?;
    let bias = get(w, &format!("{prefix}.bias"))?;
    if weight.shape.len() != 2 {
        return Err(format!("{prefix}.weight has shape {:?}, expected 2 axes", weight.shape));
    }
    let (out, inputs) = (weight.shape[0], weight.shape[1]);
    if inputs != channels * n * n {
        return Err(format!(
            "{prefix}.weight expects {inputs} inputs but the feature map is {channels}x{n}x{n}"
        ));
    }
    let mut permuted = vec![0.0f32; inputs * out];
    for o in 0..out {
        for c in 0..channels {
            for y in 0..n {
                for x in 0..n {
                    let from = o * inputs + (c * n + y) * n + x;
                    let to = ((y * n + x) * channels + c) * out + o;
                    permuted[to] = weight.data[from];
                }
            }
        }
    }
    Ok(Linear {
        weight: permuted,
        shape: [inputs as i32, out as i32],
        bias: bias.data.clone(),
    })
}

/// A plain fully connected layer, transposed from `[out, in]` to `[in, out]`.
pub fn linear(w: &Weights, prefix: &str) -> Result<Linear, String> {
    let weight = get(w, &format!("{prefix}.weight"))?;
    let bias = get(w, &format!("{prefix}.bias"))?;
    if weight.shape.len() != 2 {
        return Err(format!("{prefix}.weight has shape {:?}, expected 2 axes", weight.shape));
    }
    let (out, inputs) = (weight.shape[0], weight.shape[1]);
    let mut transposed = vec![0.0f32; inputs * out];
    for o in 0..out {
        for i in 0..inputs {
            transposed[i * out + o] = weight.data[o * inputs + i];
        }
    }
    Ok(Linear {
        weight: transposed,
        shape: [inputs as i32, out as i32],
        bias: bias.data.clone(),
    })
}
