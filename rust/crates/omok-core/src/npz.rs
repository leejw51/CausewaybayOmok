//! Reader for the `.npz` checkpoints written by `omok/checkpoint.py`.
//!
//! `np.savez` writes a ZIP of `.npy` members with no compression, so the whole
//! reader is a central-directory walk plus a `.npy` header parse -- no zip or
//! ndarray dependency, and nothing to keep in sync with the trainer beyond the
//! two formats themselves, which are frozen.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

pub type Weights = HashMap<String, Tensor>;

fn u16le(b: &[u8], at: usize) -> usize {
    u16::from_le_bytes([b[at], b[at + 1]]) as usize
}

fn u32le(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize
}

/// Parse one `.npy` member: little-endian float32, C order.
fn parse_npy(name: &str, buf: &[u8]) -> Result<Tensor, String> {
    if buf.len() < 12 || &buf[..6] != b"\x93NUMPY" {
        return Err(format!("{name}: not a .npy member"));
    }
    let (major, minor) = (buf[6], buf[7]);
    // v1 has a 2-byte header length, v2/v3 have 4.
    let (header_len, body_at) = match major {
        1 => (u16le(buf, 8), 10),
        2 | 3 => (u32le(buf, 8), 12),
        _ => return Err(format!("{name}: unsupported .npy version {major}.{minor}")),
    };
    let header = std::str::from_utf8(&buf[body_at..body_at + header_len])
        .map_err(|_| format!("{name}: non-utf8 .npy header"))?;

    if !(header.contains("'<f4'") || header.contains("\"<f4\"")) {
        return Err(format!(
            "{name}: expected float32 little-endian ('<f4') weights, header was {header}"
        ));
    }
    if header.contains("'fortran_order': True") {
        return Err(format!("{name}: Fortran-order arrays are not supported"));
    }

    let shape_at = header
        .find("'shape':")
        .ok_or_else(|| format!("{name}: .npy header has no shape"))?;
    let open = header[shape_at..].find('(').map(|i| shape_at + i)
        .ok_or_else(|| format!("{name}: malformed shape"))?;
    let close = header[open..].find(')').map(|i| open + i)
        .ok_or_else(|| format!("{name}: malformed shape"))?;
    let mut shape = Vec::new();
    for part in header[open + 1..close].split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        shape.push(part.parse::<usize>().map_err(|_| format!("{name}: bad axis {part:?}"))?);
    }

    let count: usize = shape.iter().product();
    let start = body_at + header_len;
    let need = count * 4;
    if buf.len() < start + need {
        return Err(format!("{name}: truncated ({} bytes short)", start + need - buf.len()));
    }
    let mut data = Vec::with_capacity(count);
    for chunk in buf[start..start + need].chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Tensor { shape, data })
}

/// Load every array in an `.npz`, keyed by name with the `.npy` suffix stripped.
pub fn load_npz(path: &Path) -> Result<Weights, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 22 {
        return Err(format!("{}: too small to be a .npz", path.display()));
    }

    // Walk back from the end for the end-of-central-directory record; the
    // comment is empty for numpy's writer, so this finds it immediately.
    let eocd = (0..=(bytes.len() - 22).min(0xffff))
        .map(|back| bytes.len() - 22 - back)
        .find(|&at| &bytes[at..at + 4] == b"PK\x05\x06")
        .ok_or_else(|| format!("{}: no zip end-of-central-directory", path.display()))?;
    let entries = u16le(&bytes, eocd + 10);
    let mut at = u32le(&bytes, eocd + 16);

    let mut out = Weights::with_capacity(entries);
    for _ in 0..entries {
        if at + 46 > bytes.len() || &bytes[at..at + 4] != b"PK\x01\x02" {
            return Err(format!("{}: corrupt central directory", path.display()));
        }
        let method = u16le(&bytes, at + 10);
        let compressed = u32le(&bytes, at + 20);
        let name_len = u16le(&bytes, at + 28);
        let extra_len = u16le(&bytes, at + 30);
        let comment_len = u16le(&bytes, at + 32);
        let local = u32le(&bytes, at + 42);
        let name = String::from_utf8_lossy(&bytes[at + 46..at + 46 + name_len]).into_owned();
        at += 46 + name_len + extra_len + comment_len;

        if method != 0 {
            return Err(format!(
                "{}: member {name} is compressed (method {method}); \
                 this reader handles np.savez, not np.savez_compressed",
                path.display()
            ));
        }
        // The local header repeats the name and extra fields, with its own
        // lengths -- the central directory's extra_len does not apply here.
        if local + 30 > bytes.len() || &bytes[local..local + 4] != b"PK\x03\x04" {
            return Err(format!("{}: bad local header for {name}", path.display()));
        }
        let start = local + 30 + u16le(&bytes, local + 26) + u16le(&bytes, local + 28);
        let end = start + compressed;
        if end > bytes.len() {
            return Err(format!("{}: member {name} runs past end of file", path.display()));
        }
        let key = name.strip_suffix(".npy").unwrap_or(&name).to_string();
        out.insert(key, parse_npy(&name, &bytes[start..end])?);
    }
    Ok(out)
}
