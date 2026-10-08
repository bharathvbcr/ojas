//! Affine Q4 / Q8 quantization for Hot weight banks.
//!
//! Group-wise affine: `x ≈ scale * (q - zero)` with optional symmetric (`zero=0`).
//! Q4 packs two int4 values per byte (lo nibble = even index).
//! BF16 passthrough is available for debug dumps (not the ship path).

use crate::diag;
use crate::error::{Error, Result};

/// Target storage for Hot banks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantScheme {
    /// 4-bit affine, group_size along K (in-features).
    /// Signed nibble [-8,7]: `scale * (q - zero)`.
    Q4 { group_size: usize },
    /// MLX affine Q4 (unsigned nibble [0,15]): `scale * q + bias` (bias in `zeros`).
    /// Packing matches [`Q4`]; group_size typically 64.
    Q4Mlx { group_size: usize },
    /// 8-bit affine, group_size along K.
    Q8 { group_size: usize },
    /// Debug: store bf16 bits (u16), no quant.
    Bf16,
}

impl QuantScheme {
    pub fn q4_default() -> Self {
        // Challenge stacks often use g128; 32 is a solid decode default until A/B.
        QuantScheme::Q4 { group_size: 32 }
    }

    pub fn q4_mlx_default() -> Self {
        QuantScheme::Q4Mlx { group_size: 64 }
    }

    pub fn q8_default() -> Self {
        QuantScheme::Q8 { group_size: 32 }
    }

    pub fn group_size(self) -> Option<usize> {
        match self {
            QuantScheme::Q4 { group_size }
            | QuantScheme::Q4Mlx { group_size }
            | QuantScheme::Q8 { group_size } => Some(group_size),
            QuantScheme::Bf16 => None,
        }
    }

    pub fn is_mlx_affine(self) -> bool {
        matches!(self, QuantScheme::Q4Mlx { .. })
    }
}

/// Host-side quantized matrix `[out_features, in_features]` row-major.
#[derive(Clone, Debug)]
pub struct QuantMatrix {
    pub scheme: QuantScheme,
    pub rows: usize,
    pub cols: usize,
    /// Packed q4 (nibbles) or q8 bytes; empty for Bf16.
    pub packed: Vec<u8>,
    /// Per-group scales (f32).
    pub scales: Vec<f32>,
    /// Per-group zero points (f32); all-zero for symmetric.
    pub zeros: Vec<f32>,
    /// BF16 bits when `scheme == Bf16`.
    pub bf16_bits: Vec<u16>,
}

impl QuantMatrix {
    pub fn nbytes_hot(&self) -> usize {
        match self.scheme {
            QuantScheme::Bf16 => self.bf16_bits.len() * 2,
            _ => self.packed.len() + self.scales.len() * 4 + self.zeros.len() * 4,
        }
    }

    /// Dequantize to f32 (debug / parity / host embed row).
    pub fn dequant_f32(&self) -> Result<Vec<f32>> {
        match self.scheme {
            QuantScheme::Bf16 => Ok(self
                .bf16_bits
                .iter()
                .copied()
                .map(bf16_bits_to_f32)
                .collect()),
            QuantScheme::Q4 { group_size } => {
                dequant_q4(self.rows, self.cols, group_size, &self.packed, &self.scales, &self.zeros)
            }
            QuantScheme::Q4Mlx { group_size } => dequant_q4_mlx(
                self.rows,
                self.cols,
                group_size,
                &self.packed,
                &self.scales,
                &self.zeros,
            ),
            QuantScheme::Q8 { group_size } => {
                dequant_q8(self.rows, self.cols, group_size, &self.packed, &self.scales, &self.zeros)
            }
        }
    }

    /// Dequantize a single row (host embed lookup).
    pub fn dequant_row(&self, row: usize) -> Result<Vec<f32>> {
        if row >= self.rows {
            return Err(Error::Quant(format!(
                "dequant_row {row} >= rows {}",
                self.rows
            )));
        }
        match self.scheme {
            QuantScheme::Bf16 => {
                let start = row * self.cols;
                Ok(self.bf16_bits[start..start + self.cols]
                    .iter()
                    .copied()
                    .map(bf16_bits_to_f32)
                    .collect())
            }
            QuantScheme::Q4 { group_size } => dequant_q4_row(
                row,
                self.cols,
                group_size,
                &self.packed,
                &self.scales,
                &self.zeros,
                /*mlx*/ false,
            ),
            QuantScheme::Q4Mlx { group_size } => dequant_q4_row(
                row,
                self.cols,
                group_size,
                &self.packed,
                &self.scales,
                &self.zeros,
                /*mlx*/ true,
            ),
            QuantScheme::Q8 { group_size } => {
                let groups_per_row = self.cols / group_size;
                let mut out = vec![0f32; self.cols];
                for c in 0..self.cols {
                    let idx = row * self.cols + c;
                    let q = self.packed[idx] as i8;
                    let g = c / group_size;
                    let gi = row * groups_per_row + g;
                    out[c] = self.scales[gi] * (q as f32 - self.zeros[gi]);
                }
                Ok(out)
            }
        }
    }
}

/// Build a [`QuantMatrix`] from MLX affine Q4 tensors (u32 pack + bf16/f32 scales/biases).
///
/// MLX stores `weight` as `u32[rows, cols/8]` (8 nibbles/u32, little-endian bytes =
/// our packed layout). `scales`/`biases` are `[rows, cols/group_size]`.
pub fn quant_matrix_from_mlx_q4(
    rows: usize,
    cols: usize,
    group_size: usize,
    weight_u32: &[u32],
    scales: &[f32],
    biases: &[f32],
) -> Result<QuantMatrix> {
    if group_size == 0 || cols % group_size != 0 {
        let e = Error::Quant(format!(
            "MLX Q4 group_size {group_size} must divide cols {cols}"
        ));
        diag::err("quant", "quant_matrix_from_mlx_q4", &e);
        return Err(e);
    }
    let packs_per_row = cols / 8;
    if cols % 8 != 0 {
        let e = Error::Quant(format!(
            "MLX Q4 cols {cols} must be divisible by 8"
        ));
        diag::err("quant", "quant_matrix_from_mlx_q4", &e);
        return Err(e);
    }
    if weight_u32.len() != rows * packs_per_row {
        let e = Error::Quant(format!(
            "MLX weight u32 len {} != rows*cols/8 {}",
            weight_u32.len(),
            rows * packs_per_row
        ));
        diag::err("quant", "quant_matrix_from_mlx_q4", &e);
        return Err(e);
    }
    let groups_per_row = cols / group_size;
    let n_groups = rows * groups_per_row;
    if scales.len() != n_groups || biases.len() != n_groups {
        let e = Error::Quant(format!(
            "MLX scales/biases len {}/{} != groups {n_groups}",
            scales.len(),
            biases.len()
        ));
        diag::err("quant", "quant_matrix_from_mlx_q4", &e);
        return Err(e);
    }
    // u32 LE bytes == our nibble packing (lo = even index).
    let mut packed = Vec::with_capacity(rows * packs_per_row * 4);
    for &w in weight_u32 {
        packed.extend_from_slice(&w.to_le_bytes());
    }
    let m = QuantMatrix {
        scheme: QuantScheme::Q4Mlx { group_size },
        rows,
        cols,
        packed,
        scales: scales.to_vec(),
        zeros: biases.to_vec(), // bias stored in zeros slot for Hot upload
        bf16_bits: Vec::new(),
    };
    // Only chatty for large mats (lm_head / big proj) — skip tiny unit-test shapes.
    if m.nbytes_hot() >= 1024 * 1024 {
        diag::log(
            "quant",
            format_args!(
                "MLX Q4 [{rows}x{cols}] g={group_size} packed={} scales={} hot={}",
                diag::fmt_bytes(m.packed.len() as u64),
                n_groups,
                diag::fmt_bytes(m.nbytes_hot() as u64)
            ),
        );
    }
    Ok(m)
}

/// Quantize an f32 row-major matrix.
pub fn quantize_affine_f32(rows: usize, cols: usize, data: &[f32], scheme: QuantScheme) -> Result<QuantMatrix> {
    if data.len() != rows * cols {
        return Err(Error::Quant(format!(
            "data len {} != rows*cols {}",
            data.len(),
            rows * cols
        )));
    }
    match scheme {
        QuantScheme::Bf16 => {
            let bf16_bits = data.iter().copied().map(f32_to_bf16_bits).collect();
            Ok(QuantMatrix {
                scheme,
                rows,
                cols,
                packed: Vec::new(),
                scales: Vec::new(),
                zeros: Vec::new(),
                bf16_bits,
            })
        }
        QuantScheme::Q4 { group_size } => {
            quant_q4(rows, cols, data, group_size, /*symmetric*/ true)
        }
        QuantScheme::Q4Mlx { group_size } => quant_q4_mlx(rows, cols, data, group_size),
        QuantScheme::Q8 { group_size } => {
            quant_q8(rows, cols, data, group_size, /*symmetric*/ true)
        }
    }
}

/// Quantize bf16 bits (u16) without host f32 round-trip when possible.
pub fn quantize_affine_bf16_bits(
    rows: usize,
    cols: usize,
    bits: &[u16],
    scheme: QuantScheme,
) -> Result<QuantMatrix> {
    if bits.len() != rows * cols {
        return Err(Error::Quant(format!(
            "bf16 len {} != rows*cols {}",
            bits.len(),
            rows * cols
        )));
    }
    match scheme {
        QuantScheme::Bf16 => Ok(QuantMatrix {
            scheme,
            rows,
            cols,
            packed: Vec::new(),
            scales: Vec::new(),
            zeros: Vec::new(),
            bf16_bits: bits.to_vec(),
        }),
        other => {
            let f: Vec<f32> = bits.iter().copied().map(bf16_bits_to_f32).collect();
            quantize_affine_f32(rows, cols, &f, other)
        }
    }
}

fn quant_q4(
    rows: usize,
    cols: usize,
    data: &[f32],
    group_size: usize,
    symmetric: bool,
) -> Result<QuantMatrix> {
    if group_size == 0 || cols % group_size != 0 {
        return Err(Error::Quant(format!(
            "Q4 group_size {group_size} must divide cols {cols}"
        )));
    }
    let groups_per_row = cols / group_size;
    let n_groups = rows * groups_per_row;
    let mut scales = vec![0f32; n_groups];
    let mut zeros = vec![0f32; n_groups];
    let packed_len = (rows * cols + 1) / 2;
    let mut packed = vec![0u8; packed_len];

    for r in 0..rows {
        for g in 0..groups_per_row {
            let base = r * cols + g * group_size;
            let slice = &data[base..base + group_size];
            let (scale, zero) = affine_params(slice, /*qmax*/ 7.0, /*qmin*/ -8.0, symmetric);
            let gi = r * groups_per_row + g;
            scales[gi] = scale;
            zeros[gi] = zero;
            for (i, &x) in slice.iter().enumerate() {
                let q = ((x / scale) + zero).round().clamp(-8.0, 7.0) as i8;
                let nibble = (q as i8) as u8 & 0x0f;
                let idx = r * cols + g * group_size + i;
                let byte_i = idx / 2;
                if idx % 2 == 0 {
                    packed[byte_i] = (packed[byte_i] & 0xf0) | nibble;
                } else {
                    packed[byte_i] = (packed[byte_i] & 0x0f) | (nibble << 4);
                }
            }
        }
    }

    Ok(QuantMatrix {
        scheme: QuantScheme::Q4 { group_size },
        rows,
        cols,
        packed,
        scales,
        zeros,
        bf16_bits: Vec::new(),
    })
}

/// MLX affine Q4: unsigned nibble [0,15], `x ≈ scale * q + bias` (bias stored in `zeros`).
fn quant_q4_mlx(
    rows: usize,
    cols: usize,
    data: &[f32],
    group_size: usize,
) -> Result<QuantMatrix> {
    if group_size == 0 || cols % group_size != 0 {
        return Err(Error::Quant(format!(
            "Q4Mlx group_size {group_size} must divide cols {cols}"
        )));
    }
    let groups_per_row = cols / group_size;
    let n_groups = rows * groups_per_row;
    let mut scales = vec![0f32; n_groups];
    let mut biases = vec![0f32; n_groups];
    let packed_len = (rows * cols + 1) / 2;
    let mut packed = vec![0u8; packed_len];

    for r in 0..rows {
        for g in 0..groups_per_row {
            let base = r * cols + g * group_size;
            let slice = &data[base..base + group_size];
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &x in slice {
                lo = lo.min(x);
                hi = hi.max(x);
            }
            if !lo.is_finite() || !hi.is_finite() {
                lo = 0.0;
                hi = 0.0;
            }
            let range = (hi - lo).max(1e-8);
            let scale = range / 15.0;
            let bias = lo;
            let gi = r * groups_per_row + g;
            scales[gi] = scale;
            biases[gi] = bias;
            for (i, &x) in slice.iter().enumerate() {
                let q = ((x - bias) / scale).round().clamp(0.0, 15.0) as u8;
                let idx = r * cols + g * group_size + i;
                let byte_i = idx / 2;
                if idx % 2 == 0 {
                    packed[byte_i] = (packed[byte_i] & 0xf0) | (q & 0x0f);
                } else {
                    packed[byte_i] = (packed[byte_i] & 0x0f) | ((q & 0x0f) << 4);
                }
            }
        }
    }

    Ok(QuantMatrix {
        scheme: QuantScheme::Q4Mlx { group_size },
        rows,
        cols,
        packed,
        scales,
        zeros: biases,
        bf16_bits: Vec::new(),
    })
}

fn quant_q8(
    rows: usize,
    cols: usize,
    data: &[f32],
    group_size: usize,
    symmetric: bool,
) -> Result<QuantMatrix> {
    if group_size == 0 || cols % group_size != 0 {
        return Err(Error::Quant(format!(
            "Q8 group_size {group_size} must divide cols {cols}"
        )));
    }
    let groups_per_row = cols / group_size;
    let n_groups = rows * groups_per_row;
    let mut scales = vec![0f32; n_groups];
    let mut zeros = vec![0f32; n_groups];
    let mut packed = vec![0u8; rows * cols];

    for r in 0..rows {
        for g in 0..groups_per_row {
            let base = r * cols + g * group_size;
            let slice = &data[base..base + group_size];
            let (scale, zero) = affine_params(slice, 127.0, -128.0, symmetric);
            let gi = r * groups_per_row + g;
            scales[gi] = scale;
            zeros[gi] = zero;
            for (i, &x) in slice.iter().enumerate() {
                let q = ((x / scale) + zero).round().clamp(-128.0, 127.0) as i8;
                packed[base + i] = q as u8;
            }
        }
    }

    Ok(QuantMatrix {
        scheme: QuantScheme::Q8 { group_size },
        rows,
        cols,
        packed,
        scales,
        zeros,
        bf16_bits: Vec::new(),
    })
}

fn affine_params(slice: &[f32], qmax: f32, qmin: f32, symmetric: bool) -> (f32, f32) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for &x in slice {
        mn = mn.min(x);
        mx = mx.max(x);
    }
    if !mn.is_finite() || !mx.is_finite() {
        return (1.0, 0.0);
    }
    if symmetric {
        let a = mn.abs().max(mx.abs()).max(1e-8);
        (a / qmax, 0.0)
    } else {
        let range = (mx - mn).max(1e-8);
        let scale = range / (qmax - qmin);
        let zero = qmin - mn / scale;
        (scale, zero)
    }
}

fn dequant_q4(
    rows: usize,
    cols: usize,
    group_size: usize,
    packed: &[u8],
    scales: &[f32],
    zeros: &[f32],
) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        out.extend(dequant_q4_row(
            r, cols, group_size, packed, scales, zeros, /*mlx*/ false,
        )?);
    }
    Ok(out)
}

fn dequant_q4_mlx(
    rows: usize,
    cols: usize,
    group_size: usize,
    packed: &[u8],
    scales: &[f32],
    biases: &[f32],
) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        out.extend(dequant_q4_row(
            r, cols, group_size, packed, scales, biases, /*mlx*/ true,
        )?);
    }
    Ok(out)
}

fn dequant_q4_row(
    row: usize,
    cols: usize,
    group_size: usize,
    packed: &[u8],
    scales: &[f32],
    zeros_or_biases: &[f32],
    mlx: bool,
) -> Result<Vec<f32>> {
    if group_size == 0 || cols % group_size != 0 {
        return Err(Error::Quant(format!(
            "dequant_q4_row: bad group_size {group_size} cols {cols}"
        )));
    }
    let groups_per_row = cols / group_size;
    let mut out = vec![0f32; cols];
    for c in 0..cols {
        let idx = row * cols + c;
        let byte = packed[idx / 2];
        let nibble = if idx % 2 == 0 {
            byte & 0x0f
        } else {
            (byte >> 4) & 0x0f
        };
        let g = c / group_size;
        let gi = row * groups_per_row + g;
        let s = scales[gi];
        let z = zeros_or_biases[gi];
        out[c] = if mlx {
            s * (nibble as f32) + z
        } else {
            let q = if nibble & 0x8 != 0 {
                (nibble as i8) | !0x0fi8
            } else {
                nibble as i8
            };
            s * (q as f32 - z)
        };
    }
    Ok(out)
}

fn dequant_q8(
    rows: usize,
    cols: usize,
    group_size: usize,
    packed: &[u8],
    scales: &[f32],
    zeros: &[f32],
) -> Result<Vec<f32>> {
    let groups_per_row = cols / group_size;
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let q = packed[idx] as i8;
            let g = c / group_size;
            let gi = r * groups_per_row + g;
            out[idx] = scales[gi] * (q as f32 - zeros[gi]);
        }
    }
    Ok(out)
}

/// Truncate f32 → bf16 bits (round-to-nearest-even lite).
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let round_bit = (bits >> 15) & 1;
    ((bits >> 16) as u16).wrapping_add(round_bit as u16)
}

pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q4_roundtrip_rough() {
        let rows = 4;
        let cols = 32;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
            .collect();
        let q = quantize_affine_f32(rows, cols, &data, QuantScheme::q4_default()).unwrap();
        let back = q.dequant_f32().unwrap();
        let mut max_err = 0f32;
        for (a, b) in data.iter().zip(back.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 0.15, "max_err={max_err}");
    }

    #[test]
    fn q8_roundtrip_tighter() {
        let rows = 2;
        let cols = 32;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| (i as f32) * 0.01 - 0.3)
            .collect();
        let q = quantize_affine_f32(rows, cols, &data, QuantScheme::q8_default()).unwrap();
        let back = q.dequant_f32().unwrap();
        let mut max_err = 0f32;
        for (a, b) in data.iter().zip(back.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 0.02, "max_err={max_err}");
    }

    #[test]
    fn mlx_q4_from_f32_roundtrip_rough() {
        let rows = 4;
        let cols = 64;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
            .collect();
        let q = quantize_affine_f32(rows, cols, &data, QuantScheme::q4_mlx_default()).unwrap();
        let back = q.dequant_f32().unwrap();
        let mut max_err = 0f32;
        for (a, b) in data.iter().zip(back.iter()) {
            max_err = max_err.max((a - b).abs());
        }
        assert!(max_err < 0.2, "max_err={max_err}");
    }
}
