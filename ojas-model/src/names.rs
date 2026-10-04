//! Parameter names, shapes, init rule and optimizer group: the table in
//! `docs/framework-design.md` §2.
//!
//! Names are nanolab `state_dict` keys, so a torch export loads unchanged.
//! The canonical order is [`param_table`]'s; [`ModelParams::from_flat`] and
//! [`ModelParams::into_flat`] use the same order, and a unit test pins each
//! field to its name.

use ojas_core::OjasError;
use ojas_cpu::{optim_group, OptimGroup};

use crate::spec::ModelSpec;

/// nanolab `_init_weights` standard deviation for every `nn.Linear` and
/// `nn.Embedding` weight.
pub const INIT_STD: f32 = 0.02;

/// How a parameter is initialized (nanolab `GPT.__init__`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Init {
    /// `N(0, std^2)`.
    Normal {
        std: f32,
    },
    Zeros,
    Ones,
}

/// One row of the §2 table.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub init: Init,
    /// `ojas_cpu::optim_group(ndim, embedding)`: the one owner of that policy.
    pub group: OptimGroup,
    /// False only for layer 0's `vr_lambda`: layer 0 has no earlier values
    /// to blend, so the parameter never receives a gradient and the
    /// optimizer skips it, as torch skips a parameter whose `.grad` is None.
    pub trains: bool,
}

impl ParamInfo {
    /// Element count. The spec's [`ModelSpec::validate`] bounds it.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// One block's parameters, in the canonical order of the fields.
#[derive(Clone, Debug, PartialEq)]
pub struct BlockParams<V> {
    pub norm1: V,
    pub q_proj: V,
    pub k_proj: V,
    pub v_proj: V,
    pub o_proj: V,
    pub q_norm: V,
    pub k_norm: V,
    pub gate_w: V,
    pub gate_b: V,
    pub vr_lambda: V,
    pub norm2: V,
    pub ffn_gate: V,
    pub ffn_up: V,
    pub ffn_down: V,
}

/// Fields per block. [`BlockParams::from_iter`] and [`BlockParams::push_into`]
/// read and write exactly this many, in field order.
pub const BLOCK_PARAMS: usize = 14;

impl<V> BlockParams<V> {
    fn from_iter(items: &mut impl Iterator<Item = V>) -> Option<Self> {
        Some(Self {
            norm1: items.next()?,
            q_proj: items.next()?,
            k_proj: items.next()?,
            v_proj: items.next()?,
            o_proj: items.next()?,
            q_norm: items.next()?,
            k_norm: items.next()?,
            gate_w: items.next()?,
            gate_b: items.next()?,
            vr_lambda: items.next()?,
            norm2: items.next()?,
            ffn_gate: items.next()?,
            ffn_up: items.next()?,
            ffn_down: items.next()?,
        })
    }

    fn push_into(self, out: &mut Vec<V>) {
        out.extend([
            self.norm1,
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.o_proj,
            self.q_norm,
            self.k_norm,
            self.gate_w,
            self.gate_b,
            self.vr_lambda,
            self.norm2,
            self.ffn_gate,
            self.ffn_up,
            self.ffn_down,
        ]);
    }
}

/// Every parameter of the model. `tok_emb` is also the tied head.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelParams<V> {
    pub tok_emb: V,
    pub blocks: Vec<BlockParams<V>>,
    pub norm_f: V,
}

impl<V> ModelParams<V> {
    /// Build from values in [`param_table`] order. The count must be
    /// exactly `2 + BLOCK_PARAMS * n_layer`.
    pub fn from_flat(spec: &ModelSpec, flat: Vec<V>) -> Result<Self, OjasError> {
        let want = param_count(spec)?;
        if flat.len() != want {
            return Err(OjasError::Shape {
                op: "ModelParams::from_flat",
                detail: format!("{} values for {want} parameters", flat.len()),
            });
        }
        let mut items = flat.into_iter();
        let missing = || OjasError::Shape {
            op: "ModelParams::from_flat",
            detail: "parameter list ended early".to_string(),
        };
        let tok_emb = items.next().ok_or_else(missing)?;
        let mut blocks = Vec::with_capacity(spec.n_layer);
        for _ in 0..spec.n_layer {
            blocks.push(BlockParams::from_iter(&mut items).ok_or_else(missing)?);
        }
        let norm_f = items.next().ok_or_else(missing)?;
        Ok(Self {
            tok_emb,
            blocks,
            norm_f,
        })
    }

    /// Values in [`param_table`] order.
    pub fn into_flat(self) -> Vec<V> {
        let mut out = Vec::with_capacity(2 + BLOCK_PARAMS * self.blocks.len());
        out.push(self.tok_emb);
        for block in self.blocks {
            block.push_into(&mut out);
        }
        out.push(self.norm_f);
        out
    }
}

/// `2 + 14 * n_layer`.
pub fn param_count(spec: &ModelSpec) -> Result<usize, OjasError> {
    spec.n_layer
        .checked_mul(BLOCK_PARAMS)
        .and_then(|n| n.checked_add(2))
        .ok_or_else(|| OjasError::OutOfRange {
            op: "param_count",
            detail: "parameter count overflows".to_string(),
        })
}

/// Bytes of every parameter of `spec` as `F32`, from the §2 table.
pub fn param_bytes(spec: &ModelSpec) -> Result<u64, OjasError> {
    param_table(spec)?.iter().try_fold(0u64, |total, info| {
        let elems = ojas_core::shape_product(&info.shape)? as u64;
        elems
            .checked_mul(4)
            .and_then(|b| total.checked_add(b))
            .ok_or_else(|| OjasError::OutOfRange {
                op: "param_bytes",
                detail: "parameter bytes overflow u64".to_string(),
            })
    })
}

/// The §2 table for `spec`, in canonical order.
pub fn param_table(spec: &ModelSpec) -> Result<Vec<ParamInfo>, OjasError> {
    spec.validate()?;
    let (d, v, hidden) = (spec.n_embd, spec.vocab, spec.hidden);
    let (q, kv, dh, nh) = (spec.q_width(), spec.kv_width(), spec.head_dim, spec.n_head);
    let normal = Init::Normal { std: INIT_STD };
    let mut rows = Vec::with_capacity(param_count(spec)?);
    let mut push = |name: String, shape: Vec<usize>, init: Init, embedding: bool, trains: bool| {
        let group = optim_group(shape.len(), embedding);
        rows.push(ParamInfo {
            name,
            shape,
            init,
            group,
            trains,
        });
    };
    push("tok_emb.weight".into(), vec![v, d], normal, true, true);
    for i in 0..spec.n_layer {
        let p = format!("blocks.{i}");
        let m = format!("{p}.mixer");
        push(
            format!("{p}.norm1.weight"),
            vec![d],
            Init::Ones,
            false,
            true,
        );
        push(
            format!("{m}.q_proj.weight"),
            vec![q, d],
            normal,
            false,
            true,
        );
        push(
            format!("{m}.k_proj.weight"),
            vec![kv, d],
            normal,
            false,
            true,
        );
        push(
            format!("{m}.v_proj.weight"),
            vec![kv, d],
            normal,
            false,
            true,
        );
        // `_zero_init_output_projections` (model.py:239-240,263-271).
        push(
            format!("{m}.o_proj.weight"),
            vec![d, q],
            Init::Zeros,
            false,
            true,
        );
        push(
            format!("{m}.q_norm.weight"),
            vec![dh],
            Init::Ones,
            false,
            true,
        );
        push(
            format!("{m}.k_norm.weight"),
            vec![dh],
            Init::Ones,
            false,
            true,
        );
        // mixers.py zeroes the gate weight, then `self.apply(_init_weights)`
        // in GPT.__init__ redraws it from N(0, 0.02) and zeroes the bias.
        push(format!("{m}.gate.weight"), vec![nh, d], normal, false, true);
        push(format!("{m}.gate.bias"), vec![nh], Init::Zeros, false, true);
        push(
            format!("{m}.vr_lambda"),
            vec![1],
            Init::Zeros,
            false,
            i != 0,
        );
        push(
            format!("{p}.norm2.weight"),
            vec![d],
            Init::Ones,
            false,
            true,
        );
        push(
            format!("{p}.ffn.gate.weight"),
            vec![hidden, d],
            normal,
            false,
            true,
        );
        push(
            format!("{p}.ffn.up.weight"),
            vec![hidden, d],
            normal,
            false,
            true,
        );
        push(
            format!("{p}.ffn.down.weight"),
            vec![d, hidden],
            Init::Zeros,
            false,
            true,
        );
    }
    push("norm_f.weight".into(), vec![d], Init::Ones, false, true);
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(spec: &ModelSpec) -> Vec<ParamInfo> {
        param_table(spec).unwrap()
    }

    #[test]
    fn each_field_carries_its_nanolab_name() {
        let spec = ModelSpec::tiny();
        let names: Vec<String> = table(&spec).into_iter().map(|p| p.name).collect();
        let p = ModelParams::from_flat(&spec, names.clone()).unwrap();
        assert_eq!(p.tok_emb, "tok_emb.weight");
        assert_eq!(p.norm_f, "norm_f.weight");
        let b = &p.blocks[1];
        let want = [
            (&b.norm1, "blocks.1.norm1.weight"),
            (&b.q_proj, "blocks.1.mixer.q_proj.weight"),
            (&b.k_proj, "blocks.1.mixer.k_proj.weight"),
            (&b.v_proj, "blocks.1.mixer.v_proj.weight"),
            (&b.o_proj, "blocks.1.mixer.o_proj.weight"),
            (&b.q_norm, "blocks.1.mixer.q_norm.weight"),
            (&b.k_norm, "blocks.1.mixer.k_norm.weight"),
            (&b.gate_w, "blocks.1.mixer.gate.weight"),
            (&b.gate_b, "blocks.1.mixer.gate.bias"),
            (&b.vr_lambda, "blocks.1.mixer.vr_lambda"),
            (&b.norm2, "blocks.1.norm2.weight"),
            (&b.ffn_gate, "blocks.1.ffn.gate.weight"),
            (&b.ffn_up, "blocks.1.ffn.up.weight"),
            (&b.ffn_down, "blocks.1.ffn.down.weight"),
        ];
        for (got, name) in want {
            assert_eq!(got, name);
        }
        assert_eq!(p.into_flat(), names);
    }

    #[test]
    fn shapes_inits_and_groups_follow_the_design_table() {
        let spec = ModelSpec::nanolab_124m();
        let rows = table(&spec);
        assert_eq!(rows.len(), 2 + 14 * 12);
        let find = |name: &str| rows.iter().find(|r| r.name == name).unwrap();
        let normal = Init::Normal { std: 0.02 };
        let cases: &[(&str, &[usize], Init, OptimGroup)] = &[
            (
                "tok_emb.weight",
                &[50304, 768],
                normal,
                OptimGroup::AdamEmbedding,
            ),
            (
                "blocks.3.norm1.weight",
                &[768],
                Init::Ones,
                OptimGroup::AdamVector,
            ),
            (
                "blocks.3.mixer.q_proj.weight",
                &[768, 768],
                normal,
                OptimGroup::MuonMatrix,
            ),
            (
                "blocks.3.mixer.o_proj.weight",
                &[768, 768],
                Init::Zeros,
                OptimGroup::MuonMatrix,
            ),
            (
                "blocks.3.mixer.q_norm.weight",
                &[64],
                Init::Ones,
                OptimGroup::AdamVector,
            ),
            (
                "blocks.3.mixer.gate.weight",
                &[12, 768],
                normal,
                OptimGroup::MuonMatrix,
            ),
            (
                "blocks.3.mixer.gate.bias",
                &[12],
                Init::Zeros,
                OptimGroup::AdamVector,
            ),
            (
                "blocks.3.mixer.vr_lambda",
                &[1],
                Init::Zeros,
                OptimGroup::AdamVector,
            ),
            (
                "blocks.3.ffn.gate.weight",
                &[2048, 768],
                normal,
                OptimGroup::MuonMatrix,
            ),
            (
                "blocks.3.ffn.up.weight",
                &[2048, 768],
                normal,
                OptimGroup::MuonMatrix,
            ),
            (
                "blocks.3.ffn.down.weight",
                &[768, 2048],
                Init::Zeros,
                OptimGroup::MuonMatrix,
            ),
            ("norm_f.weight", &[768], Init::Ones, OptimGroup::AdamVector),
        ];
        for (name, shape, init, group) in cases {
            let row = find(name);
            assert_eq!(row.shape, *shape, "{name}");
            assert_eq!(row.init, *init, "{name}");
            assert_eq!(row.group, *group, "{name}");
            assert!(row.trains, "{name}");
        }
        let total: usize = rows.iter().map(ParamInfo::numel).sum();
        // nanolab's tied 124M (pytorch-parity-plan): 50304*768 + 12 * per-layer + 768.
        let per_layer = 768 * 2 + 4 * 768 * 768 + 2 * 64 + 12 * 768 + 12 + 1 + 3 * 2048 * 768;
        assert_eq!(total, 50304 * 768 + 12 * per_layer + 768);
    }

    #[test]
    fn only_layer_zero_vr_lambda_is_frozen() {
        let rows = table(&ModelSpec::tiny());
        let frozen: Vec<&str> = rows
            .iter()
            .filter(|r| !r.trains)
            .map(|r| r.name.as_str())
            .collect();
        assert_eq!(frozen, ["blocks.0.mixer.vr_lambda"]);
    }

    #[test]
    fn from_flat_refuses_a_wrong_count() {
        let spec = ModelSpec::tiny();
        let n = param_count(&spec).unwrap();
        assert!(ModelParams::from_flat(&spec, vec![0u8; n - 1]).is_err());
        assert!(ModelParams::from_flat(&spec, vec![0u8; n + 1]).is_err());
        assert!(ModelParams::from_flat(&spec, vec![0u8; n]).is_ok());
    }
}
