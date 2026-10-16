//! Localize the native DFlash accept≈0 bug: diff the MLX golden against the
//! native intermediates and report the FIRST divergent tensor (that is the
//! bug's layer). GPU-free.
//!
//! ```text
//! golden_parity --compare-intermediates bench/results/golden_intermediates_31b.json \
//!     bench/results/native_intermediates_31b.json
//! ```
//!
//! The MLX golden is produced by `bench/golden_intermediates.py`; the native
//! side by `bench --dflash-31b` (`dump_31b_block1_intermediates` in
//! `src/bin/bench.rs`), with the same keys on input `[2,105,4368,1246]`:
//!   - `target_hidden_per_layer[6]`: `{sum, absmean, first8, shape}` (post-`layer_scalar` residual, last pos)
//!   - `fc_out_lastrow`, `h_ctx_lastrow`, `draft_logits_lastpos`: the same checksum dict
//!   - `draft_attn_offsets_per_layer[5]`: `{S, Lq, q_rope_offset, ctx_rope_offset, is_sliding, sliding_window}`
//!   - `proposed_block_tokens`: `[int]` (bs-1)
//!   - `target_next_argmax`: int
//!
//! Comparison order follows `docs/dflash_draft_contract.md`, so the first
//! mismatch names the suspect: target_hidden (capture) -> fc_out (fc) ->
//! h_ctx (hidden_norm) -> attn offsets (★2/★3) -> draft_logits -> proposed
//! tokens.
//!
//! Ported from `bench/compare_intermediates.py`; the report is the same text,
//! byte for byte (`tests/fixtures/compare_intermediates/` holds what the
//! Python printed). Values print as Python printed them, so the reader below
//! keeps each JSON number's literal: an integer stays `4`, a float prints in
//! Python's shortest-repr form (`1e+20`, `1.3428900241851807`).

use std::fmt::Write as _;

/// A parsed JSON value. Objects keep their keys in order; a repeated key
/// resolves to its last value, as Python's `json.load` does.
#[derive(Clone, Debug)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn parse(s: &str) -> Result<Json, String> {
        let mut p = Parser {
            b: s.as_bytes(),
            i: 0,
        };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        if p.i != p.b.len() {
            return Err(format!("extra data at byte {}", p.i));
        }
        Ok(v)
    }

    /// `d.get(k)`: the last value stored under `k`.
    pub fn get(&self, k: &str) -> Option<&Json> {
        match self {
            Json::Object(kv) => kv.iter().rev().find(|(key, _)| key == k).map(|(_, v)| v),
            _ => None,
        }
    }

    fn get_mut(&mut self, k: &str) -> Option<&mut Json> {
        match self {
            Json::Object(kv) => kv
                .iter_mut()
                .rev()
                .find(|(key, _)| key == k)
                .map(|(_, v)| v),
            _ => None,
        }
    }

    /// Replace or insert `k`, keeping its position when it exists.
    pub fn set(&mut self, k: &str, v: Json) {
        if let Some(slot) = self.get_mut(k) {
            *slot = v;
        } else if let Json::Object(kv) = self {
            kv.push((k.to_string(), v));
        }
    }

    pub fn remove(&mut self, k: &str) {
        if let Json::Object(kv) = self {
            kv.retain(|(key, _)| key != k);
        }
    }

    fn has(&self, k: &str) -> bool {
        self.get(k).is_some()
    }

    fn items(&self) -> &[Json] {
        match self {
            Json::Array(a) => a,
            _ => &[],
        }
    }

    pub fn items_mut(&mut self) -> Option<&mut Vec<Json>> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    /// The number as Python arithmetic sees it (`True` is 1).
    fn as_f64(&self) -> Option<f64> {
        match *self {
            Json::Int(i) => Some(i as f64),
            Json::Float(f) => Some(f),
            Json::Bool(b) => Some(b as u8 as f64),
            _ => None,
        }
    }
}

/// Python's `==` on decoded JSON: `1 == 1.0`, `True == 1`.
fn py_eq(a: &Json, b: &Json) -> bool {
    match (a, b) {
        (Json::Null, Json::Null) => true,
        (Json::Str(x), Json::Str(y)) => x == y,
        (Json::Array(x), Json::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        (Json::Object(_), Json::Object(_)) => {
            let keys = |o: &Json| -> Vec<String> {
                let mut k: Vec<String> = match o {
                    Json::Object(kv) => kv.iter().map(|(k, _)| k.clone()).collect(),
                    _ => vec![],
                };
                k.sort();
                k.dedup();
                k
            };
            let (ka, kb) = (keys(a), keys(b));
            ka == kb
                && ka
                    .iter()
                    .all(|k| py_eq(a.get(k).unwrap(), b.get(k).unwrap()))
        }
        (Json::Int(x), Json::Int(y)) => x == y,
        _ => match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        },
    }
}

fn opt_eq(a: Option<&Json>, b: Option<&Json>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => py_eq(x, y),
        (None, Some(Json::Null)) | (Some(Json::Null), None) => true,
        _ => false,
    }
}

/// Python's `repr(float)`.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let e = format!("{f:e}"); // shortest round-trip digits: "1.5e-7"
    let (mant, exp) = e.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    if (-4..16).contains(&exp) {
        let s = format!("{f}");
        if s.contains('.') {
            s
        } else {
            s + ".0"
        }
    } else {
        format!("{mant}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    }
}

/// Python's `str()` of a decoded JSON value, as an f-string prints it.
pub fn py_str(v: &Json) -> String {
    match v {
        Json::Null => "None".into(),
        Json::Bool(true) => "True".into(),
        Json::Bool(false) => "False".into(),
        Json::Int(i) => i.to_string(),
        Json::Float(f) => py_float_repr(*f),
        Json::Str(s) => s.clone(),
        Json::Array(a) => format!("[{}]", a.iter().map(py_repr).collect::<Vec<_>>().join(", ")),
        Json::Object(kv) => {
            let mut seen: Vec<&str> = Vec::new();
            let mut parts = Vec::new();
            for (k, _) in kv {
                if !seen.contains(&k.as_str()) {
                    seen.push(k);
                    parts.push(format!(
                        "{}: {}",
                        py_repr(&Json::Str(k.clone())),
                        py_repr(v.get(k).unwrap())
                    ));
                }
            }
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// Python's `repr()`: strings are quoted inside containers.
fn py_repr(v: &Json) -> String {
    match v {
        Json::Str(s) => {
            let q = if s.contains('\'') && !s.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut out = String::from(q);
            for c in s.chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c == q => {
                        out.push('\\');
                        out.push(c);
                    }
                    c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                        let _ = write!(out, "\\x{:02x}", c as u32);
                    }
                    c => out.push(c),
                }
            }
            out.push(q);
            out
        }
        other => py_str(other),
    }
}

fn opt_str(v: Option<&Json>) -> String {
    v.map(py_str).unwrap_or_else(|| "None".into())
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at byte {}", self.i))
    }

    fn lit(&mut self, s: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(v)
        } else {
            self.err("invalid literal")
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > 512 {
            return self.err("nesting too deep");
        }
        match self.b.get(self.i) {
            None => self.err("unexpected end"),
            Some(b'{') => {
                self.i += 1;
                let mut kv = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b'}') {
                    self.i += 1;
                    return Ok(Json::Object(kv));
                }
                loop {
                    self.ws();
                    if self.b.get(self.i) != Some(&b'"') {
                        return self.err("expected a key");
                    }
                    let k = self.string()?;
                    self.ws();
                    if self.b.get(self.i) != Some(&b':') {
                        return self.err("expected ':'");
                    }
                    self.i += 1;
                    self.ws();
                    let v = self.value(depth + 1)?;
                    kv.push((k, v));
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Object(kv));
                        }
                        _ => return self.err("expected ',' or '}'"),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.b.get(self.i) == Some(&b']') {
                    self.i += 1;
                    return Ok(Json::Array(a));
                }
                loop {
                    self.ws();
                    a.push(self.value(depth + 1)?);
                    self.ws();
                    match self.b.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Array(a));
                        }
                        _ => return self.err("expected ',' or ']'"),
                    }
                }
            }
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'n') => self.lit("null", Json::Null),
            Some(b'N') => self.lit("NaN", Json::Float(f64::NAN)),
            Some(b'I') => self.lit("Infinity", Json::Float(f64::INFINITY)),
            Some(b'-') if self.b[self.i..].starts_with(b"-Infinity") => {
                self.lit("-Infinity", Json::Float(f64::NEG_INFINITY))
            }
            Some(c) if *c == b'-' || c.is_ascii_digit() => self.number(),
            Some(_) => self.err("unexpected character"),
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        let digits = |p: &mut Self| {
            let s = p.i;
            while p.i < p.b.len() && p.b[p.i].is_ascii_digit() {
                p.i += 1;
            }
            p.i - s
        };
        if self.b[self.i] == b'-' {
            self.i += 1;
        }
        let int_start = self.i;
        if digits(self) == 0 || (self.b[int_start] == b'0' && self.i - int_start > 1) {
            return self.err("invalid number");
        }
        let mut float = false;
        if self.b.get(self.i) == Some(&b'.') {
            float = true;
            self.i += 1;
            if digits(self) == 0 {
                return self.err("invalid fraction");
            }
        }
        if matches!(self.b.get(self.i), Some(b'e' | b'E')) {
            float = true;
            self.i += 1;
            if matches!(self.b.get(self.i), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if digits(self) == 0 {
                return self.err("invalid exponent");
            }
        }
        let lit = std::str::from_utf8(&self.b[start..self.i]).expect("ASCII");
        if float {
            // `str::parse::<f64>` rounds correctly, as Python's float() does.
            lit.parse()
                .map(Json::Float)
                .map_err(|e| format!("{e}: {lit}"))
        } else {
            lit.parse()
                .map(Json::Int)
                .map_err(|e| format!("{e}: {lit}"))
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1; // opening quote
        let mut out = String::new();
        loop {
            let start = self.i;
            while self.i < self.b.len() && self.b[self.i] != b'"' && self.b[self.i] != b'\\' {
                if self.b[self.i] < 0x20 {
                    return self.err("control character in string");
                }
                self.i += 1;
            }
            out.push_str(std::str::from_utf8(&self.b[start..self.i]).map_err(|e| e.to_string())?);
            match self.b.get(self.i) {
                None => return self.err("unterminated string"),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                _ => {
                    self.i += 1;
                    let c = *self.b.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp)
                                && self.b[self.i..].starts_with(b"\\u")
                            {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xDC00..0xE000).contains(&lo) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                } else {
                                    self.i = save;
                                }
                            }
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                        _ => return self.err("invalid escape"),
                    }
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.b.get(self.i..self.i + 4).ok_or("short \\u escape")?;
        let s = std::str::from_utf8(h).map_err(|e| e.to_string())?;
        self.i += 4;
        u32::from_str_radix(s, 16).map_err(|e| format!("{e}: \\u{s}"))
    }
}

const CKSUM_ORDER: [(&str, &str); 4] = [
    ("target_hidden_per_layer", "★1 capture tap / feed"),
    ("fc_out_lastrow", "fc projection"),
    ("h_ctx_lastrow", "hidden_norm"),
    ("draft_logits_lastpos", "draft forward output"),
];

fn num(v: &Json, what: &str) -> Result<f64, String> {
    v.as_f64()
        .ok_or_else(|| format!("TypeError: {what} is not a number: {}", py_str(v)))
}

fn close(x: f64, y: f64, rtol: f64, atol: f64) -> bool {
    // Python's `not (abs(x - y) > tol)`: a NaN difference does not diverge.
    let tol = atol + rtol * x.abs().max(y.abs());
    (x - y).abs().partial_cmp(&tol) != Some(std::cmp::Ordering::Greater)
}

/// Compare two checksum dicts by absmean, sum, shape and first8, with
/// tolerance. `Err` is a malformed input; `Ok((false, why))` a divergence.
fn cks_close(a: &Json, b: &Json) -> Result<(bool, String), String> {
    let (rtol, atol) = (2e-2, 1e-2);
    for k in ["absmean", "sum"] {
        let (va, vb) = (a.get(k), b.get(k));
        let (Some(va), Some(vb)) = (
            va.filter(|v| !matches!(v, Json::Null)),
            vb.filter(|v| !matches!(v, Json::Null)),
        ) else {
            return Ok((false, format!("missing {k}")));
        };
        if !close(num(va, k)?, num(vb, k)?, rtol, atol) {
            return Ok((
                false,
                format!("{k}: golden {} vs native {}", py_str(va), py_str(vb)),
            ));
        }
    }
    if !opt_eq(a.get("shape"), b.get("shape")) {
        return Ok((
            false,
            format!(
                "shape {} vs {}",
                opt_str(a.get("shape")),
                opt_str(b.get("shape"))
            ),
        ));
    }
    let (fa, fb) = (
        a.get("first8").map(Json::items).unwrap_or(&[]),
        b.get("first8").map(Json::items).unwrap_or(&[]),
    );
    for (i, (x, y)) in fa.iter().zip(fb).enumerate() {
        if !close(num(x, "first8")?, num(y, "first8")?, rtol, atol) {
            return Ok((
                false,
                format!("first8[{i}]: {} vs {}", py_str(x), py_str(y)),
            ));
        }
    }
    Ok((true, "ok".into()))
}

fn verdict(ok: bool, why: &str) -> String {
    if ok {
        "OK".into()
    } else {
        format!("DIVERGE — {why}")
    }
}

/// The diff report for one golden/native pair.
pub fn compare(g: &Json, n: &Json) -> Result<String, String> {
    let mut out = String::from("=== DFlash intermediate diff (golden vs native) ===\n");
    let mut first_bad: Option<(String, String, String)> = None;

    // 1) per-layer captured hidden
    let gh = g
        .get("target_hidden_per_layer")
        .ok_or("KeyError: 'target_hidden_per_layer'")?
        .items();
    let nh = n
        .get("target_hidden_per_layer")
        .map(Json::items)
        .unwrap_or(&[]);
    let ids = g
        .get("target_layer_ids")
        .ok_or("KeyError: 'target_layer_ids'")?
        .items();
    for (i, gl) in gh.iter().enumerate() {
        let (ok, why) = match nh.get(i) {
            Some(nl) => cks_close(gl, nl)?,
            None => (false, "native missing layer".into()),
        };
        let lid = py_str(
            ids.get(i)
                .ok_or("IndexError: target_layer_ids is shorter than target_hidden_per_layer")?,
        );
        let _ = writeln!(out, "  target_hidden[layer {lid}]: {}", verdict(ok, &why));
        if !ok && first_bad.is_none() {
            first_bad = Some((
                format!("target_hidden[layer {lid}]"),
                "★1 capture tap/feed".into(),
                why,
            ));
        }
    }

    // 2) scalar checksum fields in contract order
    for (key, r) in &CKSUM_ORDER[1..] {
        if g.has(key) && n.has(key) {
            let (ok, why) = cks_close(g.get(key).unwrap(), n.get(key).unwrap())?;
            let _ = writeln!(out, "  {key}: {}  ({r})", verdict(ok, &why));
            if !ok && first_bad.is_none() {
                first_bad = Some((key.to_string(), r.to_string(), why));
            }
        } else {
            let _ = writeln!(out, "  {key}: native MISSING (emit it)");
        }
    }

    // 3) RoPE offsets per draft layer (★2/★3)
    let go = g
        .get("draft_attn_offsets_per_layer")
        .map(Json::items)
        .unwrap_or(&[]);
    let no = n
        .get("draft_attn_offsets_per_layer")
        .map(Json::items)
        .unwrap_or(&[]);
    let empty = Json::Object(Vec::new());
    for (i, gi) in go.iter().enumerate() {
        let ni = no.get(i).unwrap_or(&empty);
        let bad: Vec<&str> = ["q_rope_offset", "ctx_rope_offset", "S"]
            .into_iter()
            .filter(|k| !opt_eq(gi.get(k), ni.get(k)))
            .collect();
        let tag = if bad.is_empty() {
            "OK".to_string()
        } else {
            let parts: Vec<String> = bad
                .iter()
                .map(|k| format!("{k}: {} vs {}", opt_str(gi.get(k)), opt_str(ni.get(k))))
                .collect();
            format!("DIVERGE — {}", parts.join(", "))
        };
        let _ = writeln!(
            out,
            "  draft L{i} offsets (sliding={}): {tag}",
            opt_str(gi.get("is_sliding"))
        );
        if !bad.is_empty() && first_bad.is_none() {
            first_bad = Some((
                format!("draft L{i} RoPE offset"),
                "★2/★3 offset".into(),
                tag,
            ));
        }
    }

    // 4) proposed tokens
    let (gp, np) = (
        g.get("proposed_block_tokens"),
        n.get("proposed_block_tokens"),
    );
    let same = opt_eq(gp, np);
    let _ = writeln!(
        out,
        "  proposed_block_tokens: golden {} vs native {} — {}",
        opt_str(gp),
        opt_str(np),
        if same { "MATCH" } else { "DIFFER" }
    );

    out.push_str("\n=== VERDICT ===\n");
    if let Some((what, suspect, detail)) = first_bad {
        let _ = writeln!(out, "FIRST DIVERGENCE: {what}  ->  suspect: {suspect}");
        let _ = writeln!(out, "  detail: {detail}");
        out.push_str("  Fix this layer; everything downstream diverges from it.\n");
    } else if !same {
        out.push_str("Intermediates all match but proposed tokens differ -> sampling/argmax or logits-softcap in draft head.\n");
    } else {
        out.push_str(
            "Full match — draft is correct; if accept still ≈0 the bug is in the VERIFY path \
             (target argmax offset / seed), not the draft. See contract closing note.\n",
        );
    }
    Ok(out)
}

/// Read both files and compare them.
pub fn compare_files(golden: &std::path::Path, native: &std::path::Path) -> Result<String, String> {
    let read = |p: &std::path::Path| -> Result<Json, String> {
        let s = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
        Json::parse(&s).map_err(|e| format!("{}: {e}", p.display()))
    };
    compare(&read(golden)?, &read(native)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn results(name: &str) -> Json {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("bench/results")
            .join(name);
        Json::parse(&std::fs::read_to_string(p).unwrap()).unwrap()
    }

    fn want(name: &str) -> String {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/compare_intermediates")
            .join(name);
        std::fs::read_to_string(p).unwrap()
    }

    fn layer<'a>(v: &'a mut Json, list: &str, i: usize) -> &'a mut Json {
        &mut v.get_mut(list).unwrap().items_mut().unwrap()[i]
    }

    /// The same edits the oracle run made to its copies (see the fixture README).
    fn mutated(case: &str) -> (Json, Json) {
        let g = results("golden_intermediates_31b.json");
        let n = results("native_intermediates_31b.json");
        match case {
            "a" => (g, n),
            "b" => (n, g),
            "c" => (g.clone(), g),
            "d" => {
                let mut x = g.clone();
                x.get_mut("target_hidden_per_layer")
                    .unwrap()
                    .items_mut()
                    .unwrap()
                    .truncate(2);
                x.remove("fc_out_lastrow");
                x.get_mut("draft_attn_offsets_per_layer")
                    .unwrap()
                    .items_mut()
                    .unwrap()
                    .truncate(3);
                layer(&mut x, "draft_attn_offsets_per_layer", 1).set("q_rope_offset", Json::Int(9));
                x.set("proposed_block_tokens", Json::Null);
                (g, x)
            }
            "e" => {
                let mut x = g.clone();
                layer(&mut x, "target_hidden_per_layer", 0).set("absmean", Json::Null);
                x.get_mut("fc_out_lastrow")
                    .unwrap()
                    .set("shape", Json::Array(vec![Json::Int(1), Json::Int(2)]));
                let f8 = x
                    .get_mut("h_ctx_lastrow")
                    .unwrap()
                    .get_mut("first8")
                    .unwrap()
                    .items_mut()
                    .unwrap();
                f8[3] = Json::Float(f8[3].as_f64().unwrap() + 1.0);
                x.get_mut("draft_logits_lastpos")
                    .unwrap()
                    .set("sum", Json::Float(1e20));
                (g, x)
            }
            "f" => {
                let mut x = g.clone();
                x.set(
                    "proposed_block_tokens",
                    Json::Array((1..=4).map(Json::Int).collect()),
                );
                (g, x)
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn report_matches_the_python_oracle() {
        for case in ["a", "b", "c", "d", "e", "f"] {
            let (g, n) = mutated(case);
            let got = compare(&g, &n).unwrap();
            assert_eq!(got, want(&format!("{case}.txt")), "case {case}");
        }
    }

    #[test]
    fn committed_pair_names_the_first_divergent_layer() {
        let got = compare_files(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("bench/results/golden_intermediates_31b.json"),
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("bench/results/native_intermediates_31b.json"),
        )
        .unwrap();
        assert!(
            got.contains("FIRST DIVERGENCE: target_hidden[layer 35]"),
            "{got}"
        );
    }

    #[test]
    fn python_number_text() {
        for (f, s) in [
            (0.0, "0.0"),
            (1.0, "1.0"),
            (1e20, "1e+20"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (1.3428900241851807, "1.3428900241851807"),
            (-7.168, "-7.168"),
            (2.5e-7, "2.5e-07"),
        ] {
            assert_eq!(py_float_repr(f), s);
        }
        let v = Json::parse(r#"{"a": [1, 2.0, -0, true, null, "x'"], "a": 3}"#).unwrap();
        assert_eq!(py_str(v.get("a").unwrap()), "3");
        let Json::Object(kv) = &v else { unreachable!() };
        assert_eq!(py_str(&kv[0].1), r#"[1, 2.0, 0, True, None, "x'"]"#);
    }

    #[test]
    fn malformed_inputs_are_errors_not_verdicts() {
        for bad in [
            "",
            "{",
            "[1,]",
            "01",
            "1.",
            "{\"a\" 1}",
            "\"\\x\"",
            "1 2",
            "{\"a\":1}x",
        ] {
            assert!(Json::parse(bad).is_err(), "{bad:?} parsed");
        }
        let g = results("golden_intermediates_31b.json");
        let no_layers = Json::parse("{}").unwrap();
        assert!(
            compare(&no_layers, &g).is_err(),
            "golden without target_hidden_per_layer"
        );
        let mut s = g.clone();
        layer(&mut s, "target_hidden_per_layer", 0).set("sum", Json::Str("x".into()));
        assert!(
            compare(&g, &s).is_err(),
            "a string checksum is a malformed input"
        );
        assert!(py_eq(&Json::Int(1), &Json::Float(1.0)));
        assert!(py_eq(&Json::Bool(true), &Json::Int(1)));
        assert!(
            close(f64::NAN, 1.0, 2e-2, 1e-2),
            "NaN never compares greater, as in Python"
        );
        assert!(opt_eq(None, Some(&Json::Null)));
    }
}
