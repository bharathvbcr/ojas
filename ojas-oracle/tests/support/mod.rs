//! Builds safetensors bytes for negative tests.
#![allow(dead_code)]

/// A tensor to write: name, dtype tag, shape, little-endian bytes.
pub struct Raw {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

pub fn f32_raw(name: &str, shape: &[usize], values: &[f32]) -> Raw {
    Raw {
        name: name.to_string(),
        dtype: "F32",
        shape: shape.to_vec(),
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// JSON string literal with the escapes the reader must undo.
pub fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Header from tensors in order (offsets tile the buffer) plus metadata.
pub fn build(tensors: &[Raw], metadata: &[(&str, &str)]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut parts = Vec::new();
    if !metadata.is_empty() {
        let kv: Vec<String> = metadata
            .iter()
            .map(|(k, v)| format!("{}:{}", json_str(k), json_str(v)))
            .collect();
        parts.push(format!("\"__metadata__\":{{{}}}", kv.join(",")));
    }
    let mut offset = 0usize;
    for t in tensors {
        parts.push(format!(
            "{}:{{\"dtype\":\"{}\",\"shape\":{:?},\"data_offsets\":[{},{}]}}",
            json_str(&t.name),
            t.dtype,
            t.shape,
            offset,
            offset + t.bytes.len()
        ));
        offset += t.bytes.len();
    }
    header.push_str(&parts.join(","));
    header.push('}');
    build_with_header(
        &header,
        &tensors
            .iter()
            .flat_map(|t| t.bytes.clone())
            .collect::<Vec<u8>>(),
    )
}

pub fn build_with_header(header: &str, data: &[u8]) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}
