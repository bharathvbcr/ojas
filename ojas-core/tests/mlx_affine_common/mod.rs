//! Shared by the affine-quantiser tests: a reader for the golden safetensors files.
//!
//! ojas-core depends only on std, so this reads the safetensors header with a small JSON
//! scanner rather than ojas-io. The header it reads is the one `write_safetensors` in
//! `ojas-oracle/python/qwen35_fakequant_base.py` writes: an object of
//! `{"dtype", "shape", "data_offsets"}` entries.

use std::collections::BTreeMap;
use std::path::PathBuf;

pub const CASES: [&str; 5] = [
    "kinds_w64",
    "mixed_w2048",
    "mixed_w6144",
    "randn_w2048",
    "randn_w6144",
];

pub struct Entry {
    pub shape: Vec<usize>,
    pub words: Vec<u32>,
    pub halves: Vec<u16>,
}

pub struct Golden(BTreeMap<String, Entry>);

impl Golden {
    pub fn bf16(&self, name: &str) -> (&[usize], &[u16]) {
        let e = self.entry(name);
        assert!(e.words.is_empty(), "{name} is not BF16");
        (&e.shape, &e.halves)
    }

    pub fn u32s(&self, name: &str) -> (&[usize], &[u32]) {
        let e = self.entry(name);
        assert!(e.halves.is_empty(), "{name} is not U32");
        (&e.shape, &e.words)
    }

    pub fn has(&self, name: &str) -> bool {
        self.0.contains_key(name)
    }

    fn entry(&self, name: &str) -> &Entry {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("golden has no tensor {name}"))
    }
}

pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mlx_affine")
}

pub fn load(case: &str) -> Golden {
    load_from(&fixture_dir(), case)
}

/// `case` from the golden directory `dir`: the entry point for a test in another crate, which
/// includes this file by `#[path]` and so resolves `fixture_dir` against its own manifest.
pub fn load_from(dir: &std::path::Path, case: &str) -> Golden {
    let path = dir.join(format!("{case}.safetensors"));
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + n]).expect("header is UTF-8");
    let data = &bytes[8 + n..];
    let Json::Object(top) = Parser::new(header).value() else {
        panic!("safetensors header is not an object")
    };
    let mut out = BTreeMap::new();
    for (name, v) in top {
        let Json::Object(fields) = v else {
            panic!("{name}: entry is not an object")
        };
        let field = |k: &str| {
            fields
                .iter()
                .find(|(n, _)| n == k)
                .unwrap_or_else(|| panic!("{name}: no {k}"))
                .1
                .clone()
        };
        let (Json::Str(dtype), Json::Array(shape), Json::Array(offs)) =
            (field("dtype"), field("shape"), field("data_offsets"))
        else {
            panic!("{name}: malformed entry")
        };
        let num = |j: &Json| match j {
            Json::Num(n) => *n,
            _ => panic!("{name}: non-number"),
        };
        let shape: Vec<usize> = shape.iter().map(num).collect();
        let raw = &data[num(&offs[0])..num(&offs[1])];
        let (words, halves) = match dtype.as_str() {
            "U32" => {
                let (words, rest) = raw.as_chunks::<4>();
                assert!(
                    rest.is_empty(),
                    "{name}: U32 data is not a whole number of words"
                );
                (
                    words.iter().map(|c| u32::from_le_bytes(*c)).collect(),
                    Vec::new(),
                )
            }
            "BF16" => {
                let (halves, rest) = raw.as_chunks::<2>();
                assert!(
                    rest.is_empty(),
                    "{name}: BF16 data is not a whole number of halves"
                );
                (
                    Vec::new(),
                    halves.iter().map(|c| u16::from_le_bytes(*c)).collect(),
                )
            }
            other => panic!("{name}: unexpected dtype {other}"),
        };
        out.insert(
            name,
            Entry {
                shape,
                words,
                halves,
            },
        );
    }
    Golden(out)
}

#[derive(Clone)]
enum Json {
    Object(Vec<(String, Json)>),
    Array(Vec<Json>),
    Str(String),
    Num(usize),
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn new(s: &'a str) -> Self {
        Self {
            s: s.as_bytes(),
            i: 0,
        }
    }

    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) {
        self.ws();
        assert_eq!(self.s[self.i], c, "JSON: expected {}", c as char);
        self.i += 1;
    }

    fn string(&mut self) -> String {
        self.eat(b'"');
        let start = self.i;
        while self.s[self.i] != b'"' {
            assert_ne!(self.s[self.i], b'\\', "JSON: escapes are not expected");
            self.i += 1;
        }
        let text = String::from_utf8(self.s[start..self.i].to_vec()).unwrap();
        self.i += 1;
        text
    }

    fn value(&mut self) -> Json {
        self.ws();
        match self.s[self.i] {
            b'{' => {
                self.i += 1;
                let mut fields = Vec::new();
                loop {
                    self.ws();
                    if self.s[self.i] == b'}' {
                        self.i += 1;
                        return Json::Object(fields);
                    }
                    if !fields.is_empty() {
                        self.eat(b',');
                    }
                    let key = self.string();
                    self.eat(b':');
                    fields.push((key, self.value()));
                }
            }
            b'[' => {
                self.i += 1;
                let mut items = Vec::new();
                loop {
                    self.ws();
                    if self.s[self.i] == b']' {
                        self.i += 1;
                        return Json::Array(items);
                    }
                    if !items.is_empty() {
                        self.eat(b',');
                    }
                    items.push(self.value());
                }
            }
            b'"' => Json::Str(self.string()),
            c if c.is_ascii_digit() => {
                let start = self.i;
                while self.s[self.i].is_ascii_digit() {
                    self.i += 1;
                }
                Json::Num(
                    std::str::from_utf8(&self.s[start..self.i])
                        .unwrap()
                        .parse()
                        .unwrap(),
                )
            }
            c => panic!("JSON: unexpected {}", c as char),
        }
    }
}
