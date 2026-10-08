//! Tokenizer + chat template hook.
//!
//! With feature `tokenizer` (default): loads HuggingFace `tokenizer.json` via the
//! `tokenizers` crate. Without it, or when files are missing, [`GemmaTokenizer`]
//! remains a clear stub that still records the chat-template path for Phase 3+.

use crate::error::{Error, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Paths resolved from an HF model directory.
#[derive(Clone, Debug, Default)]
pub struct TokenizerPaths {
    pub tokenizer_json: Option<PathBuf>,
    pub tokenizer_config_json: Option<PathBuf>,
    /// Jinja chat template (`chat_template.jinja`) or embedded in tokenizer_config.
    pub chat_template: Option<PathBuf>,
    pub chat_template_source: ChatTemplateSource,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ChatTemplateSource {
    #[default]
    Missing,
    JinjaFile,
    TokenizerConfigField,
}

impl TokenizerPaths {
    pub fn discover(dir: impl AsRef<Path>) -> Self {
        let dir = dir.as_ref();
        let tokenizer_json = exists(dir.join("tokenizer.json"));
        let tokenizer_config_json = exists(dir.join("tokenizer_config.json"));
        let jinja = exists(dir.join("chat_template.jinja"));
        let (chat_template, chat_template_source) = if jinja.is_some() {
            (jinja, ChatTemplateSource::JinjaFile)
        } else if tokenizer_config_json.is_some() {
            // Template often lives as `chat_template` string inside tokenizer_config.json.
            (
                tokenizer_config_json.clone(),
                ChatTemplateSource::TokenizerConfigField,
            )
        } else {
            (None, ChatTemplateSource::Missing)
        };
        Self {
            tokenizer_json,
            tokenizer_config_json,
            chat_template,
            chat_template_source,
        }
    }
}

fn exists(p: PathBuf) -> Option<PathBuf> {
    if p.is_file() {
        Some(p)
    } else {
        None
    }
}

/// Encode/decode surface for Gemma 4 IT prompts.
pub struct GemmaTokenizer {
    pub paths: TokenizerPaths,
    /// Raw chat template text when discovered (Jinja or config field).
    pub chat_template: Option<String>,
    #[cfg(feature = "tokenizer")]
    inner: Option<tokenizers::Tokenizer>,
    #[cfg(not(feature = "tokenizer"))]
    _no_tokenizer: (),
}

impl GemmaTokenizer {
    /// Stub that only records paths — no encode until files + feature present.
    pub fn stub(paths: TokenizerPaths) -> Self {
        let chat_template = load_chat_template(&paths);
        Self {
            paths,
            chat_template,
            #[cfg(feature = "tokenizer")]
            inner: None,
            #[cfg(not(feature = "tokenizer"))]
            _no_tokenizer: (),
        }
    }

    pub fn from_hf_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let paths = TokenizerPaths::discover(dir.as_ref());
        let chat_template = load_chat_template(&paths);

        #[cfg(feature = "tokenizer")]
        {
            let inner = if let Some(ref tj) = paths.tokenizer_json {
                let tok = tokenizers::Tokenizer::from_file(tj).map_err(|e| {
                    Error::Tokenizer(format!("load {}: {e}", tj.display()))
                })?;
                Some(tok)
            } else {
                None
            };
            Ok(Self {
                paths,
                chat_template,
                inner,
            })
        }

        #[cfg(not(feature = "tokenizer"))]
        {
            let _ = dir;
            Ok(Self {
                paths,
                chat_template,
                _no_tokenizer: (),
            })
        }
    }

    pub fn is_ready(&self) -> bool {
        #[cfg(feature = "tokenizer")]
        {
            self.inner.is_some()
        }
        #[cfg(not(feature = "tokenizer"))]
        {
            false
        }
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        #[cfg(feature = "tokenizer")]
        {
            let tok = self
                .inner
                .as_ref()
                .ok_or_else(|| Error::Tokenizer(
                    "tokenizer.json not loaded — place HF tokenizer.json in model dir \
                     (chat template path is still available via `.chat_template` / `.paths`)"
                        .into(),
                ))?;
            let enc = tok
                .encode(text, add_special_tokens)
                .map_err(|e| Error::Tokenizer(e.to_string()))?;
            Ok(enc.get_ids().to_vec())
        }
        #[cfg(not(feature = "tokenizer"))]
        {
            let _ = (text, add_special_tokens);
            Err(Error::Tokenizer(
                "built without `tokenizer` feature — enable feature or use stub paths only"
                    .into(),
            ))
        }
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        #[cfg(feature = "tokenizer")]
        {
            let tok = self
                .inner
                .as_ref()
                .ok_or_else(|| Error::Tokenizer("tokenizer not loaded".into()))?;
            tok.decode(ids, skip_special_tokens)
                .map_err(|e| Error::Tokenizer(e.to_string()))
        }
        #[cfg(not(feature = "tokenizer"))]
        {
            let _ = (ids, skip_special_tokens);
            Err(Error::Tokenizer(
                "built without `tokenizer` feature".into(),
            ))
        }
    }

    /// Apply a minimal Gemma-4-IT-like turn wrap when no Jinja engine is wired yet.
    /// Full Jinja lands with Phase 3 parity harness — this only documents the path.
    pub fn format_user_turn_stub(&self, user: &str) -> String {
        if let Some(ref tmpl) = self.chat_template {
            // Do not pretend to evaluate Jinja; surface that the template exists.
            let _ = tmpl;
        }
        format!("<|turn|>user\n{user}<turn|>\n<|turn|>assistant\n")
    }
}

fn load_chat_template(paths: &TokenizerPaths) -> Option<String> {
    match paths.chat_template_source {
        ChatTemplateSource::JinjaFile => paths
            .chat_template
            .as_ref()
            .and_then(|p| fs::read_to_string(p).ok()),
        ChatTemplateSource::TokenizerConfigField => {
            let p = paths.tokenizer_config_json.as_ref()?;
            let s = fs::read_to_string(p).ok()?;
            let v: serde_json::Value = serde_json::from_str(&s).ok()?;
            v.get("chat_template")
                .and_then(|x| x.as_str())
                .map(|s| s.to_string())
        }
        ChatTemplateSource::Missing => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discover_chat_template_jinja() {
        let dir = tempfile::tempdir().unwrap();
        let jinja = dir.path().join("chat_template.jinja");
        fs::write(&jinja, "{{ messages }}").unwrap();
        let paths = TokenizerPaths::discover(dir.path());
        assert_eq!(paths.chat_template_source, ChatTemplateSource::JinjaFile);
        let tok = GemmaTokenizer::stub(paths);
        assert!(tok.chat_template.as_ref().unwrap().contains("messages"));
        assert!(!tok.is_ready());
    }

    #[test]
    fn stub_format_user_turn() {
        let tok = GemmaTokenizer::stub(TokenizerPaths::default());
        let s = tok.format_user_turn_stub("hi");
        assert!(s.contains("user"));
        assert!(s.contains("hi"));
    }
}
