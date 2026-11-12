//! The MSRV named in `src/arch.rs`'s module comment must equal the package's
//! `rust-version`, so a bump of one cannot leave the other stale.

#[test]
fn arch_msrv_comment_matches_rust_version() {
    const PHRASE: &str = "the workspace MSRV is ";
    // `//!` lines wrap, so join them before searching: the phrase may sit on a
    // line break.
    let text: String = include_str!("../src/arch.rs")
        .lines()
        .filter_map(|line| line.strip_prefix("//!"))
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" ");
    let at = text
        .find(PHRASE)
        .unwrap_or_else(|| panic!("src/arch.rs no longer says {PHRASE:?}; restore it"));
    let token: String = text[at + PHRASE.len()..]
        .chars()
        .take_while(|c| !c.is_whitespace())
        .collect();
    let version = token.trim_end_matches(|c: char| !c.is_ascii_digit());
    assert!(
        !version.is_empty(),
        "no version after {PHRASE:?}: {token:?}"
    );
    // Exact string match: `1.97` does not equal `1.97.0`.
    assert_eq!(
        version,
        env!("CARGO_PKG_RUST_VERSION"),
        "src/arch.rs says the workspace MSRV is {version}, rust-version is {}",
        env!("CARGO_PKG_RUST_VERSION")
    );
}
