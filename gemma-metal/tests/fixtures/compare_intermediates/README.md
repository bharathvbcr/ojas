# compare_intermediates oracle transcripts

`a.txt` … `f.txt` are what `bench/compare_intermediates.py` (the Python this
replaced, removed in the same change) printed for six golden/native pairs.
`src/intermediates.rs` must print the same text byte for byte
(`report_matches_the_python_oracle`).

All six start from `bench/results/golden_intermediates_31b.json` (g) and
`bench/results/native_intermediates_31b.json` (n). The test makes the same
edits in Rust:

| case | golden | native |
| :-- | :-- | :-- |
| a | g | n (the committed pair) |
| b | n | g |
| c | g | g (full match) |
| d | g | g with `target_hidden_per_layer` cut to 2, `fc_out_lastrow` removed, `draft_attn_offsets_per_layer` cut to 3, layer 1 `q_rope_offset` = 9, `proposed_block_tokens` = null |
| e | g | g with layer 0 `absmean` = null, `fc_out_lastrow.shape` = [1, 2], `h_ctx_lastrow.first8[3]` + 1.0, `draft_logits_lastpos.sum` = 1e20 |
| f | g | g with `proposed_block_tokens` = [1, 2, 3, 4] |
