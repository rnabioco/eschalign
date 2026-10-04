# Changelog

## 0.1.0

First release of eschalign as a standalone tool.

eschalign was carved out of [escapepod-rs](https://github.com/rnabioco/escapepod-rs)
0.31.0, where it was `escpod align` (library crate `escapepod-align`, command
`escapepod-cli/src/commands/align.rs`). The git history of those files is
preserved in this repository. The version restarts at 0.1.0; this repository is
not published to crates.io.

Changes from `escpod align` 0.31.0:

- The binary is `eschalign`, with no `align` subcommand:
  `eschalign reads.ubam -r ref.fa -o out.bam`. All flags and behaviour are unchanged.
- The `@PG` record is `ID:eschalign` / `PN:eschalign` (was `escpod-align` / `escpod`).
- `ESCHALIGN_BACKEND` and `ESCHALIGN_ROW_MAJOR_ONLY` are the documented
  environment switches. `ESCAPEPOD_ALIGN_BACKEND` and
  `ESCAPEPOD_ALIGN_ROW_MAJOR_ONLY` still work as deprecated aliases; the
  `ESCHALIGN_*` name wins when both are set.
- No dependency on any escapepod-rs crate: the flexible UUID parser, the
  `ESCHALIGN_*` switch parser, the `--device` placement, progress/style helpers
  and the tracing setup were copied or reduced into `eschalign-cli`.
- `eschalign::cuda::device_visible()` replaces the device probe the CLI used to
  borrow from `escapepod-demux`.
- Removed the `align_then_classify` test (it ran `escpod classify` over a pod5).
- Own pixi environments (`default`, `dev`, `gpu`, `dev-gpu`), CI (fmt, clippy,
  nextest, doctests) and docs.
