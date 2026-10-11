# Changelog

## Unreleased

## 0.2.0

### Added

- **Paired-end alignment** (#1). `-1 R1 -2 R2`, `--interleaved`, or a BAM of
  0x1/0x40/0x80 mates align read pairs as pairs: the reference maximising the
  two mates' summed score wins (both mates at or above `--min-score`), under an
  orientation rule and `--max-insert` (default 500). A pair no reference
  satisfies is written without 0x2, each mate at its own best reference on
  either strand. Standard paired SAM: flags, `RNEXT`/`PNEXT`/`TLEN`, `MC`,
  per-mate `NM`/`MD`/`AS`/`XS`/`XA`, MAPQ 60 only for a unique best reference,
  mates adjacent in input order, `--sort coordinate` and `--secondary` supported.
  See [docs/align.md](docs/align.md#paired-end).
- `--pair-orientation fr|rf|ff` (default `fr`): how the mates must face each
  other in a proper pair. `MQ` (the mate's MAPQ) is written beside `MC`, and the
  run summary counts discordant pairs (mates whose best references differ,
  each reported at its own) (#4).
- 3' adapter read-through is detected before alignment (TruSeq Read 1/2
  prefixes; `--adapter-r1`, `--adapter-r2`, `--min-adapter-overlap`,
  `--no-adapter-trim`), the adapter is kept as soft clip with an `ad:i` tag, and
  the rest of each mate is aligned end to end so terminal mismatches stay in
  `MD`/`NM`. Paired runs therefore default to `--mode semiglobal` (an explicit
  `--mode local` is honoured and warned about).
- Library: `eschalign::adapter`, `Aligner::map_pairs`/`map_pairs_scored`,
  `PairOptions`, `PairMapping`, `PairHit`.
- Tests: `align_paired.rs` over a synthesized pair fixture in
  `tests/fixtures/pairs/`; `align_paired::single_end_output_unchanged` pins the unpaired output to a
  golden record-stream digest.

### Changed

- A BAM whose first record is flagged paired (0x1 with 0x40/0x80) is now a
  paired run; it used to be read as unpaired reads with the flags discarded.
  Every other input, and every existing flag, behaves as before (output
  byte-identical, `single_end_output_unchanged`).

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
