# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository. The user-level
`~/.claude/CLAUDE.md` rules (never copy pod5/bam/npz bulk data into a
worktree, write tabular results compressed, no scratch output in `~/.cache`,
`cargo clean` before removing a worktree) apply here too.

## Project Overview

eschalign is an all-vs-all read-to-reference aligner for small reference panels
(tRNA): every read is scored against every reference, ties are reported as
ties, and `MD`/`NM` follow `samtools calmd`. It was carved out of
[escapepod-rs](https://github.com/rnabioco/escapepod-rs) 0.31.0 (`escpod align`)
with its git history preserved, and **depends on nothing from escapepod-rs**.
Two crates:

- `crates/eschalign` — the library (formerly `escapepod-align`): std only, optional `gpu` feature (cudarc). Layer map and invariants in `crates/eschalign/CLAUDE.md`.
- `crates/eschalign-cli` — the `eschalign` binary (formerly `escapepod-cli`'s `commands/align.rs`): all BAM/FASTA/FASTQ I/O, the pipeline, `--device` placement, logging. One flat binary: `eschalign reads.ubam -r ref.fa -o out.bam` (no subcommand).

## Requirements

- Rust 1.95 or later (`workspace.package.rust-version`, pinned by `rust-toolchain.toml`)
- pixi for the toolchain extras (mold, samtools, cargo-nextest, CUDA runtime)

## Build Commands

```bash
cargo build --release                       # the eschalign binary: target/release/eschalign
cargo build --release --features gpu -p eschalign-cli   # GPU scoring (opt-in)

cargo nextest run --workspace               # unit + integration tests
cargo test --doc --workspace                # nextest does not run doctests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check

# NOTHING CUDA is needed at build time — cudarc dlopens — so a `gpu` build is a
# plain cargo build and needs no GPU env. Create the runtime environment ONCE,
# on a networked node:
pixi run install-gpu        # CONDA_OVERRIDE_CUDA=12 pixi install -e gpu

# GPU tests on a GPU node (SLURM account `gpu_rbi`, partition `gpu`). Note
# `-e dev-gpu`, not `-e gpu`: cargo-nextest lives in the `dev` feature.
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 \
    pixi run -e dev-gpu cargo nextest run --features gpu -p eschalign --test gpu_parity
```

`--device auto` (default) scores on a visible GPU in a `gpu` build, else the
CPU; `--device gpu` demands it and errors instead of falling back; `--device
cpu` forces the CPU. Output is byte-identical either way.

mold is the linker full time: the base `[activation]` in `pixi.toml` exports
`LD_PRELOAD=$CONDA_PREFIX/lib/mold/mold-wrapper.so` for EVERY environment, so
any `pixi run [-e <env>] cargo …` links with mold — no `mold -run`, no
`-fuse-ld=mold` (the system gcc 11.5 has none). Pixi tasks in the `dev` env:
`build`, `build-rel`, `test`, `test-doc`, `check`, `clippy`.

### Build baseline and SLURM builds

Local Linux/x86_64 builds pin `-C target-cpu=x86-64-v3` via `.cargo/config.toml`
(AVX2 + FMA + BMI2 + POPCNT + F16C). This is portable across every node in
the cluster (Broadwell login, Cascade Lake rna, Ice Lake gpu). Do **not**
use `target-cpu=native`: a binary built on a gpu node uses Ice Lake-only
instructions (VBMI, VPCLMULQDQ, …) that SIGILL on rna. The AVX-512 kernels
are `#[target_feature]` + runtime `is_x86_feature_detected!` dispatch, never a
global baseline bump. An environment `RUSTFLAGS` replaces the config file's
flags wholesale rather than adding to them.

The login node has only 2 cores — wrap any heavy build, test or bench in `srun`
and run it in the foreground:

```bash
srun -p rna -c 32 --mem=32G -J eschalign-build --comment=eschalign-build -- \
    pixi run -e dev cargo nextest run --workspace
# throughput-sensitive runs — ask for a full socket (48 logical = 24 physical + HT).
# `-c 32` only allocates 16 physical cores on rna's Gold 6240R; `-c 48` fills the
# socket without crossing NUMA. `-c 64`+ crosses sockets and is a crapshoot on a
# shared node.
srun -p rna -c 48 --mem=64G pixi run ./target/release/eschalign …
```

The SIMD equality tests sweep `Backend::available()`; run them on `rna`
(Cascade Lake, AVX-512) or the AVX-512 kernels never execute.

### Running on Alpine (CU Boulder RMACC/ACCESS)

The `rna`/`gpu` partitions above are the **Beevol** (CU Anschutz) cluster. On
**Alpine** every partition needs an **explicit `--qos=` matched to the
partition** plus `-A amc-general` — the partition name alone is not enough.

| Purpose | Beevol | Alpine |
|---|---|---|
| CPU build / test / bench | `srun -p rna -c 32 --mem=32G` | `srun -p amilan --qos=normal -A amc-general -c 32 --mem=32G` |
| CPU full node (throughput) | `srun -p rna -c 48` | `srun -p amilan --qos=normal -A amc-general -c 64 --mem=120G` |
| GPU test / bench | `srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1` | `srun -p aa100 --qos=gpu-normal -A amc-general -c 16 --gres=gpu:1` |
| Light build (login is 2 cores) | login node | `srun -p acompile --qos=compile -A amc-general -c 16 --mem=32G` |

QOS↔partition: amilan→`normal`|`long`; aa100/al40→`gpu-normal`|`gpu-long`;
acompile→`compile`; atesting→`testing` (1 h cap). Default mem is 3840 MB/core.

- `amilan` nodes are AMD EPYC Milan, 2×32 = **64 physical cores, no hyperthreading**, 245 GB — the Beevol "`-c 48` fills a socket" tuning does not apply.
- GPU CUDA work must use `aa100` (A100) or `al40` (L40). `ami100` is AMD/ROCm and **cannot** run the CUDA `gpu` feature.
- amilan/aa100/acompile are all Zen3 x86-64, so the pinned `x86-64-v3` baseline is portable across them — no SIGILL risk. (AVX-512 kernels simply do not run on Zen3; the AVX2 ones do.)
- Toolchain lives in pixi, not on the bare PATH: e.g. `srun -p amilan --qos=normal -A amc-general -c 32 --mem=32G pixi run -e dev test`.

## Benchmarking

`benchmarks/benchmark_align.sh` is the end-to-end harness (header documents the
flags): arms (binary × backend × device × sort order) are **interleaved, not
grouped**, the input is paged in once before the first arm, and CPU is reported
beside wall — the lessons of benchmarking on BeeGFS, where the first arm to
touch a file pays to page it in. Output goes to node-local scratch and is
deleted. The numbers carried over from escapepod-rs are in `benchmarks/README.md`.
`crates/eschalign/examples/{long_read_probe,gpu_score_probe}.rs` are the
kernel-level probes; keep them re-runnable. Benchmark output belongs under
`results/` or `~/scratch/<topic>/`, never `~/.cache`.

```bash
srun -p rna -c 32 --mem=32G -- benchmarks/benchmark_align.sh \
    --reads sample.ubam --reference trna.fa --md5 --reps 3
```

### Build profiles

`release` (fat LTO, `codegen-units=1`, stripped, `panic=abort`), `dist`
(inherits release), `release-with-debug`, `profiling` (LTO off for readable
call graphs — use for perf/samply), `bench`. `dev.package."*"` and
`dev.package.eschalign` compile at `opt-level = 2`: the scalar-oracle equality
tests over 20–60 kb reads cost ~80 ns a cell at opt-level 0 and run for minutes.

### Allocator

The binary uses mimalloc (`eschalign-cli`'s `mimalloc` feature, on by default).
The reason is the allocation profile of this pipeline's parent: on escapepod-rs's
`escpod classify` with 8 threads a static-musl build whose old musl took one
global lock per allocation made 9.2 M voluntary context switches against 79 k
for glibc and ran 2.35× slower in wall. mimalloc keeps Rust allocations off
that path whichever libc is linked, and is worth ~10% of CPU on glibc.
`cargo build --no-default-features` is the same binary on the system allocator,
kept buildable so the two can be measured against each other.

### Runtime verbosity

All status/progress/warning output flows through `tracing` to stderr (the
`EschalignFormatter` in `main.rs`: `timestamp LEVEL [target] message`). Only the
BAM under `-o -` goes to stdout. Default level is **info** for `eschalign` and
`eschalign_cli`; dependencies are pinned at **warn**. `-v`/`-vv` raise only our
level; `-q` is errors only; `RUST_LOG` is the escape hatch and always wins.
**A new workspace crate must be added to `filter_directive` in `main.rs`**, or its
`info!` lines vanish with nothing to say so. Progress bars (`progress.rs`) hide
under `-q`. Use `tracing::info!/warn!/error!` — do not hand-prefix `Warning:`.

### Environment switches

`ESCHALIGN_BACKEND` (`scalar|avx2|avx512`, caps the SIMD dispatch — read in the
library) and `ESCHALIGN_ROW_MAJOR_ONLY` (boolean, A/B lever). The old escapepod-rs
names `ESCAPEPOD_ALIGN_BACKEND` / `ESCAPEPOD_ALIGN_ROW_MAJOR_ONLY` are
**deprecated aliases**; the `ESCHALIGN_*` name wins whenever it is set.

A boolean `ESCHALIGN_*` switch goes through `eschalign-cli/src/env.rs`
(`flag`, `flag_with_alias`), not a hand-rolled parse: on for `1|true|yes|on`,
off for unset/empty/`0|false|no|off` (case-insensitive), anything else warns
once and counts as off. `=0` turns a switch off — a raw `var_os(..).is_some()`
would treat it as on. `clippy.toml`'s `disallowed-methods` enforces this (and
`MultiGzDecoder` over `GzDecoder`, and sized `MultithreadedReader`) under
`-D warnings`; a genuinely different shape (a backend-name cap, `NO_COLOR`,
`RUST_LOG`) gets a narrow `#[allow(clippy::disallowed_methods)]` with a reason.

### Use this, not your own

| Need | Use |
|---|---|
| Reference FASTA (gzip, duplicate names refused) | `eschalign::fasta::read_fasta` |
| Boolean `ESCHALIGN_*` switch | `env::flag` / `env::flag_with_alias` (`eschalign-cli/src/env.rs`) |
| Gzip input, possibly multi-member/bgzipped | `flate2::read::MultiGzDecoder`, never `GzDecoder` |
| BGZF reader worker count | `noodles_bgzf::io::MultithreadedReader::with_worker_count`, sized from the rayon pool — `::new` is one worker |
| `@PG` `CL` on a written BAM | the real argv, as `align.rs` does — not a hand-rebuilt command line |
| Skipping a GPU test without a device | return early with a message naming the missing device, don't `panic`/`unwrap` |
| Read-name matching (`--read-ids`) | `uuid_util::parse_uuid_flexible` (dashed or 32-hex) |

## Architecture

### Workspace layout

- `crates/eschalign/` — alphabet / `Scoring` / `Mode` / scalar oracle / `simd` (two kernel families) / `sam` (CIGAR, `MD`/`NM`) / `panel` / `mapper` (`Aligner`) / `cuda` (`gpu`) / `fasta` / `error`. See its `CLAUDE.md` for the map, tests and traps.
- `crates/eschalign-cli/src/` — `main.rs` (clap, tracing, rayon pool, allocator), `align.rs` (the pipeline), `device.rs` (`--device`), `env.rs`, `threads.rs`, `progress.rs`, `style.rs`, `uuid_util.rs`.
- `crates/eschalign-cli/tests/` — `align_e2e.rs`, `align_sort.rs`, `align_unpaired.rs` over `tests/fixtures/` (60 real tRNA reads as a uBAM-shaped BAM plus the 47-reference panel and an ambiguity-code variant; **no pod5** — these tests never read signal), and `align_paired.rs` over `tests/fixtures/pairs/` (a synthesized 4-reference panel with two paralogs and one hand-built read pair per behaviour; `gen_pair_fixture.py` regenerates it). `align_unpaired.rs` pins the unpaired output to a golden digest of the uncompressed record stream (`fixtures/unpaired_golden.tsv`, `ESCHALIGN_BLESS=1` rewrites it) — a change to it is a change to unpaired behaviour and must be intended.
- `crates/eschalign/tests/fixtures/` — the parasail and calmd goldens and the scripts that regenerate them (`gen_align_golden.py` needs `parasail`; `gen_calmd_golden.py` needs samtools).

### The command

`eschalign` — all-against-all alignment of reads to a small reference panel (tRNA) — `eschalign reads.ubam -r ref.fa -o out.bam`. Takes dorado's uBAM (or forward-strand aligned BAM, or FASTA/FASTQ, plain or gzip — sniffed from content), writes records in **input order** (`SO:unsorted`) — or, with `--sort coordinate` (#417), in exactly `samtools sort`'s order via a per-reference bucket sort in the writer thread (input order restored first, as the tie-break; past `--sort-memory` the largest buckets spill to one unlinked temp file, and each reference is ordered alone at the end, so there is no merge) — with **every input tag copied byte for byte** and `NM`/`MD`/`AS`/`XS`/`XA` owned by the aligner. Primary = lowest-indexed reference at the best score; the rest of the tie set goes to `XA` (bwa's `ref,±pos,CIGAR,NM;`) and, with `--secondary`, to 0x100 records carrying `SEQ`/`QUAL` as `*` and only the alignment tags + `RG`; MAPQ is 60 unique / 0 tied, nothing in between. Reads below `--min-score` are written unmapped, never dropped, and so are reads longer than `--max-read-len` (default 1000 nt, `0` = no limit; filtered in the CLI before any scoring, GPU included). Pipeline: reader thread (raw `bam::Record`s) → bounded channel → dispatcher cutting chunks of ≤ 64 reads / ≤ 16 kb and `spawn`ing each onto rayon (decode, score, trace the chunk's winners together, rewrite, **encode** — through a `bam::io::Writer` over a `Vec<u8>`, since `noodles_bam`'s record encoder is crate-private) → writer thread reordering numbered chunks into BGZF, returning one permit per chunk written (two batches in flight at most). The decode and encode live in the workers because a uBAM record is ~5 kB of `mv`/`MM`/`ML` for a ~130 nt read. **There is no barrier between batches, on evidence:** real tRNA samples carry reads up to ~400 kb, a chunk holding one costs seconds against milliseconds, and a `par_iter` per batch left 16 of 32 threads idle behind those stragglers (30.0 → 18.8 s on a 490 k-read sample, output identical); the per-batch pattern must not come back. `-v` logs the workers' busy time against wall × threads. **`--device` places `Stage::Align`** (#401): on the GPU, a dedicated thread sits between the reader and the dispatcher, takes whole batches off the reader's bounded channel, scores each in one `GpuScorer::score_batch` launch (forward reads, then reverse complements under `--strand both`), and hands the batch on with its `ScoreMatrix` behind an `Arc`; each chunk carries the `Arc` plus its offset, and its worker calls `Aligner::map_reads_scored` instead of `map_reads` — tie selection, pairs-kernel tracebacks, `MD`/`NM` and encoding are the CPU path's own code, so the output is byte-identical (`align_e2e::device_gpu_output_is_byte_identical`; also `samtools view | md5sum`-equal on a 490 k-read sample). A read the kernel does not take comes back unscored and its worker scores it on the CPU. The channel after the GPU thread holds two batches, so memory stays O(`--batch-size`). The scorer holds two kernels and splits each batch between them per read (#9): a *half2* kernel (two references per thread in `f16x2`, the FP16 pipe; sm_80+, panel references ≤ 1,536 nt, read inside `cuda::fits_f16` — every tRNA read is) at ~2× the *int* kernel's cells/s, which takes the rest; same scores, so the output is unchanged (`gpu_parity` checks both). The GPU thread packs batch *k + 1* while batch *k*'s kernel runs (`GpuScorer::{prepare, submit, finish}`), so the device sits at ~100% whenever it has work; what leaves it idle is the CPU side stalling, and the very long reads (50–400 kb) are what stalls it (dropping the 52 reads over 20 kb takes M1 from 14.3 to 7.5 s): the GPU leaves them to the CPU, where the phase-1 score kernel is, as far as measured, **memory-bound on a read that long** (its per-row `H`/`E` state is 50 MB for a 395 kb read, streamed once per reference column: 4.9 s to score that read on one rna core, 2e9 cells/s), and the chunk holding it blocks the ordered writer while the permit window fills. Splitting such a read's scoring over the lane groups on six threads did not help *with that kernel* (six 50 MB streams at once), so #415 fixed the layout first: reads of ≥ 256 nt (`simd::TRANSPOSED_MIN_READ_LEN`) take a reference-major score kernel whose state is one panel row — 2.3× on the 395 kb read on one core — and a read of ≥ 16 kb (`SPLIT_MIN_BASES`) has its lane groups scored across the pool by `score_across_pool`, its row fed through the same `prescored` hook as a GPU row (0.29 s wall for that read; `ESCHALIGN_ROW_MAJOR_ONLY=1` is the A/B lever, output byte-identical). `--scoring` defaults to gpu-tRNA-mapper's `2,-1,-10,-1`; bwa `-x ont2d` is `1,-1,-2,-1`, and even then 27 of the 60 classify-fixture reads get a CIGAR different from bwa's (`tests/align_e2e.rs` prints the count; charging calls match the golden bit for bit on every read whose alignment is unchanged).

### Paired-end (#1)

`-1/-2`, `--interleaved`, or a BAM whose first primary record has 0x1 with 0x40/0x80 (`bam_is_paired` opens it once to peek) selects `Input::Paired` and a `PairCfg`; **nothing else changes for unpaired input** (`align_unpaired::unpaired_output_unchanged`). Where the pieces live:

- **Library** (`eschalign`, still std only): `adapter.rs` (`find`, `detect_pair`: leftmost prefix match, `overlap/10` mismatches, the two mates' insert ends must agree or the weaker detection is dropped) and `mapper/pair.rs` (`Aligner::map_pairs_scored`, `PairOptions`, `PairMapping::{Proper, Separate}`). A *candidate* is `(reference, template strand)`; both mates must clear `min_score`; candidates are traced **tier by tier in descending summed score, every open pair's next tier through one `align_pairs` call** (so the traceback lanes still fill), and the first tier with a candidate that passes the constraints on the *traced* coordinates (no outie, insert ≤ `max_insert`, both mates align a base) is the tie set. Constraints need coordinates, so a score alone can never decide a pair. No qualifying candidate → each mate goes through `map_reads_scored` with `both_strands: true`.
- **CLI** (`align.rs`): `PairSource` (`Bam` / `Interleaved` / `Files`) → `read_pair_batches` (whole pairs per batch, mate 1 then mate 2 adjacent, names stripped of `/1` `/2`, orphan/mismatch = error naming the read) → `cut_chunks(batch, paired)` (**a cut never falls between mates — the pair-in-one-chunk invariant; do not add a barrier either**) → `process_pair_chunk` → `build_pair_records` (flags, `RNEXT`/`PNEXT`/`TLEN`, `MC`, `ad`, per-mate `XA` entry *k* = tied pair *k*, secondaries per pair). The adapter bases are *not* aligned: the aligned mate is `seq[..len - ad]`, and `clipped_span` shifts a reverse hit's coordinates by `ad` so the soft clip lands at the right end.
- **GPU**: the same thread, fed flattened reads (`2i` = mate 1, `2i+1` = mate 2) with `both_strands = true` always (the fallback needs every orientation). It must trim exactly what the workers trim — both call `pair_adapters`; if they ever diverge the rows are for different queries and nothing checks it but `pair_gpu_matches_cpu`.
- **Paired default is `--mode semiglobal`**, because `local` clips the terminal mismatches the mode exists to keep; semi-global also gives every mate some positive score, which is why paired runs need `--min-score` (the tests all pass one).


## Testing notes

- `cargo nextest run --workspace` is ~100 tests; the heaviest are the scalar-oracle equality tests (opt-level 2 in dev). `align_sort::sort_coordinate_matches_samtools_sort` uses `samtools` from PATH or `$SAMTOOLS` (in the pixi env; it says it skipped otherwise).
- The GPU tests (`eschalign/tests/gpu_parity.rs`) skip with a message where there is no device; run them on a GPU node.
- Fixtures under `crates/eschalign-cli/tests/fixtures/` are tiny and tracked; never put bulk data (pod5, large BAMs) in the tree — see the user-level rules.
- An equality test must be shown to **fail** on the pre-fix code before a guardrail is trusted (round-trip tests are blind to fields nothing reads back).
