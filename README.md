# eschalign

All-vs-all alignment of reads to a small reference panel, built for tRNA
(~150–200 references of ~150 nt). Every read is scored against every reference
with an affine-gap DP on AVX2/AVX-512 inter-sequence SIMD kernels (or a CUDA
kernel); the best reference wins, **references that tie are reported as ties**
rather than hidden behind a MAPQ of 0, and `MD`/`NM` are what `samtools calmd`
writes. It replaces `bwa mem` + `samtools sort` + `samtools calmd` for
direct-RNA tRNA reads, keeping every input tag (`mv`, `MM`/`ML`, `RG`, …)
byte for byte.

There is no seed index: memory is one batch of reads and runtime is
`reads × references × lengths`, the same for every sample of the same size. The
intended range is ≤ ~10 k references of ≤ ~1 kb. A 3.3 M-read dorado uBAM
against a 164-reference panel aligns in about two minutes on 16 cores.

eschalign was carved out of [escapepod-rs](https://github.com/rnabioco/escapepod-rs)
0.31.0, where it was `escpod align`; the history of the code is preserved.

## Install / build

Rust 1.95+. From a clone:

```bash
cargo build --release                                   # target/release/eschalign
cargo build --release --features gpu -p eschalign-cli   # with the CUDA score kernel
```

On a cluster, use the pixi environments (mold linker, samtools, cargo-nextest,
CUDA runtime) and do not build on a login node:

```bash
srun -p rna -c 32 --mem=32G pixi run -e dev build-rel
```

Local builds pin `-C target-cpu=x86-64-v3` (`.cargo/config.toml`); the AVX-512
kernels are chosen at run time. Do not use `target-cpu=native`.

## Usage

```bash
eschalign reads.ubam -r trna.fa -o aligned.bam
```

`READS` is dorado's unaligned BAM, or FASTA/FASTQ, plain or gzip (sniffed from
the content). Common options:

| Option | Meaning |
|---|---|
| `-r, --reference <FASTA>` | reference panel (required) |
| `-o, --output <BAM>` | output BAM, `-` for stdout (required) |
| `--sort <unsorted\|coordinate>` | input order (default) or exactly `samtools sort` order |
| `--sort-memory`, `--tmp-dir` | spill control for `--sort coordinate` |
| `--mode <local\|semiglobal>` | Smith–Waterman (default) or overlap alignment |
| `--scoring <M,X,O,E>` | match, mismatch, gap open, gap extend (default `2,-1,-10,-1`; bwa `-x ont2d` is `1,-1,-2,-1`) |
| `--min-score <N>` | below this a read is written unmapped |
| `--max-read-len <N>` | longer reads are written unmapped, not aligned (default 1000; 0 = no limit) |
| `--max-ties <N>`, `--secondary` | how ties are reported (`XA`, 0x100 records) |
| `--strand <forward\|both>` | also score the reverse complement |
| `--read-ids <FILE>` | align only the named reads |
| `-t, --threads <N>`, `--batch-size <N>` | parallelism and memory |
| `--device <auto\|cpu\|gpu>` | where panel scoring runs (see below) |
| `-1`, `-2`, `--interleaved` | paired-end input — see [Paired-end](#paired-end) |
| `-v`, `-vv`, `-q` | log verbosity; `RUST_LOG` wins |

Full reference, output semantics and examples: [docs/align.md](docs/align.md).

## Paired-end

Illumina read pairs are aligned **as pairs**: the reference that maximises the
two mates' summed score wins, so a mate that is ambiguous alone (paralogs,
isodecoders) is placed by its partner. Input is two FASTQs, an interleaved
FASTQ, or an unaligned BAM with 0x1/0x40/0x80 flags:

```bash
eschalign -1 R1.fq.gz -2 R2.fq.gz -r panel.fa -o aligned.bam --min-score 40
eschalign pairs.fq.gz --interleaved -r panel.fa -o aligned.bam --min-score 40
eschalign pairs.ubam -r panel.fa -o aligned.bam --min-score 40
```

3' adapter read-through is detected first (TruSeq prefixes by default) and kept
as soft clip with an `ad:i` tag; the rest of each mate is aligned end to end, so
a mismatch at a read end stays an `MD`/`NM` mismatch (paired runs default to
`--mode semiglobal`). Pairs over `--max-insert` (default 500) or facing the wrong
way are written without 0x2, each mate at its own best reference. Paired runs
need a `--min-score`. See [docs/align.md](docs/align.md#paired-end) for the
rules, flags and every decision.

## GPU

A `--features gpu` build can score on a CUDA GPU (one thread per
read-reference pair). `--device auto` (default) uses it when a device is
visible, `--device gpu` requires it, `--device cpu` forbids it. Only scores come
from the device, so the output is byte-identical either way. Nothing CUDA is
needed to build; at run time it needs the driver and libnvrtc:

```bash
pixi run install-gpu     # once, on a networked node: creates the `gpu` env
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 pixi run -e gpu ./target/release/eschalign reads.ubam -r trna.fa -o out.bam --device gpu
```

## Environment variables

`ESCHALIGN_BACKEND=scalar|avx2|avx512` caps the SIMD kernel and
`ESCHALIGN_ROW_MAJOR_ONLY=1` turns the long-read paths off (both for A/B
measurement). The escapepod-rs names `ESCAPEPOD_ALIGN_BACKEND` and
`ESCAPEPOD_ALIGN_ROW_MAJOR_ONLY` still work as **deprecated aliases**.

## Tests

```bash
srun -p rna -c 32 --mem=32G pixi run -e dev cargo nextest run --workspace
srun -p rna -c 32 --mem=32G pixi run -e dev cargo test --doc --workspace
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 pixi run -e dev-gpu cargo nextest run --features gpu -p eschalign --test gpu_parity
```

## Benchmarks

`benchmarks/benchmark_align.sh` is the interleaved end-to-end harness;
`benchmarks/README.md` has the measurements carried over from escapepod-rs.

## Layout

- `crates/eschalign` — the library: DP, SIMD/CUDA kernels, `MD`/`NM` (std only, optional `gpu`)
- `crates/eschalign-cli` — the `eschalign` binary: BAM/FASTA/FASTQ I/O and the pipeline

## License

MIT, see [LICENSE](LICENSE).
