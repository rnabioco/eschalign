# Benchmarks

These are the `escpod align` measurements carried over from escapepod-rs 0.31.0.
The binary was `escpod align` then; the numbers were measured on the original
binary and not re-measured for eschalign. The full notes, including the
long-read kernel work, are in
[`benchmarks/README.md`](https://github.com/rnabioco/eschalign/blob/main/benchmarks/README.md).

Samples are dorado uBAMs aligned against the sacCer3 dual-adapter panel (164
references, 138–200 nt, 24,790 bases) with default scoring (`2,-1,-10,-1`,
local). M4 has 3,296,074 reads; M1 has 489,795 reads (mean length 122 nt, max
~400 kb).

## CPU

`srun -p rna -c 32 --mem=32G` (Gold 6240R, 16 cores + HT), `-t 32`. Two reps,
`rep 1 / rep 2`.

| sample | kernel | wall (s) | CPU (s) | MaxRSS (MiB) | reads/s |
|---|---|---:|---:|---:|---:|
| M4, 3.30 M reads | avx512 | 117.0 / 117.6 | 3,256 / 3,264 | 2,746 / 2,717 | 28,172 / 28,023 |
| M4 | avx2 | 114.1 / 113.2 | 3,407 / 3,402 | 2,339 / 2,349 | 28,895 / 29,120 |
| M1, 0.49 M reads | avx512 | 19.0 / 19.5 | 483 / 480 | 2,356 / 2,329 | 25,779 / 25,156 |
| M1 | avx2 | 16.8 / 17.0 | 499 / 500 | 2,229 / 2,203 | 29,120 / 28,778 |
| M1 | scalar | 484.8 / 484.8 | 15,273 / 15,274 | 1,497 / 1,484 | 1,010 / 1,010 |

On Cascade Lake AVX-512 does not beat AVX2 here: 512-bit integer ops issue on
ports 0 and 5 only, and the kernel is ALU-bound. AVX-512 spends ~4% less CPU;
AVX2 finishes ~3% sooner on wall. `ESCHALIGN_BACKEND=avx2` is the lever.

The step this replaces (`bwa mem -k 6` + `samtools calmd` in aa-tRNA-seq-pipeline)
is budgeted 160 GB and 12 h per sample and measured at 78–90 GiB RSS; eschalign
is ~2 minutes and under 3 GiB for a 3.3 M-read sample, and memory is set by
`--batch-size`, not by the reads.

## GPU

Both arms on one gpu node (`-c 16`, Xeon Gold 6326, one A30), the same
`--features gpu` release build, `--device cpu` against `--device gpu`, arms
interleaved, M4:

| arm | wall (s) | CPU (s) | MaxRSS (MiB) | reads/s |
|---|---:|---:|---:|---:|
| `--device cpu` | 183.8 / 184.2 | 2,898 / 2,900 | 1,967 / 1,951 | 17,930 / 17,895 |
| `--device gpu` | 69.2 / 69.0 | 685 / 682 | 3,079 / 3,066 | 47,659 / 47,790 |

The GPU arm wins on wall (2.66×) at a quarter of the CPU, and the output is
byte-identical. See [GPU](gpu.md).

## Long reads

A tRNA sample's median read is ~130 nt, but M1 carries reads up to ~400 kb. Reads
of 1,024 nt or more take a reference-major score kernel, and reads of 16 kb or
more are scored across the thread pool, with byte-identical output.
`ESCHALIGN_ROW_MAJOR_ONLY=1` turns both off for A/B runs. On M1 (rna, `-t 32`) the
avx512 arm went from 19.2 / 19.5 s to 15.2 / 15.3 s wall. Kernel-level numbers
are in `benchmarks/README.md`.

## Reproducing

`benchmarks/benchmark_align.sh` is the interleaved end-to-end harness (its
header documents the flags). Arms (binary × backend × device × sort order) are
interleaved, not grouped, the input is paged in once before the first arm, and
CPU is reported beside wall.

```bash
srun -p rna -c 32 --mem=32G -- benchmarks/benchmark_align.sh \
    --reads sample.ubam --reference trna.fa --md5 --reps 3
```

Benchmark output belongs under `results/` or `~/scratch/<topic>/`, never
`~/.cache`.
