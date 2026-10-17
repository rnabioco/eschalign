# eschalign benchmarks

Carved out of escapepod-rs 0.31.0. The sections below are that repository's `escpod align` benchmark notes, kept as measurements: the binary was `escpod align`, the library `escapepod_align`, the environment caps `ESCAPEPOD_ALIGN_BACKEND` / `ESCAPEPOD_ALIGN_ROW_MAJOR_ONLY` (now `eschalign`, `eschalign`, `ESCHALIGN_BACKEND` / `ESCHALIGN_ROW_MAJOR_ONLY`; the old names remain deprecated aliases). The command lines have been updated to the new names; the numbers were measured on the original binary and not re-measured here.

The harness is `benchmarks/benchmark_align.sh`; see its header for the flags.

## `eschalign`: long reads — reference-major score kernel and panel scoring across the pool (2026-09-26)

A tRNA sample's median read is ~130 nt, but M1 carries 175 reads over
4,096 nt, 52 over 20 kb and one of 395 kb. The phase-1 score kernel walks the
reference column by column down the read, keeping `H`/`E` per *read base*: 50
MB per lane group for the 395 kb read, streamed once per reference column, so
it is memory-bound on exactly the reads that cost the most, and the chunk
holding such a read ran for seconds on one thread while the ordered writer —
and, once the permit window filled, the pool — waited for it (#415). Two
changes, output byte-identical:

- **A reference-major ("transposed") layout of the same kernel**
  (`simd::{avx2,avx512}::score_group_transposed`): read row outer, reference
  column inner, state = one panel row per lane group (~25 KB at 200 nt × 32
  lanes, any read length). Same recurrence and op count per cell, reading a
  code-major copy of the profile. `Aligner` uses it for reads of
  `simd::TRANSPOSED_MIN_READ_LEN` = 256 nt or more (was 1,024; re-measured with the non-saturating add).
- **Panel scoring across the pool** for a read of 16 kb or more (the length at
  which `cut_chunks` already gives it a chunk of its own): one rayon task per
  lane group through the new `Aligner::score_group`, the row handed to
  `map_reads_scored` through the `prescored` hook GPU rows use.
  `ESCHALIGN_ROW_MAJOR_ONLY=1` turns both off on the same binary.

**The kernel alone** (`crates/eschalign/examples/long_read_probe.rs`,
one thread, rna / Gold 6240R, synthetic 164-reference panel of 138–200 nt,
27,661 bases, default scoring; cells/s = read length × panel bases):

| read (nt) | avx512 row-major | avx512 transposed | avx2 row-major | avx2 transposed |
|---:|---:|---:|---:|---:|
| 130 | 7.29e9 | 7.19e9 | 7.94e9 | 7.98e9 |
| 500 | 7.19e9 | 7.21e9 | 7.69e9 | 8.00e9 |
| 1,000 | 7.22e9 | 7.22e9 | 7.07e9 | 7.89e9 |
| 2,000 | 7.25e9 | 7.30e9 | 7.20e9 | 7.99e9 |
| 4,000 | 7.27e9 | 7.28e9 | 7.01e9 | 7.90e9 |
| 16,000 | 5.00e9 | 7.29e9 | 5.82e9 | 7.90e9 |
| 64,000 | 5.00e9 | 7.29e9 | 4.77e9 | 7.84e9 |
| 256,000 | 4.33e9 | 7.27e9 | 4.79e9 | 7.84e9 |
| 400,000 | 3.14e9 | 7.27e9 | 4.64e9 | 7.85e9 |

(local mode; semi-global, where the transposed kernel skips the per-cell
column max, is 8.6–8.7e9 transposed against 5.1–8.0e9 row-major at every
length from 130 nt up.) The transposed kernel is flat at its in-cache rate
at every length; the row-major one falls off once its per-read state leaves
L2 (between 4 k and 16 k nt at 32 lanes) and reaches 2.3× slower at 400 kb.
Below ~1 kb the two are within noise on AVX-512 (±1%), and AVX2's transposed
kernel is 0–4% ahead, so the threshold is 1,024 nt: the first length at which
the transposed kernel is at least as fast on both ISAs in both modes, well
above any real tRNA read, so the row-major kernel — measured and tuned for
those — stays the default for them.

With the non-saturating add (`NEG16`, plain `vpaddw`) the row-major kernel is flat to
200 nt on AVX-512, drops at 250 nt (L1: 128 B of state per read base at 32 lanes)
and loses to the transposed kernel from ~300 nt; AVX2's transposed kernel is at
least as fast at every length. The threshold is therefore 256 nt.

**M1's 395 kb read alone**, real panel (164 references, 24,790 bases):

| | avx512 | avx2 |
|---|---:|---:|
| row-major, one thread | 3.15 s | 2.09 s |
| row-major, one thread per lane group | 1.24 s (6 groups) | 0.84 s (11 groups) |
| transposed, one thread | 1.37 s | 1.27 s |
| transposed, one thread per lane group | **0.29 s** | **0.15 s** |

(The 4.9 s quoted for this read in the section below was measured inside a
full run on a busier node; 3.15 s is the probe's quiet-node number for the
same kernel.) The split that did not help with the row-major layout — six
50 MB streams contending for the same memory — is 4.7× with the transposed
one.

**M1 end to end, CPU arm** — `srun -p rna -c 32 --mem=32G`, `-t 32`, `-v`,
the base commit (`20fc3a0`) and this change (`ec51e37`) as two `--bin` arms of
`benchmarks/benchmark_align.sh`, interleaved, both `--features gpu` release
builds scoring on the CPU (no GPU on rna):

| arm | wall (s) | CPU (s) | MaxRSS (MiB) | workers busy (s of wall × 32) |
|---|---:|---:|---:|---:|
| base, avx512 | 19.2 / 19.5 | 479 / 482 | 2,346 / 2,298 | 477 of 611 / 480 of 621 |
| **new, avx512** | **15.2 / 15.3** | 467 / 470 | 1,772 / 1,783 | 458 of 486 / 460 of 490 |
| base, avx2 | 17.3 / 17.5 | 506 / 509 | 2,183 / 2,177 | 503 of 550 / 507 of 557 |
| **new, avx2** | **15.9 / 15.9** | 487 / 489 | 1,543 / 1,552 | 481 of 506 / 483 of 506 |

(`rep 1 / rep 2`.) 58 read strands of 16 kb or more were scored across the
pool. The pool is now busy 94–95% of the run (78% before): what is left is the
work itself, not waiting on stragglers. RSS drops ~0.55 GB — the chunks that
used to pile up behind a long read in the permit window no longer do.
`--strand both --secondary --mode semiglobal`, avx512, one rep: 34.9 → 26.1 s
wall, 851 → 820 CPU-s, 116 strands split.

**M1 end to end, GPU arm** — `sbatch -p gpu -A gpu_rbi -c 16 --gres=gpu:1
--mem=48G` (compgpu03: Xeon Gold 6526Y, one A30), `-t 16`, `-v`, same two
binaries, `--device cpu` and `--device gpu`, avx512, interleaved. The
2026-09-25 numbers above were on compgpu01 (Gold 6326), whose CPU side is
slower — 14.3 s there for the base binary's GPU arm, 11.1 s here — so the
target, "the run without its long reads", was re-measured on this node with
`--read-ids` naming M1's 489,743 reads under 20 kb:

| arm | wall (s) | CPU (s) | MaxRSS (MiB) | workers busy (s of wall × 16) |
|---|---:|---:|---:|---:|
| base, `--device gpu` | 11.2 / 11.1 | 87 / 87 | 2,596 / 2,603 | 63 of 178 / 64 of 176 |
| **new, `--device gpu`** | **7.7 / 7.7** | 85 / 84 | 2,029 / 1,935 | 55 of 122 / 56 of 122 |
| base, `--device gpu`, reads < 20 kb only | 7.6 / 7.4 | 63 / 63 | 1,663 / 1,591 | 45 of 120 / 45 of 118 |
| new, `--device gpu`, reads < 20 kb only | 7.4 / 7.4 | 63 / 63 | 1,615 / 1,633 | 46 of 118 / 45 of 117 |
| base, `--device cpu` | 25.1 / 25.0 | 394 / 394 | 1,712 / 1,703 | 395 of 400 / 394 of 398 |
| new, `--device cpu` | 24.7 / 24.7 | 390 / 390 | 1,262 / 1,268 | 379 of 394 / 380 of 395 |

**The GPU arm lands on its floor**: 7.7 s with every read against 7.4–7.6 s
without the 52 over 20 kb, so the long-read tail costs ~0.2 s where it cost
3.6 s; the GPU thread's own time waiting on the device is 6.2 s of that
(the same in every arm), which is now the bound. `--strand both --secondary
--mode semiglobal`, `--device gpu`, one rep: 19.0 → 11.8 s. The 16-thread CPU
arm was already work-bound on this node (the pool 98–99% busy before; one
long read's seconds are absorbed while 15 threads keep working), so it gains
only the long reads' cheaper kernel (−1%) and the ~0.45 GB of RSS the backed-up
chunks held.

**`--max-read-len` (default 1000 nt, added in the same PR).** No read over
1,000 nt is a tRNA read in practice, so the default now writes them unmapped
without scoring them (M1: 687 reads, 0.14%). avx512, `-t 32` on rna and
`--device gpu -t 16` on compgpu01 (Gold 6326 + A30), same binary
(`876ac45`), default against `--max-read-len 0`, interleaved:

| arm | wall (s) | CPU (s) | MaxRSS (MiB) |
|---|---:|---:|---:|
| rna, default (limit 1000) | 13.4 / 13.3 | 418 / 417 | 1,214 / 1,234 |
| rna, `--max-read-len 0` | 15.1 / 15.1 | 464 / 463 | 1,821 / 1,772 |
| gpu, default (limit 1000) | 6.5 / 6.6 | 67 / 66 | 1,388 / 1,486 |
| gpu, `--max-read-len 0` | 7.6 / 7.7 | 99 / 98 | 2,365 / 2,473 |

With the limit, the GPU scores 489,108 of 489,108 queries and nothing falls
back to the CPU. `--max-read-len 0` output is md5-equal to the numbers
below (`4bb45893…` at the defaults, `ecd70db3…` at `--strand both
--secondary --mode semiglobal`). At the default limit no read reaches the
transposed kernel's 1,024 nt threshold or the 16 kb pool split, so neither
runs; both stay for `--max-read-len 0` or a limit above 1,024.

**Output.** `samtools view | md5sum` is equal between base and new, and
between every backend and device arm, at the defaults
(`4bb45893f63827f603931cf1a5b4ad57`) and at `--strand both --secondary --mode
semiglobal` (`ecd70db3e34e0f4f78eedfb87d00955c`). In tests:
`simd::tests::transposed_matches_scalar_random` (every SIMD backend, both
modes, five schemes including a mismatch of 0 and of +1, a 37-reference
ragged panel, reads either side of the threshold and 20–56 kb, equal to the
scalar oracle), `transposed_matches_row_major_on_fixture_reads`,
`mapper::tests::per_group_scores_equal_score_all`, and
`align_e2e::long_read_alignment_matches_scalar_backend` (fixture reads plus
40 kb and 20 kb synthetic reads, `-t 4`, default backend against
`ESCHALIGN_BACKEND=scalar`, two flag sets, every record equal).

Reproduce:

```bash
srun -p rna -c 32 --mem=32G -- cargo run --release -p eschalign \
    --example long_read_probe -- --split
srun -p rna -c 32 --mem=32G -- cargo run --release -p eschalign \
    --example long_read_probe -- --reference .../sacCer3-mature-tRNAs-dual-adapt-v2.fa \
    --reads longest_read.fa --split
srun -p rna -c 32 --mem=32G -- benchmarks/benchmark_align.sh \
    --reads .../rebasecall/M1/M1.rbc.bam \
    --reference .../sacCer3-mature-tRNAs-dual-adapt-v2.fa \
    --backend avx512 --backend avx2 --reps 2 \
    --bin eschalign-base --bin eschalign-new --md5 -- -v
```

## `eschalign --device gpu`: CUDA panel scoring (2026-09-25)

`eschalign` phase 2 (#401) moves the one stage worth moving — scoring every
read against every reference, ~77% of phase 1's cycles — onto a CUDA kernel
(`eschalign::cuda`, one GPU thread per (read, reference) pair, a whole
`--batch-size` batch per launch); the tie set, the tracebacks, `MD`/`NM` and
the records stay on the CPU and are the CPU path's own code. Same sample,
panel and scoring as the phase 1 numbers below (M4, 3,296,074 reads, 164
references, local, `2,-1,-10,-1`). Both arms on **one gpu node** —
`sbatch -p gpu -A gpu_rbi -c 16 --gres=gpu:1` (compgpu01: Xeon Gold 6326, Ice
Lake, AVX-512; one A30), `-t 16`, the same `--features gpu` release build of
`36cd69e` for both (`--device cpu` against `--device gpu`, avx512 kernel),
arms interleaved, input paged in first, output to node-local scratch:

| arm | wall (s) | CPU (s) | MaxRSS (MiB) | reads/s |
|---|---:|---:|---:|---:|
| `--device cpu` | 183.8 / 184.2 | 2,898 / 2,900 | 1,967 / 1,951 | 17,930 / 17,895 |
| `--device gpu` | 69.2 / 69.0 | 685 / 682 | 3,079 / 3,066 | 47,659 / 47,790 |

(`rep 1 / rep 2`.) **The GPU arm wins on wall (2.66×) at a quarter of the
CPU (4.2× less), so `--device auto` places `align`'s scoring on the GPU**
(`device::ALIGN_CPU_COST`); it is also under phase 1's bar of ~117 s and
~3,300 CPU-s on a whole rna node (`-c 32`). It holds ~1.1 GB more RSS — most
likely the CUDA context and the batches the GPU thread holds (one on the
device, two queued for the dispatcher). The output is byte-identical: `align_e2e::device_gpu_output_is_byte_identical`
over four flag sets, and `samtools view | md5sum` of M1 (0.49 M reads) equal
between the arms at the defaults and at `--strand both --secondary --mode
semiglobal`. 3,295,084 of the 3,296,074 reads were scored on the GPU; the
other 990 are over 4,096 nt and were scored on the CPU in the same run.

**The kernel** (`examples/gpu_score_probe.rs`, `score_batch` alone on M1's
reads): 2.4e11 cells/s on the A30 — ~20 integer ops per DP cell against
Ampere's 64 INT32 lanes per SM per clock, so it is ALU-bound, as the CPU
kernels are. `__launch_bounds__(256, 4)` (64 registers, four blocks per SM)
took M1 from 6.4 s to 5.9 s; forcing 56 or 40 registers spills and loses. The
164-reference panel fills 164 of 192 threads per block. A 400 kb read on the
GPU is the one thing it does badly — one block owns a read serially, and
giving it M1's 175 reads over 4,096 nt took the kernel from 6 s to 17.5 s —
hence the cap.

**Why the GPU is not busy the whole run.** With scoring gone from the CPU a
batch costs the workers ~4× less, and the run is bound by the CPU side, not by
the device: M4's GPU thread spends 12.8 s of 69 s waiting on the device. The
GPU thread packs batch *k + 1* while batch *k*'s kernel runs (before that
overlap utilisation dipped to 60–80% at every batch boundary while the host
packed; the wall did not move, since the device was not the bottleneck), so `nvidia-smi` shows
~100% while there is work and 0% while the CPU side is stalled — "spiky". The
stalls are the very long reads: M1 without its 52 reads over 20 kb runs in
7.5 s instead of 14.3 s. Those reads fall to the CPU, where phase 1's score
kernel is slow on a read that long (4.9 s for a 395 kb read on one core,
2e9 cells/s: its per-row state is 50 MB, streamed once per reference column),
and the chunk holding one blocks the ordered writer until the permit window
(two batches) fills. Tried: splitting that read's scoring over the six lane
groups on six threads (no faster: its chunk still took 3.4–3.9 s), and a
larger window (4 batches: 14.3 → 12.6 s for +0.7 GB; 8: 11.5 s for +2 GB — not
taken). The fix that should work is a row-tiled CPU score kernel for reads far
longer than the panel, which helps both arms; it is a phase 1 kernel change and
not part of this one. (Done in #415 — see the long-reads section above: this
arm now runs M1 in 7.7 s against 11.1 s on the same node.)

Reproduce:

```bash
sbatch -p gpu -A gpu_rbi -c 16 --gres=gpu:1 --mem=48G --wrap \
  "benchmarks/benchmark_align.sh --reads .../rebasecall/M4/M4.rbc.bam \
     --reference .../sacCer3-mature-tRNAs-dual-adapt-v2.fa \
     --backend avx512 --device cpu --device gpu --reps 2 --threads 16"
```

(`SBATCH_PARTITION`/`SBATCH_ACCOUNT` in the environment override `#SBATCH`
lines in a script, so give `-p`/`-A` on the command line.)

## `eschalign`: all-vs-all tRNA alignment, first numbers (2026-09-25)

`eschalign` (#395) scores every read against every reference of the
sacCer3 dual-adapter panel (164 references, 138–200 nt, 24,790 bases) and
traces the winners. The cost is `reads × references × lengths` and nothing
else, so any real sample measures it; these are dorado uBAMs from
`aa-tRNA-seq-pipeline-runs/results/2025-06-09/outs/bam/rebasecall/` (M4:
3,296,074 reads, 1.5 GB; M1: 489,795 reads, mean length 122 nt, max ~400 kb),
default scoring (`2,-1,-10,-1`, local), aligned against
`sacCer3-mature-tRNAs-dual-adapt-v2.fa`. `srun -p rna -c 32 --mem=32G`
(Gold 6240R, 16 cores + HT), `-t 32`, release build of `a3047cb`,
`benchmarks/benchmark_align.sh` (arms interleaved, input paged in first,
output to node-local scratch):

| sample | kernel | wall (s) | CPU (s) | MaxRSS (MiB) | reads/s |
|---|---|---:|---:|---:|---:|
| M4, 3.30 M reads | avx512 | 117.0 / 117.6 | 3,256 / 3,264 | 2,746 / 2,717 | 28,172 / 28,023 |
| M4 | avx2 | 114.1 / 113.2 | 3,407 / 3,402 | 2,339 / 2,349 | 28,895 / 29,120 |
| M1, 0.49 M reads | avx512 | 19.0 / 19.5 | 483 / 480 | 2,356 / 2,329 | 25,779 / 25,156 |
| M1 | avx2 | 16.8 / 17.0 | 499 / 500 | 2,229 / 2,203 | 29,120 / 28,778 |
| M1 | scalar | 484.8 / 484.8 | 15,273 / 15,274 | 1,497 / 1,484 | 1,010 / 1,010 |

(Two reps each, `rep 1 / rep 2`.) The step it replaces — `bwa mem -k 6`
+ `samtools calmd` in aa-tRNA-seq-pipeline — is budgeted 160 GB and 12 h per
sample and measured at 78–90 GiB RSS; this is ~2 minutes and under 3 GiB for a
3.3 M-read sample, and memory is set by `--batch-size`, not by the reads.

**AVX-512 does not beat AVX2 here, and that is the port count, not the
kernel.** On Cascade Lake, 512-bit integer ops issue on ports 0 and 5 only
(port 1's 512-bit path is fused off), while 256-bit ops use 0, 1 and 5: 32
lanes × 2 ports against 16 lanes × 3 is 64 against 48 lane-ops per cycle
before the AVX-512 frequency licence, and the kernel is ALU-bound (11 vector
ops per cell against 3 loads and 2 stores, ~5–7 cycles per 32-lane cell
measured). AVX-512 still spends ~4% less CPU; AVX2 finishes ~3% sooner on wall,
most likely because the licence also slows the BGZF threads sharing the cores.
The dispatch keeps preferring AVX-512 (CPU is the shared currency on a busy
node); `ESCHALIGN_BACKEND=avx2` is the lever. Scalar is 32× the SIMD CPU.

**How it got here (M1, CPU seconds / wall seconds, output byte-identical at
every step):**

| change | CPU (s) | wall (s) |
|---|---:|---:|
| first version: SIMD scoring, scalar traceback for every winner | 1,019 | 40.1 |
| tie sets traced by an inter-sequence SIMD kernel (one lane per tied reference) | 550 | 26.6 |
| one *pairs* kernel for all tracebacks (each lane its own read and reference; a chunk's winners share lanes) | ~470 | 24.0 |
| no barrier between batches; chunks capped at 16 kb; writer reorders | ~470 | 18.8 |

The first profile was 65% scalar traceback. About 5% of M1's reads are
adapter-only and tie with **all 164** references (another 2% with 81), and
each tie got its own scalar DP at ~45 instructions per cell. Then the unique
winners' scalar tracebacks were 19% of CPU; the pairs kernel took all
tracebacks to ~5%, and the score kernel is now ~77% of cycles. The last row is
wall only: a per-batch `par_iter` waited at every batch for its slowest chunk,
and reads of 50–400 kb make chunks of 0.5–6 s where a typical one is
milliseconds — `-v`'s busy counter showed 16 of 32 threads working on average.

Not done, and where the next measurable wins are: the panel's last lane group
holds 4 references in 32 lanes (a 16-lane tail group would save ~5% of the
score kernel); and the score kernel's 11 ops per cell have no obvious fat left.
An 8-bit first pass (SWIPE's trick) does not apply at the default scoring —
a tRNA read's local score passes 255.

Reproduce:

```bash
srun -p rna -c 32 --mem=32G -- benchmarks/benchmark_align.sh \
    --reads .../rebasecall/M4/M4.rbc.bam \
    --reference .../resources/ref/sacCer3-mature-tRNAs-dual-adapt-v2.fa \
    --backend avx512 --backend avx2 --reps 2
```

