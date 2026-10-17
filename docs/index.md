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

```bash
eschalign reads.ubam -r trna.fa -o aligned.bam
```

## Where to go next

- [Installation](installation.md): build from source, pixi environments.
- [Usage](align.md): every option, output semantics and examples.
- [GPU](gpu.md): CUDA panel scoring with `--device`.
- [Benchmarks](benchmarks.md): measurements carried over from escapepod-rs.
- [Development](development.md): build, test and Slurm rules, environment switches.
- [Changelog](changelog.md)

eschalign was carved out of [escapepod-rs](https://github.com/rnabioco/escapepod-rs)
0.31.0, where it was `escpod align`; the history of the code is preserved.
Released under the MIT license.
