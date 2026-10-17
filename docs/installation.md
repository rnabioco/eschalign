# Installation

## From source

Rust 1.95 or later. From a clone of
[rnabioco/eschalign](https://github.com/rnabioco/eschalign):

```bash
cargo build --release                                   # target/release/eschalign
cargo build --release --features gpu -p eschalign-cli   # with the CUDA score kernel
```

Local builds pin `-C target-cpu=x86-64-v3` (`.cargo/config.toml`); the AVX-512
kernels are chosen at run time. Do not use `target-cpu=native`: see
[Development](development.md#build-baseline).

## On a cluster

Use the pixi environments, which provide the mold linker, samtools,
cargo-nextest and the CUDA runtime. Do not build on a login node:

```bash
srun -p rna -c 32 --mem=32G pixi run -e dev build-rel
```

See [Development](development.md) for the Beevol and Alpine `srun` forms.

## GPU runtime

Nothing CUDA is needed to build. For the run-time environment see
[GPU](gpu.md).

## First run

```bash
eschalign reads.ubam -r trna.fa -o aligned.bam
```

Options and output are documented under [Usage](align.md).
