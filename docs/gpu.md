# GPU

A binary built with `--features gpu` can score on a CUDA GPU: one GPU thread per
(read, reference) pair, a whole `--batch-size` batch per kernel launch. Only the
scores come from the device. The tie set, the tracebacks, `MD`/`NM` and the
records are the CPU path's own code, so `--device gpu` and `--device cpu` write
the same records byte for byte.

## Placement

| `--device` | Behaviour |
|---|---|
| `auto` (default) | score on the GPU when a device is visible in a `gpu` build, else the CPU |
| `gpu` | demand the device; error instead of falling back |
| `cpu` | forbid the GPU |

Reads over 4,096 nt are scored on the CPU inside the same run, and a panel with
a reference over 3,072 nt cannot use the GPU (`auto` then stays on the CPU and
says so).

## Build and runtime

Nothing CUDA is needed at build time, because cudarc dlopens at run time:

```bash
cargo build --release --features gpu -p eschalign-cli
```

At run time the GPU path needs the CUDA driver (from the node) and NVRTC
(`libnvrtc.so.12`, which compiles the score kernel). Create the `gpu` pixi
environment once, on a networked node:

```bash
pixi run install-gpu     # CONDA_OVERRIDE_CUDA=12 pixi install -e gpu
```

Then run on a GPU node:

```bash
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 \
    pixi run -e gpu ./target/release/eschalign reads.ubam -r trna.fa -o out.bam --device gpu
```

On Alpine use `aa100` (A100) or `al40` (L40); `ami100` is AMD/ROCm and cannot
run the CUDA feature. See [Development](development.md#running-on-alpine).

## Tests

```bash
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 \
    pixi run -e dev-gpu cargo nextest run --features gpu -p eschalign --test gpu_parity
```

Use `-e dev-gpu`, not `-e gpu`: cargo-nextest lives in the `dev` feature.

## Performance

On one A30 (`-c 16`), the 3.3 M-read sample takes ~70 s and ~700 CPU-seconds
with GPU scoring against 184 s and ~2,900 CPU-seconds without it on the same
16 cores. See [Benchmarks](benchmarks.md) for the arms and caveats.
