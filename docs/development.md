# Development

## Build and test

```bash
cargo build --release                       # target/release/eschalign
cargo nextest run --workspace               # unit + integration tests
cargo test --doc --workspace                # nextest does not run doctests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
```

Pixi tasks in the `dev` environment: `build`, `build-rel`, `test`, `test-doc`,
`check`, `clippy`. mold is the linker full time: the base `[activation]` in
`pixi.toml` exports `LD_PRELOAD=$CONDA_PREFIX/lib/mold/mold-wrapper.so` for every
environment, so any `pixi run [-e <env>] cargo …` links with mold.

## Build baseline

Local Linux/x86_64 builds pin `-C target-cpu=x86-64-v3` via `.cargo/config.toml`
(AVX2 + FMA + BMI2 + POPCNT + F16C). This is portable across every node in the
cluster (Broadwell login, Cascade Lake rna, Ice Lake gpu). Do **not** use
`target-cpu=native`: a binary built on a gpu node uses Ice Lake-only instructions
that SIGILL on rna. The AVX-512 kernels are `#[target_feature]` plus runtime
`is_x86_feature_detected!` dispatch, never a global baseline bump. An environment
`RUSTFLAGS` replaces the config file's flags wholesale rather than adding to them.

The SIMD equality tests sweep `Backend::available()`; run them on `rna` (Cascade
Lake, AVX-512) or the AVX-512 kernels never execute.

## Slurm: Beevol

The login node has only 2 cores. Wrap any heavy build, test or bench in `srun`
and run it in the foreground:

```bash
srun -p rna -c 32 --mem=32G -J eschalign-build --comment=eschalign-build -- \
    pixi run -e dev cargo nextest run --workspace
```

For throughput-sensitive runs ask for a full socket (48 logical = 24 physical +
HT). `-c 32` only allocates 16 physical cores on rna's Gold 6240R; `-c 64` and up
crosses sockets and is unpredictable on a shared node.

```bash
srun -p rna -c 48 --mem=64G pixi run ./target/release/eschalign …
```

GPU tests (account `gpu_rbi`, partition `gpu`):

```bash
srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 \
    pixi run -e dev-gpu cargo nextest run --features gpu -p eschalign --test gpu_parity
```

## Running on Alpine

On Alpine (CU Boulder RMACC/ACCESS) every partition needs an explicit `--qos=`
matched to the partition, plus `-A amc-general`.

| Purpose | Beevol | Alpine |
|---|---|---|
| CPU build / test / bench | `srun -p rna -c 32 --mem=32G` | `srun -p amilan --qos=normal -A amc-general -c 32 --mem=32G` |
| CPU full node (throughput) | `srun -p rna -c 48` | `srun -p amilan --qos=normal -A amc-general -c 64 --mem=120G` |
| GPU test / bench | `srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1` | `srun -p aa100 --qos=gpu-normal -A amc-general -c 16 --gres=gpu:1` |
| Light build (login is 2 cores) | login node | `srun -p acompile --qos=compile -A amc-general -c 16 --mem=32G` |

QOS to partition: amilan is `normal` or `long`; aa100 and al40 are `gpu-normal`
or `gpu-long`; acompile is `compile`; atesting is `testing` (1 h cap). Default
memory is 3840 MB per core.

- `amilan` nodes are AMD EPYC Milan, 2×32 = 64 physical cores, no hyperthreading, 245 GB, so the Beevol "`-c 48` fills a socket" tuning does not apply.
- GPU CUDA work must use `aa100` (A100) or `al40` (L40). `ami100` is AMD/ROCm and cannot run the CUDA `gpu` feature.
- amilan, aa100 and acompile are all Zen3 x86-64, so the pinned `x86-64-v3` baseline is portable across them. The AVX-512 kernels do not run on Zen3; the AVX2 ones do.
- The toolchain lives in pixi, not on the bare PATH, e.g. `srun -p amilan --qos=normal -A amc-general -c 32 --mem=32G pixi run -e dev test`.

## Environment switches

| Variable | Effect |
|---|---|
| `ESCHALIGN_BACKEND` | Cap the SIMD kernel: `scalar`, `avx2` or `avx512` (read in the library) |
| `ESCHALIGN_ROW_MAJOR_ONLY` | `1`/`true`/`yes`/`on` turns the long-read paths off (A/B only); `0`/`false`/`no`/`off`/unset leaves them on |

The escapepod-rs names `ESCAPEPOD_ALIGN_BACKEND` and
`ESCAPEPOD_ALIGN_ROW_MAJOR_ONLY` still work as **deprecated aliases**; the
`ESCHALIGN_*` name wins whenever it is set. A boolean switch is on for
`1|true|yes|on` and off for unset, empty or `0|false|no|off` (case-insensitive);
anything else warns once and counts as off.

## Logging

All status, progress and warning output goes through `tracing` to stderr; only
the BAM under `-o -` goes to stdout. The default level is info for eschalign's
own crates and warn for dependencies. `-v`/`-vv` raise only eschalign's level,
`-q` is errors only, and `RUST_LOG` always wins when set.

## Benchmarking

See [Benchmarks](benchmarks.md#reproducing). Profiles: `release` (fat LTO,
`codegen-units=1`, stripped, `panic=abort`), `dist`, `release-with-debug`,
`profiling` (LTO off for readable call graphs) and `bench`.

## Documentation

This site is built with [zensical](https://zensical.org) from `docs/` and
`zensical.toml`:

```bash
srun -p rna -c 4 --mem=8G -- pixi run -e docs docs-check   # strict build
pixi run -e docs docs-serve                                # live preview
```

`docs-check` is `zensical build --clean --strict`, the same gate CI runs.
`--clean` forces a cold cache because the incremental cache makes strict link
validation nondeterministic.
