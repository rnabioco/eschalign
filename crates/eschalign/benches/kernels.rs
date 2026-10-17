// SPDX-License-Identifier: MIT

//! Criterion benchmarks of the CPU kernels, on the synthetic 164-reference
//! panel (138–200 nt, the shape of the sacCer3 dual-adapter panel) that
//! `examples/long_read_probe.rs` uses, so the numbers line up with
//! `benchmarks/README.md`.
//!
//! ```text
//! srun -p rna -c 32 --mem=32G -- pixi run -e dev cargo bench -p eschalign --bench kernels
//! cargo bench -p eschalign --bench kernels -- score_all/local/avx512   # a filter
//! ```
//!
//! Groups: `score_all` (one read against the whole panel, per backend, mode
//! and read length, throughput in DP cells), `loop_order` (row-major against
//! reference-major around `TRANSPOSED_MIN_READ_LEN`), and `map_reads` (a
//! 64-read chunk through score, tie selection, traceback and `MD`/`NM`, the
//! unit the CLI hands a rayon worker). Every benchmark is one thread; the
//! backends are `Backend::available()`, so AVX-512 only appears on a node
//! that has it. `ESCHALIGN_BACKEND` is not read here.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use eschalign::alphabet::encode;
use eschalign::simd::Kernel;
use eschalign::{Aligner, Backend, MapOptions, Mode, Panel, Scoring};

fn xorshift(x: &mut u64) -> u64 {
    *x ^= *x << 13;
    *x ^= *x >> 7;
    *x ^= *x << 17;
    *x
}

fn random_seq(x: &mut u64, len: usize) -> Vec<u8> {
    (0..len)
        .map(|_| b"ACGT"[(xorshift(x) % 4) as usize])
        .collect()
}

fn panel() -> Panel {
    let mut x = 0x0123_4567_89ab_cdefu64;
    Panel::new((0..164).map(|k| {
        let len = 138 + (xorshift(&mut x) % 63) as usize;
        (format!("r{k}"), random_seq(&mut x, len))
    }))
    .unwrap()
}

/// A read of `len` bases taken from reference `k` of the panel (wrapping
/// around it for a long one) with a substitution about every 20 bases, so
/// the traceback has real mismatches to report.
fn read_like(panel: &Panel, x: &mut u64, k: usize, len: usize) -> Vec<u8> {
    let src = &panel.get(k % panel.len()).seq;
    (0..len)
        .map(|i| {
            let b = src[i % src.len()];
            if xorshift(x).is_multiple_of(20) {
                b"ACGT"[(xorshift(x) % 4) as usize]
            } else {
                b
            }
        })
        .collect()
}

const MODES: [Mode; 2] = [Mode::Local, Mode::SemiGlobal];
const LENGTHS: [usize; 4] = [100, 130, 200, 1000];

fn backends() -> impl Iterator<Item = Backend> {
    Backend::available()
        .into_iter()
        .filter(|&b| b != Backend::Scalar)
}

fn score_all(c: &mut Criterion) {
    let panel = panel();
    let bases = panel.total_len() as u64;
    let mut x = 0xfeed_beef_u64;
    for mode in MODES {
        let mut g = c.benchmark_group(format!("score_all/{}", mode.name()));
        for backend in backends() {
            let a =
                Aligner::with_backend(panel.clone(), Scoring::default(), mode, backend).unwrap();
            for len in LENGTHS {
                let q = encode(&read_like(&panel, &mut x, len, len));
                g.throughput(Throughput::Elements(len as u64 * bases));
                g.bench_with_input(BenchmarkId::new(backend.name(), len), &q, |b, q| {
                    let mut out = Vec::new();
                    b.iter(|| a.score_all(black_box(q), &mut out));
                });
            }
        }
        g.finish();
    }
}

/// Row-major against reference-major at the lengths around
/// `TRANSPOSED_MIN_READ_LEN`; the measurement that threshold is set from
/// (`examples/long_read_probe.rs` is the one to re-run for the long end).
fn loop_order(c: &mut Criterion) {
    let panel = panel();
    let bases = panel.total_len() as u64;
    let mut x = 0xabad_1dea_u64;
    let mut g = c.benchmark_group("loop_order/local");
    for backend in backends() {
        let base =
            Aligner::with_backend(panel.clone(), Scoring::default(), Mode::Local, backend).unwrap();
        for len in [130, 250, 300, 600, 1000] {
            let q = encode(&read_like(&panel, &mut x, len, len));
            g.throughput(Throughput::Elements(len as u64 * bases));
            for (name, kernel) in [
                ("row_major", Kernel::RowMajor),
                ("transposed", Kernel::Transposed),
            ] {
                let a = base.clone().with_transposed_min_len(match kernel {
                    Kernel::RowMajor => None,
                    Kernel::Transposed => Some(0),
                });
                g.bench_with_input(
                    BenchmarkId::new(format!("{}/{name}", backend.name()), len),
                    &q,
                    |b, q| {
                        let mut out = Vec::new();
                        b.iter(|| a.score_all(black_box(q), &mut out));
                    },
                );
            }
        }
    }
    g.finish();
}

/// A 64-read chunk end to end on the CPU: score every read against the
/// panel, pick the ties, trace the winners together, build `MD`/`NM`.
fn map_reads(c: &mut Criterion) {
    let panel = panel();
    let mut x = 0x5eed_c0de_u64;
    let reads: Vec<Vec<u8>> = (0..64)
        .map(|k| {
            let len = 100 + (xorshift(&mut x) % 100) as usize;
            read_like(&panel, &mut x, k * 7, len)
        })
        .collect();
    let refs: Vec<&[u8]> = reads.iter().map(Vec::as_slice).collect();
    let mut g = c.benchmark_group("map_reads/chunk64");
    g.throughput(Throughput::Elements(refs.len() as u64));
    for mode in MODES {
        for backend in backends() {
            let a =
                Aligner::with_backend(panel.clone(), Scoring::default(), mode, backend).unwrap();
            g.bench_function(BenchmarkId::new(mode.name(), backend.name()), |b| {
                b.iter(|| black_box(a.map_reads(black_box(&refs), &MapOptions::default())));
            });
        }
    }
    g.finish();
}

criterion_group!(benches, score_all, loop_order, map_reads);
criterion_main!(benches);
