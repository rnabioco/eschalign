// SPDX-License-Identifier: MIT

//! Criterion benchmark of the CUDA score kernels (`gpu` feature): one
//! `GpuScorer::score_batch` over a batch of reads against the synthetic
//! 164-reference panel, the int kernel against the half2 kernel (#9), local
//! and semi-global. Skips with a message where there is no device.
//!
//! ```text
//! srun -p gpu -A gpu_rbi -c 16 --gres=gpu:1 -- \
//!     pixi run -e dev-gpu cargo bench --features gpu -p eschalign --bench gpu
//! ```
//!
//! `examples/gpu_score_probe.rs` stays the tool for real reads and register
//! counts; this is the one for a regression check with a confidence interval.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use eschalign::alphabet::encode;
use eschalign::cuda::{GpuScorer, device_visible};
use eschalign::{Mode, Panel, Scoring};

const BATCH: usize = 20_000;

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

fn gpu(c: &mut Criterion) {
    if !device_visible() {
        eprintln!("gpu bench skipped: no CUDA device visible");
        return;
    }
    let mut x = 0x0123_4567_89ab_cdefu64;
    let panel = Panel::new((0..164).map(|k| {
        let len = 138 + (xorshift(&mut x) % 63) as usize;
        (format!("r{k}"), random_seq(&mut x, len))
    }))
    .unwrap();
    let bases = panel.total_len() as u64;
    for mode in [Mode::Local, Mode::SemiGlobal] {
        let mut scorer = GpuScorer::new(&panel, Scoring::default(), mode).expect("GpuScorer");
        eprintln!("# {} ({} mode)", scorer.device_name(), mode.name());
        let mut g = c.benchmark_group(format!("gpu_score_batch/{}", mode.name()));
        g.sample_size(20);
        for len in [100usize, 130, 200] {
            let reads: Vec<Vec<u8>> = (0..BATCH)
                .map(|_| encode(&random_seq(&mut x, len)))
                .collect();
            let queries: Vec<&[u8]> = reads.iter().map(Vec::as_slice).collect();
            g.throughput(Throughput::Elements((BATCH * len) as u64 * bases));
            for (name, half2) in [("int", false), ("half2", true)] {
                scorer.set_half2(half2);
                if half2 && !scorer.half2() {
                    continue; // pre-sm_80 device: no half2 kernel
                }
                g.bench_with_input(BenchmarkId::new(name, len), &queries, |b, q| {
                    b.iter(|| black_box(scorer.score_batch(black_box(q)).expect("score_batch")));
                });
            }
        }
        g.finish();
    }
}

criterion_group!(benches, gpu);
criterion_main!(benches);
