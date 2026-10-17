// SPDX-License-Identifier: MIT

//! Time the CUDA score kernel alone, on real reads, without BAM I/O or
//! tracebacks: the measurement to re-run when the kernel changes.
//!
//! ```text
//! samtools view reads.bam | cut -f10 > reads.seq      # one sequence per line
//! cargo run --release --features gpu -p eschalign --example gpu_score_probe -- \
//!     panel.fa reads.seq [batch_size] [local|semiglobal] [auto|int]
//! ```
//!
//! Prints the kernels' geometry and registers, then the total seconds in
//! `score_batch` (host packing, transfers and the kernel) and cells per
//! second (`read length × panel bases`, GPU-scored reads only), after one
//! untimed warm-up batch. `auto` (the default) lets reads inside `fits_f16`
//! take the half2 kernel where the device has one; `int` sends every read to
//! the int kernel — the A/B of the two.

use std::io::BufRead;
use std::time::Instant;

use eschalign::alphabet::encode;
use eschalign::cuda::GpuScorer;
use eschalign::{Mode, Panel, Scoring};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(
        args.len() >= 3,
        "usage: gpu_score_probe PANEL.fa READS.seq [BATCH] [MODE] [auto|int]"
    );
    let batch: usize = args.get(3).map_or(50_000, |s| s.parse().unwrap());
    let mode: Mode = args.get(4).map_or(Mode::Local, |s| s.parse().unwrap());
    let half2 = match args.get(5).map(String::as_str) {
        None | Some("auto") => true,
        Some("int") => false,
        Some(other) => panic!("kernel must be auto or int, got {other}"),
    };

    let file = std::fs::File::open(&args[1]).unwrap();
    let refs = eschalign::fasta::read_fasta(std::io::BufReader::new(file)).unwrap();
    let panel = Panel::new(refs).unwrap();
    let reads: Vec<Vec<u8>> = std::io::BufReader::new(std::fs::File::open(&args[2]).unwrap())
        .lines()
        .map(|l| encode(l.unwrap().trim().as_bytes()))
        .collect();

    let t0 = Instant::now();
    let mut gpu = GpuScorer::new(&panel, Scoring::default(), mode).unwrap();
    gpu.set_half2(half2);
    let (lanes, groups, blocks) = gpu.geometry();
    println!(
        "{}: int kernel {groups} group(s) x {blocks} blocks x {lanes} threads, {:?} registers, ready in {:.2} s",
        gpu.device_name(),
        gpu.registers(),
        t0.elapsed().as_secs_f64()
    );
    match gpu.geometry_h2() {
        Some((lanes, groups, blocks)) => println!(
            "half2 kernel {groups} group(s) x {blocks} blocks x {lanes} threads, {:?} registers, {}",
            gpu.registers_h2(),
            if gpu.half2() { "in use" } else { "off" }
        ),
        None => println!("half2 kernel: not available on this device for this panel"),
    }
    let chunks: Vec<Vec<&[u8]>> = reads
        .chunks(batch)
        .map(|c| c.iter().map(Vec::as_slice).collect())
        .collect();
    gpu.score_batch(&chunks[0]).unwrap(); // warm-up: JIT, first allocations

    let (mut secs, mut cells, mut scored, mut on_h2) = (0.0, 0f64, 0usize, 0usize);
    let panel_bases = panel.total_len() as f64;
    for c in &chunks {
        let t = Instant::now();
        let m = gpu.score_batch(c).unwrap();
        secs += t.elapsed().as_secs_f64();
        for (k, q) in c.iter().enumerate() {
            if m.row(k).is_some() {
                cells += q.len() as f64 * panel_bases;
                scored += 1;
                on_h2 += usize::from(gpu.half2_scores_len(q.len()));
            }
        }
    }
    println!(
        "{} reads ({scored} on the GPU, {on_h2} on the half2 kernel) in {} batches: {secs:.2} s, {:.3e} cells/s ({})",
        reads.len(),
        chunks.len(),
        cells / secs,
        mode.name()
    );
}
