// SPDX-License-Identifier: MIT

//! The CUDA score kernel (feature `gpu`): every read of a batch against every
//! reference of the panel, one GPU thread per pair, scores only.
//!
//! This replaces [`crate::Aligner::score_all`] for a batch and nothing else.
//! The [`ScoreMatrix`] it returns goes to [`crate::Aligner::map_reads_scored`],
//! which picks the tie set, traces the winners with the CPU pairs kernels and
//! computes `MD`/`NM` exactly as the CPU path does — so the output can only
//! differ if a score does, and the scores are pinned to [`crate::scalar::score`]
//! by equality (`tests/gpu_parity.rs`).
//!
//! Same pattern as escapepod-rs's CUDA DTW kernel: the kernel is a source string
//! compiled by NVRTC when the scorer is built, cudarc loads the driver and
//! NVRTC with `dlopen` (its `dynamic-loading` feature), so nothing CUDA is
//! needed at build time, and a machine without either gets an error from
//! [`GpuScorer::new`] rather than a panic.
//!
//! # Two kernels
//!
//! The module compiles two kernels over the same DP and the scorer picks one
//! **per read**:
//!
//! * the **int kernel** (`align_score_kernel`): one reference per thread,
//!   `int` arithmetic, the same `i32::MIN / 4` sentinel as the scalar oracle.
//!   ALU-bound at ~15 INT32 ops a cell.
//! * the **half2 kernel** (`align_score_kernel_h2`): two references per
//!   thread in the halves of an `f16x2` register, the recurrence in
//!   `add.f16x2`/`max.f16x2`/`fma.rn.f16x2` — the FP16 pipe, which on sm_80
//!   is separate from the INT32 pipe — with a native `−∞`. Integers of
//!   magnitude ≤ 2048 are exact in fp16, so a read goes here only when
//!   [`fits_f16`] bounds its whole DP inside that; the output is the same
//!   integer either way, and both are equal to the oracle by test. It needs
//!   sm_80 or later (`max.f16x2`) and a panel whose longest reference is at
//!   most [`MAX_REF_LEN_H2`] (two references per thread, same shared memory).
//!   Elsewhere every read takes the int kernel.
//!
//! Both walk the read in 16-row stripes held in registers, and both take the
//! last stripe at the narrowest of 16, 8 or 4 rows that covers what is left
//! of the read, so a 130 nt read costs 132 rows rather than 144 (the int
//! kernel since #11, the half2 kernel since #13: 1.09× / 1.11× at 130 nt,
//! local / semi-global, on an A30). [`GpuScorer::registers`],
//! [`GpuScorer::registers_h2`] and the `local_bytes` pair report what the
//! JIT made of each kernel; `examples/gpu_score_probe.rs` prints them.
//!
//! # What goes to the GPU, and what does not
//!
//! A query is scored here when it is non-empty, at most [`MAX_READ_LEN`]
//! bases, and inside the same [`fits_i16`] bound the SIMD kernels use (which
//! the int kernel relies on for its packed `i16` boundary and result). Anything
//! else comes back as an unscored row and the caller's CPU kernel scores it,
//! in the same batch.
//!
//! The cap on read length is a *scheduling* bound, not a memory one: nothing
//! about a read is held in registers or shared memory beyond one 16-row
//! stripe, so a read of any length would fit. But one block owns a read for
//! its whole length, serially, and a batch finishes when its longest read
//! does; a real sample carries reads of 50–400 kb (0.5–6 s each on one block)
//! among ~10⁵ reads of ~130 nt. Past 4 kb — some 30× the median tRNA read —
//! a read costs one block longer than the whole rest of a 50 k-read batch is
//! likely to take, so it goes to the CPU instead, where `eschalign`
//! already gives such a read a chunk of its own.
//!
//! The cap on *reference* length is the shared-memory budget: the panel is
//! resident in shared memory at 4 bits a base, and a block of at least one
//! warp must fit in the 48 KB every CUDA device gives a block without opt-in —
//! 3,072 bases. A panel with a longer reference is refused by
//! [`GpuScorer::new`] (the tRNA panels this is built for are ~150 nt).

mod kernel;

use std::fmt;
use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
    sys::CUdevice_attribute,
};
use cudarc::nvrtc::{CompileError, CompileOptions, compile_ptx, compile_ptx_with_opts};

use crate::mapper::ScoreMatrix;
use crate::panel::Panel;
use crate::scoring::{Mode, Scoring};
use crate::simd::fits_i16;

use kernel::STRIPE_ROWS;

/// Longest read the GPU scores; longer ones come back unscored (see the
/// module docs for why this is a scheduling bound).
pub const MAX_READ_LEN: usize = 4096;

/// Longest reference the int kernel can hold in shared memory (see the
/// module docs): 48 KB for one warp at 4 bits a base.
pub const MAX_REF_LEN: usize = SMEM_BUDGET * 8 / 4 / 32;

/// Longest reference the half2 kernel can hold: one warp's threads carry two
/// references each, so half of [`MAX_REF_LEN`]. A panel between the two
/// scores on the int kernel alone.
pub const MAX_REF_LEN_H2: usize = MAX_REF_LEN / 2;

/// Largest magnitude fp16 holds every integer up to.
const F16_EXACT: i64 = 2048;

/// Shared memory a block may use without opting in, on every CUDA device.
const SMEM_BUDGET: usize = 48 * 1024;

/// Most threads per block.
const MAX_LANES: usize = 256;

/// Whether a read of `n` bases against references of at most `max_ref_len`
/// bases can be scored in fp16 exactly: every cell, and every intermediate
/// sum the recurrence forms, has magnitude at most
/// `min(n, L)·max(|match|, |mismatch|) + |open| + |extend| + |open|`, and
/// fp16 is exact up to 2048. The last `|open|` is a margin over the bound
/// [`fits_i16`] uses. A tRNA panel at the default scoring is ~420.
pub fn fits_f16(n: usize, max_ref_len: usize, scoring: &Scoring) -> bool {
    let k = n.min(max_ref_len) as i64;
    let m = scoring.match_score.abs().max(scoring.mismatch.abs()) as i64;
    let o = scoring.gap_open.abs() as i64;
    let e = scoring.gap_extend.abs() as i64;
    k * m + o + e + o <= F16_EXACT
}

/// Whether at least one CUDA device is visible to this process, cached for the
/// process. `false` when the driver library cannot be loaded at all.
///
/// Cached because the answer cannot change mid-process (SLURM does not hand a
/// job a GPU it did not allocate) and `cuDeviceGetCount` implies `cuInit`,
/// which is tens of milliseconds the first time.
pub fn device_visible() -> bool {
    static VISIBLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VISIBLE.get_or_init(|| {
        // Safety: only attempts to `dlopen` a shared library by name and
        // reports whether that succeeded; no pointers from it are dereferenced.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return false;
        }
        CudaContext::device_count().is_ok_and(|n| n > 0)
    })
}

/// Why the GPU scorer could not be built or run.
#[derive(Debug)]
pub enum GpuError {
    /// The CUDA driver or NVRTC library cannot be loaded on this machine.
    Unavailable(&'static str),
    /// A CUDA driver call failed: no device, out of memory, a launch failure.
    Driver(DriverError),
    /// NVRTC could not compile the kernel.
    Compile(CompileError),
    /// The panel has a reference the kernel cannot hold.
    ReferenceTooLong { len: usize, max: usize },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(what) => write!(f, "{what} cannot be loaded"),
            Self::Driver(e) => write!(f, "CUDA driver error: {e}"),
            Self::Compile(e) => write!(f, "CUDA kernel compilation failed: {e}"),
            Self::ReferenceTooLong { len, max } => write!(
                f,
                "the panel's longest reference ({len} nt) exceeds the GPU kernel's \
                 {max} nt (the panel must fit in shared memory)"
            ),
        }
    }
}

impl std::error::Error for GpuError {}

impl From<DriverError> for GpuError {
    fn from(e: DriverError) -> Self {
        Self::Driver(e)
    }
}

impl From<CompileError> for GpuError {
    fn from(e: CompileError) -> Self {
        Self::Compile(e)
    }
}

/// 4-bit mask of an [`crate::alphabet`] code: A/C/G/T one bit each, a
/// wildcard all four, so "masks intersect" is `Scoring::substitution`'s
/// match rule.
#[inline]
fn mask(code: u8) -> u32 {
    if code < 4 { 1 << code } else { 0xF }
}

/// The fp16 bit pattern of an integer of magnitude at most 2048 (exact).
fn half_bits(v: i32) -> u16 {
    debug_assert!((v as i64).abs() <= F16_EXACT, "{v} is not exact in fp16");
    if v == 0 {
        return 0;
    }
    let sign = if v < 0 { 0x8000 } else { 0 };
    let mag = v.unsigned_abs();
    let e = 31 - mag.leading_zeros();
    let mant = if e >= 10 {
        (mag >> (e - 10)) & 0x3FF
    } else {
        (mag << (10 - e)) & 0x3FF
    };
    sign | (((e + 15) as u16) << 10) | mant as u16
}

/// `v` in both halves of an `f16x2` word.
fn half2_bits(v: i32) -> u32 {
    let h = u32::from(half_bits(v));
    h | (h << 16)
}

/// One kernel's panel on the device and the launch geometry around it.
struct Layout {
    func: CudaFunction,
    panel_dev: CudaSlice<u32>,
    ref_len_dev: CudaSlice<i32>,
    ref_idx_dev: CudaSlice<i32>,
    /// Threads per block.
    lanes: usize,
    /// Reference groups (`gridDim.y`).
    n_groups: usize,
    /// Persistent blocks per group (`gridDim.x`).
    grid_x: usize,
    /// Panel words per lane, and the shared memory the block copies.
    words_per_lane: usize,
    smem: usize,
    /// Scratch words per (reference column, launched thread).
    scratch_words: usize,
}

impl Layout {
    /// Lay `panel` out for a kernel holding `per_thread` references per
    /// thread, at `cols_per_word` reference columns per panel word, or
    /// `None` when not even a one-warp block fits the shared-memory budget.
    fn new(
        ctx: &CudaContext,
        stream: &Arc<CudaStream>,
        func: CudaFunction,
        panel: &Panel,
        order: &[usize],
        per_thread: usize,
        scratch_words: usize,
    ) -> Result<Option<Self>, GpuError> {
        let cols_per_word = 8 / per_thread;
        let max_ref_len = panel.max_len();
        let words_per_lane = max_ref_len.div_ceil(cols_per_word);
        // One group if the panel fits a block, else as wide as shared memory allows.
        let mut lanes = panel
            .len()
            .div_ceil(per_thread)
            .next_multiple_of(32)
            .min(MAX_LANES);
        while lanes > 32 && words_per_lane * 4 * lanes > SMEM_BUDGET {
            lanes -= 32;
        }
        let smem = words_per_lane * 4 * lanes;
        if smem > SMEM_BUDGET {
            return Ok(None);
        }
        let per_block = lanes * per_thread;
        let n_groups = order.len().div_ceil(per_block);
        let mut words = vec![0u32; n_groups * words_per_lane * lanes];
        let mut ref_len = vec![0i32; n_groups * per_block];
        let mut ref_idx = vec![-1i32; n_groups * per_block];
        for (g, chunk) in order.chunks(per_block).enumerate() {
            let base = g * words_per_lane * lanes;
            for (slot, &ri) in chunk.iter().enumerate() {
                let (lane, h) = (slot / per_thread, slot % per_thread);
                let codes = &panel.get(ri).codes;
                ref_len[g * per_block + slot] = codes.len() as i32;
                ref_idx[g * per_block + slot] = ri as i32;
                for (j, &c) in codes.iter().enumerate() {
                    let shift = 4 * (j % cols_per_word) + 16 * h;
                    words[base + (j / cols_per_word) * lanes + lane] |= mask(c) << shift;
                }
            }
        }
        let per_sm = func
            .occupancy_max_active_blocks_per_multiprocessor(lanes as u32, smem, None)?
            .max(1) as usize;
        let sms = ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?
            .max(1) as usize;
        // Enough persistent blocks to fill every SM once, spread over the groups.
        let grid_x = (sms * per_sm).div_ceil(n_groups).max(1);
        Ok(Some(Self {
            func,
            panel_dev: stream.clone_htod(&words)?,
            ref_len_dev: stream.clone_htod(&ref_len)?,
            ref_idx_dev: stream.clone_htod(&ref_idx)?,
            lanes,
            n_groups,
            grid_x,
            words_per_lane,
            smem,
            scratch_words,
        }))
    }

    /// Threads one launch runs.
    fn threads(&self) -> usize {
        self.grid_x * self.n_groups * self.lanes
    }

    fn config(&self) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (self.grid_x as u32, self.n_groups as u32, 1),
            block_dim: (self.lanes as u32, 1, 1),
            shared_mem_bytes: self.smem as u32,
        }
    }

    fn geometry(&self) -> (usize, usize, usize) {
        (self.lanes, self.n_groups, self.grid_x)
    }
}

/// A panel resident on the device under one scoring and one mode, ready to
/// score batches of reads.
///
/// Build once per run: construction compiles the kernels (NVRTC) and uploads
/// the panel. [`GpuScorer::score_batch`] takes `&mut self` because every
/// launch reuses one scratch buffer; `eschalign` gives the scorer a
/// dedicated thread.
pub struct GpuScorer {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    int: Layout,
    /// The half2 kernel, where the device and the panel allow it.
    h2: Option<Layout>,
    /// Whether reads that fit it go to the half2 kernel.
    use_h2: bool,
    scratch_dev: CudaSlice<u32>,
    n_refs: usize,
    max_ref_len: usize,
    scoring: Scoring,
    mode: Mode,
}

impl GpuScorer {
    /// Prepare `panel` on CUDA device 0.
    pub fn new(panel: &Panel, scoring: Scoring, mode: Mode) -> Result<Self, GpuError> {
        Self::new_on_device(0, panel, scoring, mode)
    }

    /// Prepare `panel` on CUDA device `ordinal`.
    pub fn new_on_device(
        ordinal: usize,
        panel: &Panel,
        scoring: Scoring,
        mode: Mode,
    ) -> Result<Self, GpuError> {
        // cudarc's dlopen wrappers panic when a library is missing, and the
        // release profile aborts on panic; ask first so the caller gets an
        // error it can report (or fall back from).
        // SAFETY: both only try to dlopen a library and report whether that
        // worked; no pointer from them is dereferenced.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(GpuError::Unavailable("the CUDA driver library (libcuda)"));
        }
        if !unsafe { cudarc::nvrtc::sys::is_culib_present() } {
            return Err(GpuError::Unavailable("the CUDA NVRTC library (libnvrtc)"));
        }
        let max_ref_len = panel.max_len();
        if max_ref_len > MAX_REF_LEN {
            return Err(GpuError::ReferenceTooLong {
                len: max_ref_len,
                max: MAX_REF_LEN,
            });
        }

        // Longest first, stable, as the SIMD profile does: a warp's threads
        // then have references of similar length and finish together — and
        // the half2 kernel's two references per thread are neighbours in
        // this order, so a thread's padding past the shorter one is small.
        let mut order: Vec<usize> = (0..panel.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(panel.get(i).codes.len()));

        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        // The half2 kernel needs sm_80 (`max.f16x2`); the int kernel runs on
        // anything, at NVRTC's default target.
        let sm80 =
            ctx.attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)? >= 8;
        let module = if sm80 {
            let src = [kernel::KERNEL_SRC, kernel::KERNEL_H2_SRC].concat();
            let opts = CompileOptions {
                arch: Some("compute_80"),
                ..Default::default()
            };
            ctx.load_module(compile_ptx_with_opts(src, opts)?)?
        } else {
            ctx.load_module(compile_ptx(kernel::KERNEL_SRC)?)?
        };
        let int = Layout::new(
            &ctx,
            &stream,
            module.load_function(kernel::KERNEL_NAME)?,
            panel,
            &order,
            1,
            1,
        )?
        .expect("a reference of at most MAX_REF_LEN fits one warp's shared memory");
        let h2 = if sm80 {
            Layout::new(
                &ctx,
                &stream,
                module.load_function(kernel::KERNEL_H2_NAME)?,
                panel,
                &order,
                2,
                2,
            )?
        } else {
            None
        };
        debug_assert_eq!(h2.is_some(), sm80 && max_ref_len <= MAX_REF_LEN_H2);

        let scratch = max_ref_len.max(1)
            * h2.iter()
                .chain(std::iter::once(&int))
                .map(|l| l.threads() * l.scratch_words)
                .max()
                .unwrap_or(1);
        let scratch_dev = stream.alloc_zeros::<u32>(scratch)?;

        Ok(Self {
            ctx,
            stream,
            int,
            h2,
            use_h2: true,
            scratch_dev,
            n_refs: panel.len(),
            max_ref_len,
            scoring,
            mode,
        })
    }

    /// The device's name, for the log.
    pub fn device_name(&self) -> String {
        self.ctx
            .name()
            .unwrap_or_else(|_| "unknown CUDA device".into())
    }

    /// How the int kernel was laid out, for the log: `(threads per block,
    /// reference groups, persistent blocks per group)`.
    pub fn geometry(&self) -> (usize, usize, usize) {
        self.int.geometry()
    }

    /// The half2 kernel's layout, the same way, or `None` where the device
    /// or the panel rules that kernel out.
    pub fn geometry_h2(&self) -> Option<(usize, usize, usize)> {
        self.h2.as_ref().map(Layout::geometry)
    }

    /// Registers per thread the driver compiled the int kernel to — what
    /// bounds its occupancy (the shared-memory panel is a few KB per block).
    pub fn registers(&self) -> Option<i32> {
        self.int.func.num_regs().ok()
    }

    /// Registers per thread of the half2 kernel, where there is one.
    pub fn registers_h2(&self) -> Option<i32> {
        self.h2.as_ref().and_then(|l| l.func.num_regs().ok())
    }

    /// Local memory per thread the driver compiled the int kernel to. The
    /// kernel keeps everything in registers, so a nonzero value is spills
    /// (or a stripe array the JIT demoted) — what a tighter launch bound
    /// costs, and what the register count alone does not show.
    pub fn local_bytes(&self) -> Option<i32> {
        self.int.func.local_size_bytes().ok()
    }

    /// Local memory per thread of the half2 kernel, where there is one.
    pub fn local_bytes_h2(&self) -> Option<i32> {
        self.h2
            .as_ref()
            .and_then(|l| l.func.local_size_bytes().ok())
    }

    /// Whether the half2 kernel is in use: available on this device for this
    /// panel, and not switched off by [`GpuScorer::set_half2`].
    pub fn half2(&self) -> bool {
        self.h2.is_some() && self.use_h2
    }

    /// Send every read to the int kernel (`false`), or let those inside
    /// [`fits_f16`] take the half2 kernel where there is one (`true`, the
    /// default). The A/B lever for `examples/gpu_score_probe.rs` and the
    /// parity test; the result is the same either way.
    pub fn set_half2(&mut self, enabled: bool) {
        self.use_h2 = enabled;
    }

    /// Whether [`GpuScorer::score_batch`] scores a query of `len` bases (the
    /// rest come back unscored, for the CPU).
    pub fn scores_len(&self, len: usize) -> bool {
        len > 0 && len <= MAX_READ_LEN && fits_i16(len, self.max_ref_len, &self.scoring)
    }

    /// Whether a query of `len` bases goes to the half2 kernel (else the int
    /// kernel) when [`GpuScorer::scores_len`] says it is scored at all.
    pub fn half2_scores_len(&self, len: usize) -> bool {
        self.half2() && fits_f16(len, self.max_ref_len, &self.scoring)
    }

    /// Score every query (codes from [`crate::alphabet`]) against every
    /// reference: row `k` of the result is query `k`, in panel order, or
    /// unscored when [`GpuScorer::scores_len`] says no.
    ///
    /// [`GpuScorer::prepare`], [`GpuScorer::submit`] and
    /// [`GpuScorer::finish`] in one call. A caller with a stream of batches
    /// does better with the three apart: prepare batch *k + 1* while batch
    /// *k*'s kernel runs, so the device is not idle while the host packs.
    pub fn score_batch(&mut self, queries: &[&[u8]]) -> Result<ScoreMatrix, GpuError> {
        let prepared = self.prepare(queries);
        let pending = self.submit(prepared)?;
        self.finish(pending)
    }

    /// The host half of a batch: decide what the GPU scores, on which
    /// kernel, and pack it. Touches no device state, so it can overlap a
    /// running kernel.
    pub fn prepare(&self, queries: &[&[u8]]) -> PreparedBatch {
        let n = queries.len();
        let scored: Vec<bool> = queries.iter().map(|q| self.scores_len(q.len())).collect();
        let (mut work_h2, mut work): (Vec<i32>, Vec<i32>) = (0..n as i32)
            .filter(|&k| scored[k as usize])
            .partition(|&k| self.half2_scores_len(queries[k as usize].len()));
        // Longest first: a long read starts early rather than finishing last.
        work.sort_by_key(|&k| std::cmp::Reverse(queries[k as usize].len()));
        work_h2.sort_by_key(|&k| std::cmp::Reverse(queries[k as usize].len()));

        // Every query gets whole stripes (`STRIPE_ROWS` codes, `STRIPE_ROWS / 8`
        // words), zero-padded: code 0 matches nothing, and the kernel masks
        // the rows anyway. Unscored queries get an empty slot.
        let stripe_words = STRIPE_ROWS / 8;
        let mut qoff = vec![0u64; n];
        let mut qlen = vec![0i32; n];
        let total_words: usize = queries
            .iter()
            .zip(&scored)
            .filter(|(_, s)| **s)
            .map(|(q, _)| q.len().div_ceil(STRIPE_ROWS) * stripe_words)
            .sum();
        let mut qwords = vec![0u32; total_words.max(1)];
        let mut at = 0usize;
        for (k, q) in queries.iter().enumerate() {
            if !scored[k] {
                continue;
            }
            qoff[k] = at as u64;
            qlen[k] = q.len() as i32;
            for (i, &c) in q.iter().enumerate() {
                qwords[at + i / 8] |= mask(c) << (4 * (i % 8));
            }
            at += q.len().div_ceil(STRIPE_ROWS) * stripe_words;
        }
        PreparedBatch {
            scored,
            work,
            work_h2,
            qwords,
            qoff,
            qlen,
        }
    }

    /// Upload a prepared batch and queue its kernel(s); returns without
    /// waiting. Launches are ordered on the scorer's one stream (they share
    /// its scratch), so a second `submit` queues behind the first.
    pub fn submit(&mut self, batch: PreparedBatch) -> Result<PendingBatch, GpuError> {
        let n = batch.scored.len();
        let s = &self.stream;
        let mut out_dev = s.alloc_zeros::<i16>((n * self.n_refs).max(1))?;
        if batch.work.is_empty() && batch.work_h2.is_empty() {
            return Ok(PendingBatch {
                out_dev,
                scored: batch.scored,
                inputs: None,
            });
        }
        let qwords_dev = s.clone_htod(&batch.qwords)?;
        let qoff_dev = s.clone_htod(&batch.qoff)?;
        let qlen_dev = s.clone_htod(&batch.qlen)?;
        let mut launches = Vec::with_capacity(2);

        let n_refs = self.n_refs as i32;
        if !batch.work.is_empty() {
            let l = &self.int;
            let order_dev = s.clone_htod(&batch.work)?;
            let mut counters_dev = s.alloc_zeros::<i32>(l.n_groups)?;
            let n_work = batch.work.len() as i32;
            let words_per_lane = l.words_per_lane as i32;
            let (ma, mi, o, e) = (
                self.scoring.match_score,
                self.scoring.mismatch,
                self.scoring.gap_open,
                self.scoring.gap_extend,
            );
            let local = i32::from(self.mode == Mode::Local);
            let mut b = s.launch_builder(&l.func);
            b.arg(&qwords_dev)
                .arg(&qoff_dev)
                .arg(&qlen_dev)
                .arg(&order_dev)
                .arg(&l.panel_dev)
                .arg(&l.ref_len_dev)
                .arg(&l.ref_idx_dev)
                .arg(&mut out_dev)
                .arg(&mut self.scratch_dev)
                .arg(&mut counters_dev)
                .arg(&n_work)
                .arg(&n_refs)
                .arg(&words_per_lane)
                .arg(&ma)
                .arg(&mi)
                .arg(&o)
                .arg(&e)
                .arg(&local);
            // SAFETY: the argument list matches `align_score_kernel`'s
            // signature in order and type; every buffer is sized for the
            // indices the kernel forms from these same arguments (`qoff`/`qlen`
            // per query, `order` `n_work` long, the panel `n_groups *
            // words_per_lane * lanes`, the scratch `max_ref_len` words per
            // launched thread, `out` a row per query), and the shared memory is
            // the panel slice it copies. The input buffers live in the returned
            // `PendingBatch` until `finish`.
            unsafe { b.launch(l.config()) }?;
            launches.push(LaunchInputs {
                _order: order_dev,
                _counters: counters_dev,
            });
        }
        if !batch.work_h2.is_empty() {
            let l = self
                .h2
                .as_ref()
                .expect("prepare sends reads to the half2 kernel only where there is one");
            let order_dev = s.clone_htod(&batch.work_h2)?;
            let mut counters_dev = s.alloc_zeros::<i32>(l.n_groups)?;
            let n_work = batch.work_h2.len() as i32;
            let words_per_lane = l.words_per_lane as i32;
            let mi2 = half2_bits(self.scoring.mismatch);
            let d2 = half2_bits(self.scoring.match_score - self.scoring.mismatch);
            let o2 = half2_bits(self.scoring.gap_open);
            let e2 = half2_bits(self.scoring.gap_extend);
            let local = i32::from(self.mode == Mode::Local);
            let mut b = s.launch_builder(&l.func);
            b.arg(&qwords_dev)
                .arg(&qoff_dev)
                .arg(&qlen_dev)
                .arg(&order_dev)
                .arg(&l.panel_dev)
                .arg(&l.ref_len_dev)
                .arg(&l.ref_idx_dev)
                .arg(&mut out_dev)
                .arg(&mut self.scratch_dev)
                .arg(&mut counters_dev)
                .arg(&n_work)
                .arg(&n_refs)
                .arg(&words_per_lane)
                .arg(&mi2)
                .arg(&d2)
                .arg(&o2)
                .arg(&e2)
                .arg(&local);
            // SAFETY: as above, against `align_score_kernel_h2`'s signature:
            // the panel, `ref_len` and `ref_idx` are this layout's (two
            // references per thread), the scratch holds `max_ref_len` uint2s
            // per launched thread of the larger layout, and the two launches
            // are ordered on one stream so they never share it at once. Each
            // launch writes only its own queries' rows of `out`.
            unsafe { b.launch(l.config()) }?;
            launches.push(LaunchInputs {
                _order: order_dev,
                _counters: counters_dev,
            });
        }
        Ok(PendingBatch {
            out_dev,
            scored: batch.scored,
            inputs: Some(DeviceInputs {
                _qwords: qwords_dev,
                _qoff: qoff_dev,
                _qlen: qlen_dev,
                _launches: launches,
            }),
        })
    }

    /// Wait for a submitted batch's kernel(s) and bring its scores back.
    pub fn finish(&self, pending: PendingBatch) -> Result<ScoreMatrix, GpuError> {
        let n = pending.scored.len();
        let mut scores = self.stream.clone_dtoh(&pending.out_dev)?;
        scores.truncate(n * self.n_refs);
        drop(pending.inputs);
        Ok(ScoreMatrix::from_parts(self.n_refs, scores, pending.scored))
    }
}

/// A batch packed for the device by [`GpuScorer::prepare`].
pub struct PreparedBatch {
    scored: Vec<bool>,
    /// Query indices for the int kernel, longest first.
    work: Vec<i32>,
    /// Query indices for the half2 kernel, longest first.
    work_h2: Vec<i32>,
    qwords: Vec<u32>,
    qoff: Vec<u64>,
    qlen: Vec<i32>,
}

impl PreparedBatch {
    /// How many queries each kernel will score: `(int, half2)`.
    pub fn work_split(&self) -> (usize, usize) {
        (self.work.len(), self.work_h2.len())
    }
}

/// A batch whose kernel has been queued by [`GpuScorer::submit`]; hand it to
/// [`GpuScorer::finish`].
pub struct PendingBatch {
    out_dev: CudaSlice<i16>,
    scored: Vec<bool>,
    /// Kept alive until the kernel that reads them has finished.
    inputs: Option<DeviceInputs>,
}

struct DeviceInputs {
    _qwords: CudaSlice<u32>,
    _qoff: CudaSlice<u64>,
    _qlen: CudaSlice<i32>,
    _launches: Vec<LaunchInputs>,
}

struct LaunchInputs {
    _order: CudaSlice<i32>,
    _counters: CudaSlice<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fp16 bits back to a value, by the format's definition.
    fn half_value(bits: u16) -> f32 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let e = ((bits >> 10) & 0x1F) as i32;
        let m = (bits & 0x3FF) as f32;
        if e == 0 {
            sign * m * 2f32.powi(-24)
        } else {
            sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
        }
    }

    #[test]
    fn half_bits_are_exact_up_to_2048() {
        assert_eq!(half_bits(0), 0x0000);
        assert_eq!(half_bits(1), 0x3C00);
        assert_eq!(half_bits(-1), 0xBC00);
        assert_eq!(half_bits(2), 0x4000);
        assert_eq!(half_bits(-10), 0xC900);
        assert_eq!(half_bits(2048), 0x6800);
        for v in -2048..=2048 {
            assert_eq!(half_value(half_bits(v)), v as f32, "{v}");
        }
        assert_eq!(half2_bits(-1), 0xBC00_BC00);
    }

    #[test]
    fn fits_f16_bound() {
        let s = Scoring::default(); // 2,-1,-10,-1: k*2 + 21
        assert!(fits_f16(130, 200, &s));
        assert!(fits_f16(4096, 200, &s)); // min(n, L) = 200 → 421
        assert!(fits_f16(1013, 4096, &s)); // 2026 + 21 = 2047
        assert!(!fits_f16(1014, 4096, &s)); // 2028 + 21 = 2049
        let big = Scoring::new(1000, -1000, -1000, -1000).unwrap();
        assert!(!fits_f16(1, 10, &big)); // the gap terms alone are 3000
        let s = Scoring::new(3, -2, -5, -2).unwrap(); // k*3 + 12
        assert!(fits_f16(678, 1000, &s));
        assert!(!fits_f16(679, 1000, &s));
    }
}
