// SPDX-License-Identifier: MIT

//! CUDA source of the score kernel, compiled by NVRTC at run time (so the
//! build needs no `nvcc` and no CUDA toolkit).
//!
//! # Layout
//!
//! * **One thread per (read, reference) pair.** A block holds one *group* of
//!   references, one per thread (`blockDim.x` = lanes, a multiple of 32), and
//!   the grid is `(persistent blocks, groups)`. Every thread of a block works
//!   on the same read at the same time, so the read's codes are uniform loads
//!   and the per-read loop bounds are uniform; each thread walks its *own*
//!   reference to its own length, so there is no column mask at all — a
//!   thread with a shorter reference simply finishes its column loop first.
//! * **Persistent blocks pull reads from an atomic counter**, in the order the
//!   host gives (longest read first), so a long read starts early instead of
//!   being the tail the whole batch waits behind.
//! * **The panel is resident in shared memory**, loaded once per block: four
//!   bits per reference base (A=1, C=2, G=4, T=8, anything else 0xF), eight
//!   bases per word, transposed so word `w` of lane `t` is `sp[w * lanes + t]`
//!   — consecutive threads, consecutive banks. A read base scores a match iff
//!   its mask and the reference's intersect, which is exactly the wildcard
//!   rule of `Scoring::substitution`.
//! * **The read is walked in stripes of `R` rows held in registers.** For each
//!   reference column the thread updates `R` cells (the scalar recurrence,
//!   verbatim, in `int`), and the only state that leaves registers is the
//!   stripe's bottom row — `H` and `F` of the last row, two `i16`s packed in
//!   one word per column — which the next stripe reads back as its top border.
//!   That boundary lives in a global scratch buffer interleaved by thread
//!   (`scratch[col * total_threads + thread]`, so a warp's accesses coalesce):
//!   one 4-byte load and one store per column per stripe, 0.5 B of traffic per
//!   cell at `R = 16`. The last stripe is instantiated at the narrowest of
//!   16, 8 or 4 rows that covers what is left of the read (`stripe<…, RR>`,
//!   `rows ≤ 4 → 4`, `≤ 8 → 8`), so a 130 nt read pays for 132 rows rather
//!   than 144 — 1.09× at that length on an A30.
//!
//! * **`__launch_bounds__(256, 4)`** caps the kernel at 64 registers, so four
//!   blocks of up to 256 threads fit an SM. Left to itself the JIT took 88
//!   (three 192-thread blocks on an A30); at 64 the stripe state still lives in
//!   registers and M1's 0.49 M reads score in 5.9 s instead of 6.4 s
//!   (`examples/gpu_score_probe.rs`). Tighter (56, 40) spills and loses.
//!
//! The values are exactly the scalar oracle's: the arithmetic is `int` with
//! the same `NEG = i32::MIN / 4` sentinel, and the only narrowing — the packed
//! boundary and the `i16` result — is of values `fits_i16` has bounded.
//!
//! # Semi-global
//!
//! The best cell of the last row is taken column by column in the stripe that
//! holds row `n`; the best cell of the reference's last column is what the
//! stripe's `H` registers hold when the thread's column loop ends, taken per
//! stripe. Local mode takes every valid cell and floors at zero. Rows past the
//! read's end (the last stripe's padding, code 0: matches nothing) are masked
//! out of both, and cannot reach a valid cell because information only flows
//! to larger row indices.

/// Rows per stripe. Changing it changes the query packing (`R / 8` words per
/// stripe) on the host side too.
pub(crate) const STRIPE_ROWS: usize = 16;

pub(crate) const KERNEL_NAME: &str = "align_score_kernel";

pub(crate) const KERNEL_SRC: &str = r#"
#define NEG (-(1 << 29))
#define R 16

template <bool LOCAL, bool TAIL, int RR>
__device__ __forceinline__ void stripe(
    const unsigned int* __restrict__ sp, int lanes, int t, int m,
    unsigned int q0, unsigned int q1, int rows, bool first,
    unsigned int* __restrict__ bnd, size_t bstride,
    int ma, int mi, int o, int e, int& best)
{
    int qm[RR];
#pragma unroll
    for (int r = 0; r < RR; r++) {
        qm[r] = ((r < 8 ? (q0 >> (4 * r)) : (q1 >> (4 * (r - 8)))) & 0xF);
    }
    int H[RR];
    int E[RR];
#pragma unroll
    for (int r = 0; r < RR; r++) {
        H[r] = 0;
        E[r] = NEG;
    }
    int diag_top = 0;
    unsigned int rw = 0;
    unsigned int bnext = first ? 0u : bnd[0];
    for (int j = 0; j < m; j++) {
        if ((j & 7) == 0) rw = sp[(j >> 3) * lanes + t];
        const int rm = (rw >> ((j & 7) * 4)) & 0xF;
        int hup;
        int f;
        if (first) {
            hup = 0;
            f = NEG;
        } else {
            const unsigned int b = bnext;
            if (j + 1 < m) bnext = bnd[(size_t)(j + 1) * bstride];
            hup = (int)(short)(b & 0xFFFFu);
            f = (int)(short)(b >> 16);
        }
        int hdiag = diag_top;
        diag_top = hup;
        int last = NEG;
#pragma unroll
        for (int r = 0; r < RR; r++) {
            const int hleft = H[r];
            const int ev = max(hleft + o, E[r] + e);
            E[r] = ev;
            f = max(hup + o, f + e);
            const int s = (qm[r] & rm) ? ma : mi;
            int h = max(max(hdiag + s, ev), f);
            if (LOCAL) h = max(h, 0);
            hdiag = hleft;
            H[r] = h;
            hup = h;
            if (LOCAL) {
                if (!TAIL || r < rows) best = max(best, h);
            } else if (TAIL) {
                if (r == rows - 1) last = h;
            }
        }
        if (!TAIL) {
            bnd[(size_t)j * bstride] =
                ((unsigned int)(unsigned short)hup) | (((unsigned int)(unsigned short)f) << 16);
        }
        if (!LOCAL && TAIL) best = max(best, last);
    }
    if (!LOCAL) {
#pragma unroll
        for (int r = 0; r < RR; r++) {
            if (!TAIL || r < rows) best = max(best, H[r]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 4) align_score_kernel(
    const unsigned int* __restrict__ qwords,
    const unsigned long long* __restrict__ qoff,
    const int* __restrict__ qlen,
    const int* __restrict__ order,
    const unsigned int* __restrict__ panel,
    const int* __restrict__ ref_len,
    const int* __restrict__ ref_idx,
    short* __restrict__ out,
    unsigned int* __restrict__ scratch,
    int* __restrict__ counters,
    int n_work,
    int n_refs,
    int words_per_lane,
    int ma, int mi, int o, int e,
    int local)
{
    extern __shared__ unsigned int sp[];
    __shared__ int next;
    const int lanes = blockDim.x;
    const int t = threadIdx.x;
    const int g = blockIdx.y;
    const unsigned int* gp = panel + (size_t)g * words_per_lane * lanes;
    for (int w = t; w < words_per_lane * lanes; w += lanes) sp[w] = gp[w];
    const int m = ref_len[g * lanes + t];
    const int ri = ref_idx[g * lanes + t];
    const size_t total = (size_t)gridDim.x * gridDim.y * lanes;
    const size_t gtid = ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * lanes + t;
    unsigned int* bnd = scratch + gtid;
    for (;;) {
        // First pass: the panel is loaded. Later passes: everyone has read
        // `next` before thread 0 overwrites it.
        __syncthreads();
        if (t == 0) next = atomicAdd(&counters[g], 1);
        __syncthreads();
        const int k = next;
        if (k >= n_work) break;
        const int qi = order[k];
        const int n = qlen[qi];
        const unsigned int* q = qwords + qoff[qi];
        int best = local ? 0 : NEG;
        if (m > 0) {
            for (int i0 = 0; i0 < n; i0 += R) {
                const unsigned int q0 = q[i0 >> 3];
                const unsigned int q1 = q[(i0 >> 3) + 1];
                const int rows = min(R, n - i0);
                const bool first = i0 == 0;
                const bool tail = i0 + R >= n;
#define ARGS sp, lanes, t, m, q0, q1, rows, first, bnd, total, ma, mi, o, e, best
                if (local) {
                    if (!tail) stripe<true, false, 16>(ARGS);
                    else if (rows <= 4) stripe<true, true, 4>(ARGS);
                    else if (rows <= 8) stripe<true, true, 8>(ARGS);
                    else stripe<true, true, 16>(ARGS);
                } else {
                    if (!tail) stripe<false, false, 16>(ARGS);
                    else if (rows <= 4) stripe<false, true, 4>(ARGS);
                    else if (rows <= 8) stripe<false, true, 8>(ARGS);
                    else stripe<false, true, 16>(ARGS);
                }
#undef ARGS
            }
            out[(size_t)qi * n_refs + ri] = (short)best;
        }
    }
}
"#;

pub(crate) const KERNEL_H2_NAME: &str = "align_score_kernel_h2";

/// The half2 (fp16x2) score kernel: the same DP, **two references per
/// thread**, one in each half of a packed `f16x2` register. Appended to
/// [`KERNEL_SRC`] in the one module the scorer compiles; needs `compute_80`
/// (`max.f16x2` is native from sm_80, where `HADD2`/`HMNMX2`/`HFMA2` run on
/// the FP16 pipe — not the INT32 pipe the int kernel saturates), so it is
/// compiled out below that.
///
/// # Why it is exact
///
/// Every value in the DP is an integer, and fp16 holds integers of magnitude
/// ≤ 2048 exactly; the host sends a read here only when
/// [`super::fits_f16`] bounds every cell and every intermediate sum of its
/// DP inside that. `−∞` is the native `0xFC00`: `−∞ + e = −∞`,
/// `max(−∞, x) = x`, and no `+∞` ever arises (values are bounded above by
/// the real best), so there is no NaN path.
///
/// # What differs from the int kernel
///
/// * **Panel words hold four columns of both references**: nibbles 0–3 of
///   the low half are reference A's bases `4w..4w+3`, the high half is
///   reference B's, so `(word >> 4k) & 0x000F000F` is one column's pair of
///   masks in half2 lanes, and the read's mask replicated into both halves
///   ANDs against it in one op. The nibbles sit at bits 10–13 of each half
///   (exponent field: a nonzero nibble is a *normal* fp16 value), and
///   `set.ne.f16x2` turns "nonzero" into 1.0/0.0 per half — the match flag —
///   which one `fma` maps to `ma`/`mi`. The int kernel's AND + compare +
///   select for one reference becomes AND + set + fma for two.
/// * **The two references can differ in length.** The thread walks
///   `max(mA, mB)` columns; the shorter reference's half computes padding
///   past its end whose cells never reach a real cell (information flows to
///   larger columns only) and never reach the result: each column's
///   contribution to the running best is masked to `−∞` in a half whose
///   reference has ended (`mask_ended`), and semi-global takes a half's last
///   column where *its* reference ends (`j + 1 == mA`, `j + 1 == mB`), not
///   at the loop's end.
/// * **The stripe boundary is a `uint2`** per column: `H` of the bottom row
///   as one half2 and `F` as the other, 8 B per column per thread — the same
///   4 B per reference the int kernel writes.
/// * **`__launch_bounds__(256, 3)`**, 80 registers, not the int kernel's
///   `(256, 4)`: it carries a `uint2` boundary, two reference lengths and the
///   column accumulator on top of the same 48 stripe registers, and at 64 it
///   spills — on an A30 the 100–200 nt reads of M1 score at 3.4e11 cells/s
///   capped at 64, 4.6e11 at 96 (six blocks of 96 threads an SM), 4.5e11 at
///   72 (spills again) and **5.3e11 at 80** (eight blocks), local mode;
///   semi-global 3.5 / 5.6 / 4.1 / 5.8e11 (`examples/gpu_score_probe.rs`).
/// * **The last stripe is as narrow as the read's remaining rows allow**, as
///   in the int kernel: `stripe2<…, RR>` with `RR ∈ {16, 8, 4}` (`rows ≤ 4
///   → 4`, `≤ 8 → 8`), so a 130 nt read computes 132 rows, not 144. On an
///   A30 that is 1.09× at 130 nt in local mode and 1.11× in semi-global
///   (5.07 → 5.51e11 and 5.55 → 6.16e11 cells/s), 1.04× / 1.07× over all of
///   M1, with the kernel still at 80 registers and 72 B of local memory a
///   thread (56 B before; the int kernel holds 120 B) — `benchmarks/README.md`.
///   Rows past `n` inside that stripe are computed and masked out of the
///   best, as the int kernel does.
pub(crate) const KERNEL_H2_SRC: &str = r#"
#if __CUDA_ARCH__ >= 800
#define NEGINF2 0xFC00FC00u
#define RH 16

__device__ __forceinline__ unsigned int hadd2(unsigned int a, unsigned int b)
{
    unsigned int d;
    asm("add.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ unsigned int hmax2(unsigned int a, unsigned int b)
{
    unsigned int d;
    asm("max.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ unsigned int hfma2(unsigned int a, unsigned int b, unsigned int c)
{
    unsigned int d;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}
// 1.0 in each half where the halves of a and b differ, else 0.0.
__device__ __forceinline__ unsigned int hne2(unsigned int a, unsigned int b)
{
    unsigned int d;
    asm("set.ne.f16x2.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
__device__ __forceinline__ int half_lo_to_int(unsigned int v)
{
    unsigned short lo, hi;
    int d;
    asm("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(v));
    asm("cvt.rni.s32.f16 %0, %1;" : "=r"(d) : "h"(lo));
    return d;
}
__device__ __forceinline__ int half_hi_to_int(unsigned int v)
{
    unsigned short lo, hi;
    int d;
    asm("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(v));
    asm("cvt.rni.s32.f16 %0, %1;" : "=r"(d) : "h"(hi));
    return d;
}
// -inf in the half whose reference (length mA low, mB high) ended before column j.
__device__ __forceinline__ unsigned int mask_ended(unsigned int v, int j, int mA, int mB)
{
    if (j >= mA) v = (v & 0xFFFF0000u) | 0x0000FC00u;
    if (j >= mB) v = (v & 0x0000FFFFu) | 0xFC000000u;
    return v;
}

template <bool LOCAL, bool TAIL, int RR>
__device__ __forceinline__ void stripe2(
    const unsigned int* __restrict__ sp, int lanes, int t, int m, int mA, int mB, int mmin,
    unsigned int q0, unsigned int q1, int rows, bool first,
    uint2* __restrict__ bnd, size_t bstride,
    unsigned int mi2, unsigned int d2, unsigned int o2, unsigned int e2, unsigned int& best)
{
    unsigned int qm[RR];
#pragma unroll
    for (int r = 0; r < RR; r++) {
        const unsigned int a = (r < 8 ? (q0 >> (4 * r)) : (q1 >> (4 * (r - 8)))) & 0xFu;
        qm[r] = (a << 10) | (a << 26);
    }
    unsigned int H[RR];
    unsigned int E[RR];
#pragma unroll
    for (int r = 0; r < RR; r++) {
        H[r] = 0u;
        E[r] = NEGINF2;
    }
    unsigned int diag_top = 0u;
    unsigned int rw = 0u;
    uint2 bnext = first ? make_uint2(0u, NEGINF2) : bnd[0];
    for (int j = 0; j < m; j++) {
        if ((j & 3) == 0) rw = sp[(j >> 2) * lanes + t];
        const unsigned int rm = ((rw >> ((j & 3) * 4)) & 0x000F000Fu) << 10;
        unsigned int hup;
        unsigned int f;
        if (first) {
            hup = 0u;
            f = NEGINF2;
        } else {
            const uint2 b = bnext;
            if (j + 1 < m) bnext = bnd[(size_t)(j + 1) * bstride];
            hup = b.x;
            f = b.y;
        }
        unsigned int hdiag = diag_top;
        diag_top = hup;
        // Local: the column's best cell. Semi-global tail: the column's cell
        // in the read's last row.
        unsigned int cb = NEGINF2;
#pragma unroll
        for (int r = 0; r < RR; r++) {
            const unsigned int hleft = H[r];
            const unsigned int ev = hmax2(hadd2(hleft, o2), hadd2(E[r], e2));
            E[r] = ev;
            f = hmax2(hadd2(hup, o2), hadd2(f, e2));
            const unsigned int s = hfma2(hne2(qm[r] & rm, 0u), d2, mi2);
            unsigned int h = hmax2(hmax2(hadd2(hdiag, s), ev), f);
            if (LOCAL) h = hmax2(h, 0u);
            hdiag = hleft;
            H[r] = h;
            hup = h;
            if (LOCAL) {
                if (!TAIL || r < rows) cb = hmax2(cb, h);
            } else if (TAIL) {
                if (r == rows - 1) cb = h;
            }
        }
        if (!TAIL) bnd[(size_t)j * bstride] = make_uint2(hup, f);
        if (LOCAL || TAIL) {
            if (j >= mmin) cb = mask_ended(cb, j, mA, mB);
            best = hmax2(best, cb);
        }
        if (!LOCAL && (j + 1 == mA || j + 1 == mB)) {
            // A reference's last column: its half of the stripe's H column.
            unsigned int cm = NEGINF2;
#pragma unroll
            for (int r = 0; r < RR; r++) {
                if (!TAIL || r < rows) cm = hmax2(cm, H[r]);
            }
            if (j + 1 != mA) cm = (cm & 0xFFFF0000u) | 0x0000FC00u;
            if (j + 1 != mB) cm = (cm & 0x0000FFFFu) | 0xFC000000u;
            best = hmax2(best, cm);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 3) align_score_kernel_h2(
    const unsigned int* __restrict__ qwords,
    const unsigned long long* __restrict__ qoff,
    const int* __restrict__ qlen,
    const int* __restrict__ order,
    const unsigned int* __restrict__ panel,
    const int* __restrict__ ref_len,
    const int* __restrict__ ref_idx,
    short* __restrict__ out,
    unsigned int* __restrict__ scratch,
    int* __restrict__ counters,
    int n_work,
    int n_refs,
    int words_per_lane,
    unsigned int mi2, unsigned int d2, unsigned int o2, unsigned int e2,
    int local)
{
    extern __shared__ unsigned int sp[];
    __shared__ int next;
    const int lanes = blockDim.x;
    const int t = threadIdx.x;
    const int g = blockIdx.y;
    const unsigned int* gp = panel + (size_t)g * words_per_lane * lanes;
    for (int w = t; w < words_per_lane * lanes; w += lanes) sp[w] = gp[w];
    const int p = g * lanes + t;
    const int mA = ref_len[2 * p];
    const int mB = ref_len[2 * p + 1];
    const int riA = ref_idx[2 * p];
    const int riB = ref_idx[2 * p + 1];
    const int m = max(mA, mB);
    const int mmin = min(mA, mB);
    const size_t total = (size_t)gridDim.x * gridDim.y * lanes;
    const size_t gtid = ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * lanes + t;
    uint2* bnd = reinterpret_cast<uint2*>(scratch) + gtid;
    for (;;) {
        __syncthreads();
        if (t == 0) next = atomicAdd(&counters[g], 1);
        __syncthreads();
        const int k = next;
        if (k >= n_work) break;
        const int qi = order[k];
        const int n = qlen[qi];
        const unsigned int* q = qwords + qoff[qi];
        unsigned int best = local ? 0u : NEGINF2;
        if (m > 0) {
            for (int i0 = 0; i0 < n; i0 += RH) {
                const unsigned int q0 = q[i0 >> 3];
                const unsigned int q1 = q[(i0 >> 3) + 1];
                const int rows = min(RH, n - i0);
                const bool first = i0 == 0;
                const bool tail = i0 + RH >= n;
#define ARGS sp, lanes, t, m, mA, mB, mmin, q0, q1, rows, first, bnd, total, mi2, d2, o2, e2, best
                if (local) {
                    if (!tail) stripe2<true, false, 16>(ARGS);
                    else if (rows <= 4) stripe2<true, true, 4>(ARGS);
                    else if (rows <= 8) stripe2<true, true, 8>(ARGS);
                    else stripe2<true, true, 16>(ARGS);
                } else {
                    if (!tail) stripe2<false, false, 16>(ARGS);
                    else if (rows <= 4) stripe2<false, true, 4>(ARGS);
                    else if (rows <= 8) stripe2<false, true, 8>(ARGS);
                    else stripe2<false, true, 16>(ARGS);
                }
#undef ARGS
            }
            if (riA >= 0) out[(size_t)qi * n_refs + riA] = (short)half_lo_to_int(best);
            if (riB >= 0) out[(size_t)qi * n_refs + riB] = (short)half_hi_to_int(best);
        }
    }
}
#endif
"#;
