//! `--site-table`: a per-(reference, position) pileup of the primary
//! alignments, the input to population-level calling of novel modifications.
//!
//! One read cannot tell a modification from stochastic error; recurrence
//! across reads can. Base quality is a **covariate** here (a histogram of the
//! qualities seen at matches and at mismatches), never a weight.
//!
//! Counted from each written primary record's CIGAR and sequence, so what is
//! counted is exactly what the BAM says. Soft clips (including a trimmed
//! adapter) are not counted; secondary and unmapped records are not counted.
//! An insertion is attributed to the reference position it follows. A worker
//! accumulates a chunk locally and merges it into one shared table under a
//! mutex when the chunk ends: addition commutes, so the table is independent
//! of chunk order, thread count and `--sort`, and nothing waits on a batch.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::Write;
use std::sync::Mutex;

use eschalign::Panel;
use noodles_sam::alignment::RecordBuf;
use noodles_sam::alignment::record::Flags;
use noodles_sam::alignment::record::cigar::op::Kind;

/// Counter slots of one (reference, position).
const MATCH: usize = 0;
/// `MM..MM+5`: mismatches by read base (A, C, G, T, other).
const MM: usize = 1;
const DEL: usize = 6;
const INS: usize = 7;
/// Aligned bases (match + mismatch + deletion) from MAPQ-60 / tied reads.
const UNIQ_DEPTH: usize = 8;
/// Mismatch + deletion among those.
const UNIQ_DIFF: usize = 9;
const TIED_DEPTH: usize = 10;
const TIED_DIFF: usize = 11;
/// Quality histogram of matching bases, then of mismatching bases.
const QH_MATCH: usize = 12;
const QBINS: usize = 6;
const QH_MISMATCH: usize = QH_MATCH + QBINS;
const SLOTS: usize = QH_MISMATCH + QBINS;

type Cell = [u32; SLOTS];

const QBIN_NAMES: [&str; QBINS] = ["q0_9", "q10_19", "q20_29", "q30_39", "q40p", "qna"];

fn qbin(q: Option<u8>) -> usize {
    match q {
        Some(q) if q != 0xFF => ((q / 10) as usize).min(4),
        _ => 5,
    }
}

/// A multiplicative hasher for the dense `(reference, position)` keys.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = v.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type Key = u64;
type Cells = HashMap<Key, Cell, BuildHasherDefault<KeyHasher>>;

fn key(reference: usize, pos: usize) -> Key {
    ((reference as u64) << 32) | pos as u64
}

/// Counts for one chunk of reads, merged into [`SiteTable`] when it ends.
#[derive(Default)]
pub struct SiteCounts(Cells);

fn fold(b: u8) -> u8 {
    match b.to_ascii_uppercase() {
        b'U' => b'T',
        c => c,
    }
}

impl SiteCounts {
    /// Count one written record; anything but a mapped primary is ignored.
    pub fn add(&mut self, rec: &RecordBuf, panel: &Panel) {
        let flags = rec.flags();
        if flags.is_unmapped() || flags.intersects(Flags::SECONDARY | Flags::SUPPLEMENTARY) {
            return;
        }
        let (Some(ri), Some(start)) = (
            rec.reference_sequence_id(),
            rec.alignment_start().map(usize::from),
        ) else {
            return;
        };
        let refseq = &panel.get(ri).seq;
        let unique = rec.mapping_quality().map(u8::from) == Some(60);
        let (depth_slot, diff_slot) = if unique {
            (UNIQ_DEPTH, UNIQ_DIFF)
        } else {
            (TIED_DEPTH, TIED_DIFF)
        };
        let seq: &[u8] = rec.sequence().as_ref();
        let qual: &[u8] = rec.quality_scores().as_ref();
        let have_q = qual.len() == seq.len();
        let (mut r, mut q) = (start - 1, 0usize);
        let mut prev: Option<usize> = None;
        for op in rec.cigar().as_ref() {
            let n = op.len();
            match op.kind() {
                Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                    for _ in 0..n {
                        let (Some(&rb), Some(&qb)) = (refseq.get(r), seq.get(q)) else {
                            return;
                        };
                        let c = self.0.entry(key(ri, r)).or_insert([0; SLOTS]);
                        let bin = qbin(have_q.then(|| qual[q]));
                        c[depth_slot] += 1;
                        if fold(rb) == fold(qb) {
                            c[MATCH] += 1;
                            c[QH_MATCH + bin] += 1;
                        } else {
                            let m = match fold(qb) {
                                b'A' => 0,
                                b'C' => 1,
                                b'G' => 2,
                                b'T' => 3,
                                _ => 4,
                            };
                            c[MM + m] += 1;
                            c[diff_slot] += 1;
                            c[QH_MISMATCH + bin] += 1;
                        }
                        prev = Some(r);
                        r += 1;
                        q += 1;
                    }
                }
                Kind::Deletion => {
                    for _ in 0..n {
                        let c = self.0.entry(key(ri, r)).or_insert([0; SLOTS]);
                        c[DEL] += 1;
                        c[depth_slot] += 1;
                        c[diff_slot] += 1;
                        prev = Some(r);
                        r += 1;
                    }
                }
                Kind::Insertion => {
                    if let Some(p) = prev {
                        self.0.entry(key(ri, p)).or_insert([0; SLOTS])[INS] += 1;
                    }
                    q += n;
                }
                Kind::SoftClip => q += n,
                Kind::Skip => r += n,
                Kind::HardClip | Kind::Pad => {}
            }
        }
    }
}

/// The shared table every chunk's counts are merged into.
pub struct SiteTable {
    cells: Mutex<Cells>,
}

impl SiteTable {
    pub fn new() -> Self {
        SiteTable {
            cells: Mutex::new(Cells::default()),
        }
    }

    pub fn merge(&self, local: SiteCounts) {
        let mut all = self.cells.lock().expect("site table lock");
        for (k, c) in local.0 {
            let dst = all.entry(k).or_insert([0; SLOTS]);
            for (d, s) in dst.iter_mut().zip(c) {
                *d += s;
            }
        }
    }

    /// Write the table, gzip-compressed TSV, rows by reference then position.
    pub fn write<W: Write>(self, panel: &Panel, out: W) -> std::io::Result<()> {
        let all = self.cells.into_inner().expect("site table lock");
        let mut rows: Vec<(Key, Cell)> = all.into_iter().collect();
        rows.sort_unstable_by_key(|r| r.0);
        let mut w = flate2::write::GzEncoder::new(
            std::io::BufWriter::new(out),
            flate2::Compression::default(),
        );
        write!(
            w,
            "reference\tposition\tref_base\tdepth\tmatch\tmm_A\tmm_C\tmm_G\tmm_T\tmm_N\tdel\tins\t\
             uniq_depth\tuniq_diff\ttied_depth\ttied_diff"
        )?;
        for kind in ["m", "x"] {
            for b in QBIN_NAMES {
                write!(w, "\t{kind}_{b}")?;
            }
        }
        writeln!(w)?;
        for (k, c) in rows {
            let (ri, pos) = ((k >> 32) as usize, (k & 0xFFFF_FFFF) as usize);
            let reference = panel.get(ri);
            let depth = c[UNIQ_DEPTH] + c[TIED_DEPTH];
            write!(
                w,
                "{}\t{}\t{}\t{}",
                reference.name,
                pos + 1,
                char::from(reference.seq[pos].to_ascii_uppercase()),
                depth
            )?;
            for v in &c[MATCH..=INS] {
                write!(w, "\t{v}")?;
            }
            for v in &c[UNIQ_DEPTH..SLOTS] {
                write!(w, "\t{v}")?;
            }
            writeln!(w)?;
        }
        w.finish()?.flush()
    }
}
