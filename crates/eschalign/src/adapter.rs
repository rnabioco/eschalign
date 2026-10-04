// SPDX-License-Identifier: MIT

//! 3' adapter read-through, found before alignment so that nothing is clipped
//! by the aligner.
//!
//! A mate that reads past the end of a short insert carries the sequencing
//! adapter on its 3' end. Left in, the adapter either costs score against the
//! reference or, in `local` mode, is soft-clipped together with whatever
//! mismatches sit beside it — which would erase the misincorporation and
//! RT-stop signal at the read end. So the adapter is found *first*, by its
//! known prefix, and the caller aligns only the bases before it (end to end on
//! the reference) and writes the adapter bases as a soft clip it knows the
//! length of.
//!
//! # The match rule
//!
//! The adapter's prefix is searched at every position `p` of the read, leftmost
//! first. The compared stretch is `read[p..]` against the adapter's first
//! `min(adapter length, read length - p)` bases, so a read that runs out
//! mid-adapter still counts. A stretch is the adapter when it is at least
//! `min_overlap` long and has at most one mismatch per ten bases of overlap
//! (cutadapt's default error rate, `overlap / 10`, rounded down — so none below
//! ten bases). Letters outside A/C/G/T are wildcards, as everywhere in this
//! crate. Everything from `p` to the end of the read is adapter.
//!
//! # Mates agree where the insert ends
//!
//! Two mates of one molecule that both read through must cut the insert at the
//! same place: `len1 - ad1 == len2 - ad2`. [`detect_pair`] checks it, and where
//! the two disagree one detection is chance. The one with the shorter adapter
//! overlap (the weaker evidence) is dropped; on a tie both are. The check is
//! made in insert coordinates — the read length less the adapter — rather
//! than after alignment: a trimmed mate's 3' end *is* the insert end, so the
//! two are the same quantity, and checking before alignment keeps the
//! alignment's input fixed.

use crate::alphabet::{WILDCARD, encode};

/// TruSeq Read 1 adapter, as it appears on the 3' end of a read-through mate 1.
pub const TRUSEQ_R1: &[u8] = b"AGATCGGAAGAGCACACGTCTGAACTCCAGTCA";
/// TruSeq Read 2 adapter, as it appears on the 3' end of a read-through mate 2.
pub const TRUSEQ_R2: &[u8] = b"AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT";
/// Default shortest stretch counted as adapter.
pub const DEFAULT_MIN_OVERLAP: usize = 5;

/// What to look for.
#[derive(Debug, Clone, Copy)]
pub struct AdapterConfig<'a> {
    /// Adapter prefix on mate 1's 3' end.
    pub r1: &'a [u8],
    /// Adapter prefix on mate 2's 3' end.
    pub r2: &'a [u8],
    /// Shortest match counted as adapter (bases).
    pub min_overlap: usize,
}

impl Default for AdapterConfig<'static> {
    fn default() -> Self {
        Self {
            r1: TRUSEQ_R1,
            r2: TRUSEQ_R2,
            min_overlap: DEFAULT_MIN_OVERLAP,
        }
    }
}

/// How many bases at the 3' end of `read` are `adapter` (0 = none).
pub fn find(read: &[u8], adapter: &[u8], min_overlap: usize) -> usize {
    let read = encode(read);
    let adapter = encode(adapter);
    let min_overlap = min_overlap.max(1);
    if read.len() < min_overlap || adapter.is_empty() {
        return 0;
    }
    for p in 0..=read.len() - min_overlap {
        let overlap = adapter.len().min(read.len() - p);
        if overlap < min_overlap {
            break;
        }
        let mismatches = (0..overlap)
            .filter(|&k| {
                let (a, b) = (read[p + k], adapter[k]);
                a != b && a != WILDCARD && b != WILDCARD
            })
            .count();
        if mismatches <= overlap / 10 {
            return read.len() - p;
        }
    }
    0
}

/// Adapter bases at the 3' end of each mate, with the two mates' insert ends
/// reconciled (see the module docs): `[ad1, ad2]`.
pub fn detect_pair(r1: &[u8], r2: &[u8], cfg: &AdapterConfig<'_>) -> [usize; 2] {
    let mut ad = [
        find(r1, cfg.r1, cfg.min_overlap),
        find(r2, cfg.r2, cfg.min_overlap),
    ];
    let (l1, l2) = (r1.len(), r2.len());
    if ad[0] > 0 && ad[1] > 0 && l1 - ad[0] != l2 - ad[1] {
        let (e1, e2) = (ad[0].min(cfg.r1.len()), ad[1].min(cfg.r2.len()));
        match e1.cmp(&e2) {
            std::cmp::Ordering::Greater => ad[1] = 0,
            std::cmp::Ordering::Less => ad[0] = 0,
            std::cmp::Ordering::Equal => ad = [0, 0],
        }
    }
    ad
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_and_partial_adapters_are_found() {
        let mut r = b"ACGTACGTAC".to_vec();
        r.extend_from_slice(&TRUSEQ_R1[..20]);
        assert_eq!(find(&r, TRUSEQ_R1, 5), 20);
        // The read ends inside the adapter's first bases.
        let mut r = b"ACGTACGTAC".to_vec();
        r.extend_from_slice(&TRUSEQ_R1[..6]);
        assert_eq!(find(&r, TRUSEQ_R1, 5), 6);
        // A full adapter followed by more sequence: all of it is adapter.
        let mut r = b"ACGTACGTAC".to_vec();
        r.extend_from_slice(TRUSEQ_R1);
        r.extend_from_slice(b"TTTT");
        assert_eq!(find(&r, TRUSEQ_R1, 5), TRUSEQ_R1.len() + 4);
    }

    #[test]
    fn mismatch_tolerance_scales_with_overlap() {
        let mut r = b"GGGGGGGGGGGG".to_vec();
        let mut a = TRUSEQ_R1[..20].to_vec();
        a[5] = b'T'; // 1 mismatch in 20 bases: allowed (2)
        a[12] = b'T'; // 2 mismatches: still allowed
        r.extend_from_slice(&a);
        assert_eq!(find(&r, TRUSEQ_R1, 5), 20);
        a[15] = b'C'; // 3 in 20: refused
        let mut r = b"GGGGGGGGGGGG".to_vec();
        r.extend_from_slice(&a);
        assert_eq!(find(&r, TRUSEQ_R1, 5), 0);
        // Under ten bases of overlap there is no tolerance at all.
        let mut r = b"GGGGGGGGGGGG".to_vec();
        r.extend_from_slice(b"AGATCGGAT");
        assert_eq!(find(&r, TRUSEQ_R1, 5), 0);
    }

    #[test]
    fn short_overlap_below_the_minimum_is_not_adapter() {
        let mut r = b"GGGGGGGGGGGG".to_vec();
        r.extend_from_slice(b"AGAT");
        assert_eq!(find(&r, TRUSEQ_R1, 5), 0);
        assert_eq!(find(&r, TRUSEQ_R1, 4), 4);
        assert_eq!(find(b"AG", TRUSEQ_R1, 5), 0);
        assert_eq!(find(b"", TRUSEQ_R1, 5), 0);
    }

    #[test]
    fn wildcards_match() {
        let mut r = b"GGGGGGGGGGGG".to_vec();
        r.extend_from_slice(b"AGATNGGAAGAG");
        assert_eq!(find(&r, TRUSEQ_R1, 5), 12);
    }

    #[test]
    fn mates_must_agree_where_the_insert_ends() {
        let cfg = AdapterConfig::default();
        let insert = b"GGGGCCCCGGGGCCCCGGGG"; // 20 nt
        let mk = |ad: &[u8], extra: usize| {
            let mut r = insert.to_vec();
            r.extend_from_slice(&ad[..extra]);
            r
        };
        // Both read through by 12 bases: they agree.
        let (a, b) = (mk(TRUSEQ_R1, 12), mk(TRUSEQ_R2, 12));
        assert_eq!(detect_pair(&a, &b, &cfg), [12, 12]);
        // Mate 2's insert ends two bases later: the weaker detection goes.
        let shifted = |ad: &[u8], extra: usize| {
            let mut r = insert.to_vec();
            r.extend_from_slice(b"CC");
            r.extend_from_slice(&ad[..extra]);
            r
        };
        let b2 = shifted(TRUSEQ_R2, 9);
        assert_eq!(detect_pair(&a, &b2, &cfg), [12, 0]);
        let a2 = shifted(TRUSEQ_R1, 7);
        assert_eq!(detect_pair(&a2, &b, &cfg), [0, 12]);
        // Equal evidence, different ends: neither.
        let a3 = mk(TRUSEQ_R1, 8);
        let b3 = shifted(TRUSEQ_R2, 8);
        assert_eq!(detect_pair(&a3, &b3, &cfg), [0, 0]);
        // One mate only: stands.
        let clean = insert.to_vec();
        assert_eq!(detect_pair(&a, &clean, &cfg), [12, 0]);
    }
}
