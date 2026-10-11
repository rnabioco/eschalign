// SPDX-License-Identifier: MIT

//! Pair-aware winner selection: [`Aligner::map_pairs`].
//!
//! # Semantics
//!
//! A *candidate* is a reference together with a template strand. On the
//! forward template strand mate 1 is read as sequenced and mate 2 is
//! reverse-complemented (an `FR` pair, mate 1 upstream); with
//! [`PairOptions::both_strands`] the reverse template strand is a candidate
//! too (mate 1 reverse-complemented, mate 2 as sequenced). [`PairOrientation`]
//! picks the other layouts: `RF` keeps the strands but wants the mates facing
//! away from each other, `FF` puts both mates on the template strand. A candidate's score
//! is `score(mate 1) + score(mate 2)`, each against the reference, and it is
//! eligible only when **both** mates are at or above
//! [`PairOptions::min_score`] — the sum alone does not let one good mate carry
//! a bad one.
//!
//! Eligible candidates are taken in descending score. The first score for
//! which at least one candidate also satisfies the pair constraints is the
//! pair's score, and **all** such candidates at that score are the *tie set*
//! (reference order, forward template strand first); its first member is the
//! primary. The constraints are checked on the traced alignments, so a score
//! that only exists as an impossible pair (mates 10 kb apart, facing away from
//! each other) falls through to the next score rather than winning:
//!
//! * both mates align at least one base;
//! * the forward-strand mate starts before the reverse-strand mate ends (no
//!   "outie" — a pair facing away from each other);
//! * the insert — from the leftmost aligned base of either mate to the
//!   rightmost — is at most [`PairOptions::max_insert`]. An insert shorter than
//!   the reads (the mates overlap completely, one reading through into
//!   adapter) is valid, and is what it is: the span, not the read length.
//!
//! If no candidate qualifies the pair is [`PairMapping::Separate`]: each mate
//! is mapped exactly as [`Aligner::map_reads`] maps a read, on **both**
//! strands (a mate that is the wrong way round must be reported where it is,
//! not forced onto the strand the pair rule wanted) with its own tie set.
//!
//! `XS` per mate is that mate's best score over the candidates *outside* the
//! pair's tie set, mirroring the unpaired definition.

use super::*;

/// How the two mates of a proper pair face each other.
///
/// Every candidate has a *template strand*: mate 1's strand. The orientation
/// fixes mate 2's strand relative to it, and the geometry a pair must have on
/// the reference (`--pair-orientation` on the CLI).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PairOrientation {
    /// Mates on opposite strands facing each other (Illumina's usual
    /// "innie"): the forward-strand mate starts before the reverse-strand
    /// mate ends.
    #[default]
    Fr,
    /// Mates on opposite strands facing away from each other ("outie"): the
    /// reverse-strand mate starts at or before the forward-strand mate.
    Rf,
    /// Mates on the same strand, mate 1 upstream of mate 2 along it.
    Ff,
}

impl PairOrientation {
    /// Whether mate `m` (0 or 1) is read reverse-complemented when mate 1
    /// follows the template strand `template_reverse`.
    fn mate_reverse(self, m: usize, template_reverse: bool) -> bool {
        match self {
            PairOrientation::Fr | PairOrientation::Rf => template_reverse ^ (m == 1),
            PairOrientation::Ff => template_reverse,
        }
    }

    /// The geometry check on two traced mates (`al[0]`, `al[1]`).
    fn geometry_ok(self, al: &[Alignment; 2], template_reverse: bool) -> bool {
        // Mate order along the template strand: mate 1 first on the forward
        // template strand, mate 2 first on the reverse one.
        let (first, second) = if template_reverse {
            (&al[1], &al[0])
        } else {
            (&al[0], &al[1])
        };
        match self {
            PairOrientation::Fr => first.ref_start < second.ref_end,
            PairOrientation::Rf => second.ref_start <= first.ref_start,
            PairOrientation::Ff => first.ref_start <= second.ref_start,
        }
    }
}

/// Per-pair policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairOptions {
    /// Both mates must score at least this to be a candidate.
    pub min_score: i32,
    /// How many tie-set members *besides the primary* are reported
    /// (`None` = all). The tie set is always counted in full.
    pub max_ties: Option<usize>,
    /// Longest allowed insert, bases.
    pub max_insert: usize,
    /// Also consider the reverse template strand (mate 1 reversed).
    pub both_strands: bool,
    /// How the mates must face each other.
    pub orientation: PairOrientation,
}

impl Default for PairOptions {
    fn default() -> Self {
        Self {
            min_score: 0,
            max_ties: None,
            max_insert: 500,
            both_strands: false,
            orientation: PairOrientation::Fr,
        }
    }
}

/// One traced placement of a pair on one reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PairHit {
    /// Panel index of the reference.
    pub reference: usize,
    /// The template is on the reverse strand: mate 1 aligned reverse-
    /// complemented, mate 2 as sequenced.
    pub reverse: bool,
    /// Mate 1's and mate 2's alignments. Each [`Hit::reverse`] is that mate's
    /// own strand, and its coordinates are on the mate as aligned.
    pub mates: [Hit; 2],
    /// Leftmost aligned reference base of either mate to the rightmost.
    pub insert: usize,
}

impl PairHit {
    /// The pair score: the two mates' scores summed.
    pub fn score(&self) -> i32 {
        self.mates[0].alignment.score + self.mates[1].alignment.score
    }
}

/// What one read pair produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairMapping {
    /// A pair that satisfies every constraint on some reference.
    Proper {
        /// The pair score (both mates' scores summed).
        score: i32,
        /// Size of the tie set (≥ 1).
        n_tied: usize,
        /// Per mate: the best score outside the tie set (SAM `XS`).
        suboptimal: [Option<i32>; 2],
        /// Primary first, then up to `max_ties` more tie-set members.
        hits: Vec<PairHit>,
    },
    /// No reference satisfies the constraints: each mate at its own best
    /// reference, as an unpaired read, on either strand.
    Separate([ReadMapping; 2]),
}

/// One eligible candidate.
#[derive(Clone, Copy)]
struct Cand {
    sum: i32,
    reference: usize,
    reverse: bool,
    scores: [i32; 2],
}

struct PairState<'a> {
    letters: [&'a [u8]; 2],
    /// `[mate][reverse]`: codes of the mate as sequenced / reverse-complemented.
    codes: [[Vec<u8>; 2]; 2],
    /// `[mate][reverse]`: the panel scores, when this pair has candidates.
    rows: [[Vec<i32>; 2]; 2],
    cands: Vec<Cand>,
    /// `cands` ranges of equal sum, best first.
    tiers: Vec<(usize, usize)>,
    next_tier: usize,
    /// The tie set, once found.
    found: Vec<(Cand, [Alignment; 2])>,
}

impl PairState<'_> {
    fn open(&self) -> bool {
        self.found.is_empty() && self.next_tier < self.tiers.len()
    }
}

impl Aligner {
    /// Align read pairs (letters, as sequenced; mate 1 then mate 2), one
    /// [`PairMapping`] each, in order. See the module docs for the rules.
    pub fn map_pairs(&self, pairs: &[[&[u8]; 2]], opts: &PairOptions) -> Vec<PairMapping> {
        self.map_pairs_scored(pairs, opts, |_, _, _| None)
    }

    /// [`Aligner::map_pairs`] with some panel scores already computed
    /// elsewhere (the CUDA kernel). `prescored(pair, mate, reverse)` is that
    /// mate's score against every reference, in panel order, as sequenced
    /// (`reverse == false`) or reverse-complemented; `None` scores it here.
    /// The scores are the caller's contract, exactly as for
    /// [`Aligner::map_reads_scored`]: they must equal [`Aligner::score_all`]'s.
    pub fn map_pairs_scored<'s>(
        &self,
        pairs: &[[&[u8]; 2]],
        opts: &PairOptions,
        prescored: impl Fn(usize, usize, bool) -> Option<&'s [i16]>,
    ) -> Vec<PairMapping> {
        let n_refs = self.panel.len();
        let strands: &[bool] = if opts.both_strands {
            &[false, true]
        } else {
            &[false]
        };

        // Phase 1: score, and rank the eligible candidates.
        let mut states: Vec<PairState<'_>> = pairs
            .iter()
            .enumerate()
            .map(|(p, letters)| {
                let fwd = [encode(letters[0]), encode(letters[1])];
                let rev = [
                    alphabet::reverse_complement_codes(&fwd[0]),
                    alphabet::reverse_complement_codes(&fwd[1]),
                ];
                let mut st = PairState {
                    letters: *letters,
                    codes: [
                        [fwd[0].clone(), rev[0].clone()],
                        [fwd[1].clone(), rev[1].clone()],
                    ],
                    rows: Default::default(),
                    cands: Vec::new(),
                    tiers: Vec::new(),
                    next_tier: 0,
                    found: Vec::new(),
                };
                if fwd[0].is_empty() || fwd[1].is_empty() {
                    return st;
                }
                for &t in strands {
                    for m in 0..2 {
                        let rv = opts.orientation.mate_reverse(m, t);
                        if st.rows[m][usize::from(rv)].is_empty() {
                            let mut row = Vec::new();
                            self.score_or_take(
                                &st.codes[m][usize::from(rv)],
                                prescored(p, m, rv),
                                &mut row,
                            );
                            st.rows[m][usize::from(rv)] = row;
                        }
                    }
                    let r0 = &st.rows[0][usize::from(opts.orientation.mate_reverse(0, t))];
                    let r1 = &st.rows[1][usize::from(opts.orientation.mate_reverse(1, t))];
                    for r in 0..n_refs {
                        let (a, b) = (r0[r], r1[r]);
                        if a >= opts.min_score && b >= opts.min_score {
                            st.cands.push(Cand {
                                sum: a + b,
                                reference: r,
                                reverse: t,
                                scores: [a, b],
                            });
                        }
                    }
                }
                // Best sum first; ties in reference order, forward strand first.
                st.cands
                    .sort_by_key(|c| (std::cmp::Reverse(c.sum), c.reference, c.reverse));
                let mut start = 0;
                for k in 1..=st.cands.len() {
                    if k == st.cands.len() || st.cands[k].sum != st.cands[start].sum {
                        st.tiers.push((start, k));
                        start = k;
                    }
                }
                st
            })
            .collect();

        // Phase 2: trace tier by tier, every open pair's next tier together,
        // until each pair has a tie set or runs out of tiers.
        loop {
            let plan: Vec<(usize, usize, usize)> = states
                .iter()
                .enumerate()
                .filter(|(_, st)| st.open())
                .map(|(p, st)| {
                    let (a, b) = st.tiers[st.next_tier];
                    (p, a, b)
                })
                .collect();
            if plan.is_empty() {
                break;
            }
            let traced = {
                let mut jobs: Vec<(&[u8], &[u8])> = Vec::new();
                for &(p, a, b) in &plan {
                    let st = &states[p];
                    for c in &st.cands[a..b] {
                        let reference = &self.panel.get(c.reference).codes[..];
                        for m in 0..2 {
                            let rv = opts.orientation.mate_reverse(m, c.reverse);
                            jobs.push((&st.codes[m][usize::from(rv)], reference));
                        }
                    }
                }
                simd::align_pairs(self.backend, &jobs, &self.scoring, self.mode)
            };
            let mut traced = traced.into_iter();
            for &(p, a, b) in &plan {
                let st = &mut states[p];
                let mut found = Vec::new();
                for k in a..b {
                    let c = st.cands[k];
                    let al = [traced.next().unwrap(), traced.next().unwrap()];
                    debug_assert!(
                        al[0].score == c.scores[0] && al[1].score == c.scores[1],
                        "traceback disagrees with the kernel"
                    );
                    if self.pair_ok(&al, c.reverse, opts) {
                        found.push((c, al));
                    }
                }
                if found.is_empty() {
                    st.next_tier += 1;
                } else {
                    st.found = found;
                }
            }
        }

        // Phase 3: build the proper pairs; collect the rest for the fallback.
        let mut out: Vec<Option<PairMapping>> = Vec::with_capacity(states.len());
        let mut separate: Vec<usize> = Vec::new();
        for (p, st) in states.iter_mut().enumerate() {
            if st.found.is_empty() {
                separate.push(p);
                out.push(None);
                continue;
            }
            let found = std::mem::take(&mut st.found);
            let n_tied = found.len();
            let score = found[0].0.sum;
            // Best score outside the tie set, per mate.
            let mut in_set = vec![false; 2 * n_refs];
            for (c, _) in &found {
                in_set[2 * c.reference + usize::from(c.reverse)] = true;
            }
            let mut suboptimal = [None; 2];
            for &t in strands {
                for r in 0..n_refs {
                    if in_set[2 * r + usize::from(t)] {
                        continue;
                    }
                    for (m, sub) in suboptimal.iter_mut().enumerate() {
                        let s = st.rows[m][usize::from(opts.orientation.mate_reverse(m, t))][r];
                        *sub = Some(sub.map_or(s, |x: i32| x.max(s)));
                    }
                }
            }
            let keep = opts
                .max_ties
                .map_or(n_tied, |k| n_tied.min(k.saturating_add(1)));
            let rc = [
                alphabet::reverse_complement(st.letters[0]),
                alphabet::reverse_complement(st.letters[1]),
            ];
            let hits = found
                .into_iter()
                .take(keep)
                .map(|(c, al)| {
                    let reference = self.panel.get(c.reference);
                    let [a0, a1] = al;
                    let mut mates = [a0, a1].into_iter().enumerate().map(|(m, alignment)| {
                        let rv = opts.orientation.mate_reverse(m, c.reverse);
                        let letters: &[u8] = if rv { &rc[m] } else { st.letters[m] };
                        let (md, nm) = sam::md_nm(
                            letters,
                            &reference.seq,
                            alignment.query_start,
                            alignment.ref_start,
                            &alignment.ops,
                        );
                        Hit {
                            reference: c.reference,
                            reverse: rv,
                            alignment,
                            md,
                            nm,
                        }
                    });
                    let mates = [mates.next().unwrap(), mates.next().unwrap()];
                    let insert = span(&mates);
                    PairHit {
                        reference: c.reference,
                        reverse: c.reverse,
                        mates,
                        insert,
                    }
                })
                .collect();
            out.push(Some(PairMapping::Proper {
                score,
                n_tied,
                suboptimal,
                hits,
            }));
        }

        // Pairs with no qualifying candidate: each mate on its own, either strand.
        if !separate.is_empty() {
            let flat: Vec<&[u8]> = separate
                .iter()
                .flat_map(|&p| [pairs[p][0], pairs[p][1]])
                .collect();
            let each = MapOptions {
                min_score: opts.min_score,
                max_ties: opts.max_ties,
                both_strands: true,
            };
            let mut mapped = self
                .map_reads_scored(&flat, &each, |k, rev| {
                    prescored(separate[k / 2], k % 2, rev)
                })
                .into_iter();
            for &p in &separate {
                let a = mapped.next().unwrap();
                let b = mapped.next().unwrap();
                out[p] = Some(PairMapping::Separate([a, b]));
            }
        }
        out.into_iter().map(Option::unwrap).collect()
    }

    /// The pair constraints on two traced mates (see the module docs).
    fn pair_ok(&self, al: &[Alignment; 2], template_reverse: bool, opts: &PairOptions) -> bool {
        if al[0].is_empty() || al[1].is_empty() {
            return false;
        }
        if !opts.orientation.geometry_ok(al, template_reverse) {
            return false;
        }
        let lo = al[0].ref_start.min(al[1].ref_start);
        let hi = al[0].ref_end.max(al[1].ref_end);
        hi - lo <= opts.max_insert
    }
}

/// Leftmost aligned reference base to rightmost, over both mates.
fn span(mates: &[Hit; 2]) -> usize {
    let lo = mates[0]
        .alignment
        .ref_start
        .min(mates[1].alignment.ref_start);
    let hi = mates[0].alignment.ref_end.max(mates[1].alignment.ref_end);
    hi - lo
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two paralogs that differ at one position, 60 nt each, in a panel with
    /// an unrelated third reference.
    fn panel() -> (Panel, Vec<u8>, Vec<u8>) {
        let mut x = 0x2545_f491_4f6c_dd1du64;
        let mut base = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            b"ACGT"[(x % 4) as usize]
        };
        let a: Vec<u8> = (0..60).map(|_| base()).collect();
        let mut b = a.clone();
        b[50] = if a[50] == b'A' { b'C' } else { b'A' };
        let c: Vec<u8> = (0..60).map(|_| base()).collect();
        let p = Panel::new([("a", a.clone()), ("b", b.clone()), ("c", c)]).unwrap();
        (p, a, b)
    }

    fn aligner(mode: Mode) -> (Aligner, Vec<u8>, Vec<u8>) {
        let (p, a, b) = panel();
        (Aligner::new(p, Scoring::default(), mode).unwrap(), a, b)
    }

    #[test]
    fn pair_picks_the_paralog_its_mates_support() {
        let (al, a, _) = aligner(Mode::SemiGlobal);
        // Mate 1 covers 0..30 (identical in both paralogs); mate 2 covers
        // 30..60 of the template, reverse-complemented, and carries a's base.
        let r1 = a[..30].to_vec();
        let r2 = alphabet::reverse_complement(&a[30..]);
        let alone = al.map_read(&r1, &MapOptions::default());
        assert!(alone.n_tied >= 2, "mate 1 alone is a tie");
        let m = al.map_pairs(&[[&r1, &r2]], &PairOptions::default());
        match &m[0] {
            PairMapping::Proper {
                n_tied,
                hits,
                score,
                ..
            } => {
                assert_eq!(*n_tied, 1);
                assert_eq!(hits[0].reference, 0);
                assert_eq!(*score, 120);
                assert_eq!(hits[0].insert, 60);
                assert!(!hits[0].mates[0].reverse && hits[0].mates[1].reverse);
            }
            other => panic!("not proper: {other:?}"),
        }
    }

    #[test]
    fn over_max_insert_and_wrong_orientation_are_separate() {
        let (al, a, _) = aligner(Mode::SemiGlobal);
        let r1 = a[..20].to_vec();
        let r2 = alphabet::reverse_complement(&a[40..]);
        let opts = PairOptions {
            max_insert: 50,
            min_score: 30,
            ..PairOptions::default()
        };
        assert!(matches!(
            al.map_pairs(&[[&r1, &r2]], &opts)[0],
            PairMapping::Separate(_)
        ));
        // Semi-global scoring gives any read some positive score somewhere,
        // so a floor is what makes a wrong-way mate "not aligned".
        let roomy = PairOptions {
            min_score: 30,
            ..PairOptions::default()
        };
        assert!(matches!(
            al.map_pairs(&[[&r1, &r2]], &roomy)[0],
            PairMapping::Proper { .. }
        ));
        // Both mates as sequenced on the same strand: FF.
        let ff2 = a[40..].to_vec();
        assert!(matches!(
            al.map_pairs(&[[&r1, &ff2]], &roomy)[0],
            PairMapping::Separate(_)
        ));
        // An outie: mate 2 entirely upstream of mate 1.
        let up = alphabet::reverse_complement(&a[..20]);
        let down = a[40..].to_vec();
        assert!(matches!(
            al.map_pairs(&[[&down, &up]], &roomy)[0],
            PairMapping::Separate(_)
        ));
    }

    #[test]
    fn reverse_template_strand_only_with_both_strands() {
        let (al, a, _) = aligner(Mode::SemiGlobal);
        // Mate 1 reverse-complemented from the downstream part, mate 2 forward upstream.
        let r1 = alphabet::reverse_complement(&a[40..]);
        let r2 = a[..20].to_vec();
        let fwd = PairOptions {
            min_score: 30,
            ..PairOptions::default()
        };
        assert!(matches!(
            al.map_pairs(&[[&r1, &r2]], &fwd)[0],
            PairMapping::Separate(_)
        ));
        let both = PairOptions {
            both_strands: true,
            ..fwd
        };
        match &al.map_pairs(&[[&r1, &r2]], &both)[0] {
            PairMapping::Proper { hits, .. } => {
                assert!(hits[0].reverse);
                assert!(hits[0].mates[0].reverse && !hits[0].mates[1].reverse);
            }
            other => panic!("not proper: {other:?}"),
        }
    }

    #[test]
    fn orientation_rf_and_ff() {
        let (al, a, _) = aligner(Mode::SemiGlobal);
        let rf = PairOptions {
            min_score: 30,
            orientation: PairOrientation::Rf,
            ..PairOptions::default()
        };
        let ff = PairOptions {
            orientation: PairOrientation::Ff,
            ..rf
        };
        // An FR pair (mate 1 forward upstream, mate 2 reverse downstream) is
        // not an RF pair: it is the same alignments, so only the geometry
        // decides, and an innie is not an outie.
        let r1 = a[..30].to_vec();
        let r2 = alphabet::reverse_complement(&a[30..]);
        assert!(matches!(
            al.map_pairs(
                &[[&r1, &r2]],
                &PairOptions {
                    orientation: PairOrientation::Fr,
                    ..rf
                }
            )[0],
            PairMapping::Proper { .. }
        ));
        assert!(matches!(
            al.map_pairs(&[[&r1, &r2]], &rf)[0],
            PairMapping::Separate(_)
        ));
        // An outie: mate 1 forward downstream, mate 2 reverse upstream.
        let o1 = a[30..].to_vec();
        let o2 = alphabet::reverse_complement(&a[..30]);
        match &al.map_pairs(&[[&o1, &o2]], &rf)[0] {
            PairMapping::Proper { hits, .. } => {
                assert!(!hits[0].mates[0].reverse && hits[0].mates[1].reverse);
            }
            other => panic!("not proper: {other:?}"),
        }
        // Same strand, mate 1 upstream.
        let f1 = a[..30].to_vec();
        let f2 = a[30..].to_vec();
        match &al.map_pairs(&[[&f1, &f2]], &ff)[0] {
            PairMapping::Proper { hits, .. } => {
                assert!(!hits[0].mates[0].reverse && !hits[0].mates[1].reverse);
            }
            other => panic!("not proper: {other:?}"),
        }
        // Mate 2 upstream of mate 1 on the same strand is not FF.
        assert!(matches!(
            al.map_pairs(&[[&f2, &f1]], &ff)[0],
            PairMapping::Separate(_)
        ));
    }

    #[test]
    fn both_mates_must_clear_min_score() {
        let (al, a, _) = aligner(Mode::SemiGlobal);
        let r1 = a[..30].to_vec();
        let r2 = alphabet::reverse_complement(&a[30..]);
        let strict = PairOptions {
            min_score: 61,
            ..PairOptions::default()
        };
        // Each mate scores 60: the sum (120) clears 61, a mate (60) does not.
        assert!(matches!(
            al.map_pairs(&[[&r1, &r2]], &strict)[0],
            PairMapping::Separate(_)
        ));
    }

    #[test]
    fn batched_pairs_equal_one_at_a_time() {
        let (al, a, b) = aligner(Mode::SemiGlobal);
        let reads: Vec<[Vec<u8>; 2]> = vec![
            [a[..30].to_vec(), alphabet::reverse_complement(&a[30..])],
            [b[..30].to_vec(), alphabet::reverse_complement(&b[30..])],
            [a[..20].to_vec(), a[40..].to_vec()],
            [Vec::new(), a[..20].to_vec()],
        ];
        let pairs: Vec<[&[u8]; 2]> = reads.iter().map(|r| [&r[0][..], &r[1][..]]).collect();
        let opts = PairOptions::default();
        let all = al.map_pairs(&pairs, &opts);
        for (p, m) in pairs.iter().zip(&all) {
            assert_eq!(&al.map_pairs(&[*p], &opts)[0], m);
        }
    }
}
