// SPDX-License-Identifier: MIT

//! Known-modification table and the event classifier behind `--mods`.
//!
//! A modified base shows up in direct-RNA reads as mismatches, deletions or
//! insertions around it. Given a table of known sites, every such event of an
//! alignment is either *explained* (inside a site's influence window and of an
//! effect that site expects) or *unexplained*. This is pure annotation: it
//! reads an [`Alignment`] and never touches scoring, and base quality plays no
//! part. No I/O here; the CLI parses the sidecar and validates it against the
//! panel.
//!
//! Every inserted or deleted base is one event (like `NM`); an insertion is
//! anchored at the reference base before it (the first aligned base when it
//! opens the alignment). Unaligned (clipped or free-end) bases are not events.

use crate::alphabet;
use crate::scalar::{Alignment, CigarOp};

/// What a site is expected to do to the read. Built from the sidecar's
/// `expect` strings: `mismatch:<bases>`, `deletion`, `insertion`, `qdip`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Expect {
    /// Read bases (one bit per A/C/G/T [`alphabet`] code) a mismatch may show.
    mismatch: u8,
    pub deletion: bool,
    pub insertion: bool,
    /// A base-quality dip is expected here. Kept for the site table; quality
    /// never enters the classification, so it explains no event.
    pub qdip: bool,
}

impl Expect {
    /// Add one `expect` entry.
    pub fn add(&mut self, s: &str) -> Result<(), String> {
        match s {
            "deletion" => self.deletion = true,
            "insertion" => self.insertion = true,
            "qdip" => self.qdip = true,
            _ => {
                let bases = s.strip_prefix("mismatch:").ok_or_else(|| {
                    format!("unknown effect '{s}' (mismatch:<bases>, deletion, insertion, qdip)")
                })?;
                if bases.is_empty() {
                    return Err(format!("'{s}' names no bases"));
                }
                for b in bases.bytes() {
                    match alphabet::encode_base(b) {
                        c if c < 4 => self.mismatch |= 1 << c,
                        _ => return Err(format!("'{s}': '{}' is not A, C, G, T or U", b as char)),
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether a mismatch showing read base `b` is expected.
    pub fn explains_mismatch(&self, b: u8) -> bool {
        let c = alphabet::encode_base(b);
        c < 4 && self.mismatch & (1 << c) != 0
    }
}

/// One known modification site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModSite {
    /// Panel index of the reference.
    pub reference: usize,
    /// 0-based position in the reference sequence as given.
    pub pos: usize,
    pub expect: Expect,
    /// Influence window, in reference bases relative to `pos`, inclusive.
    pub window: (i64, i64),
}

impl ModSite {
    fn covers(&self, ref_pos: usize) -> bool {
        let d = ref_pos as i64 - self.pos as i64;
        d >= self.window.0 && d <= self.window.1
    }
}

/// Events of one alignment, split by whether a known site explains them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModCounts {
    pub explained: u32,
    pub unexplained: u32,
}

/// The known sites of a panel, indexed by reference.
#[derive(Debug, Clone, Default)]
pub struct ModTable {
    by_ref: Vec<Vec<ModSite>>,
}

impl ModTable {
    /// `n_refs` is the panel size; every site's `reference` must be below it.
    pub fn new(n_refs: usize, sites: impl IntoIterator<Item = ModSite>) -> Self {
        let mut by_ref = vec![Vec::new(); n_refs];
        for s in sites {
            by_ref[s.reference].push(s);
        }
        Self { by_ref }
    }

    pub fn n_sites(&self) -> usize {
        self.by_ref.iter().map(Vec::len).sum()
    }

    /// Classify every event of `aln` against reference `reference`.
    ///
    /// `query` is the read in the orientation it aligned in and `ref_seq` the
    /// reference letters as given; `aln` indexes both.
    pub fn classify(
        &self,
        reference: usize,
        aln: &Alignment,
        query: &[u8],
        ref_seq: &[u8],
    ) -> ModCounts {
        let sites = &self.by_ref[reference];
        let mut out = ModCounts::default();
        let mut tally = |explained: bool| {
            if explained {
                out.explained += 1;
            } else {
                out.unexplained += 1;
            }
        };
        let (mut i, mut j) = (aln.query_start, aln.ref_start);
        for &(op, len) in &aln.ops {
            for _ in 0..len {
                match op {
                    CigarOp::Match => {
                        let q = query[i];
                        if !crate::sam::calmd_match(q, ref_seq[j]) {
                            tally(
                                sites
                                    .iter()
                                    .any(|s| s.covers(j) && s.expect.explains_mismatch(q)),
                            );
                        }
                        i += 1;
                        j += 1;
                    }
                    CigarOp::Ins => {
                        let anchor = if j > aln.ref_start { j - 1 } else { j };
                        tally(sites.iter().any(|s| s.covers(anchor) && s.expect.insertion));
                        i += 1;
                    }
                    CigarOp::Del => {
                        tally(sites.iter().any(|s| s.covers(j) && s.expect.deletion));
                        j += 1;
                    }
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn site(pos: usize, expect: &[&str], window: (i64, i64)) -> ModSite {
        let mut e = Expect::default();
        for s in expect {
            e.add(s).unwrap();
        }
        ModSite {
            reference: 0,
            pos,
            expect: e,
            window,
        }
    }

    fn aln(ops: &[(CigarOp, u32)], ref_len: usize) -> Alignment {
        Alignment {
            score: 0,
            query_start: 0,
            query_end: 0,
            ref_start: 0,
            ref_end: ref_len,
            ops: ops.to_vec(),
        }
    }

    fn counts(explained: u32, unexplained: u32) -> ModCounts {
        ModCounts {
            explained,
            unexplained,
        }
    }

    #[test]
    fn classifies_events_in_window() {
        let reference = b"ACGTACGTAC";
        let table = ModTable::new(
            1,
            [
                site(3, &["mismatch:C"], (-1, 1)),
                site(7, &["deletion", "insertion"], (0, 0)),
            ],
        );
        let m = CigarOp::Match;
        let run = |q: &[u8], ops: &[(CigarOp, u32)]| table.classify(0, &aln(ops, 10), q, reference);
        // T->C at the site itself.
        assert_eq!(run(b"ACGCACGTAC", &[(m, 10)]), counts(1, 0));
        // A->C one base inside the window.
        assert_eq!(run(b"ACGTCCGTAC", &[(m, 10)]), counts(1, 0));
        // C->G at 5, two bases from the site: outside the window.
        assert_eq!(run(b"ACGTAGGTAC", &[(m, 10)]), counts(0, 1));
        // Inside the window, but a base the site does not expect (T->A).
        assert_eq!(run(b"ACGAACGTAC", &[(m, 10)]), counts(0, 1));
        // Deleting refs 7 (explained) and 8 (outside the zero-width window).
        assert_eq!(
            run(b"ACGTACGC", &[(m, 7), (CigarOp::Del, 2), (m, 1)]),
            counts(1, 1)
        );
        // An insertion after ref 7 anchors at 7; after ref 6 it anchors at 6.
        assert_eq!(
            run(b"ACGTACGTGAC", &[(m, 8), (CigarOp::Ins, 1), (m, 2)]),
            counts(1, 0)
        );
        assert_eq!(
            run(b"ACGTACGGTAC", &[(m, 7), (CigarOp::Ins, 1), (m, 3)]),
            counts(0, 1)
        );
        // A clean match has no events.
        assert_eq!(run(reference, &[(m, 10)]), counts(0, 0));
    }

    #[test]
    fn expect_parsing() {
        for bad in ["mismatch:", "mismatch:AX", "bogus"] {
            assert!(Expect::default().add(bad).is_err(), "{bad}");
        }
        let mut e = Expect::default();
        e.add("mismatch:CU").unwrap();
        assert!(e.explains_mismatch(b'c') && e.explains_mismatch(b'T'));
        assert!(!e.explains_mismatch(b'A') && !e.explains_mismatch(b'N'));
    }
}
