// SPDX-License-Identifier: MIT

//! `--site-table` (#19): the per-position pileup.
//!
//! * `counts_match_pileup_golden` — the table equals a pileup rebuilt in this
//!   test from the output BAM (CIGAR + SEQ + the reference FASTA), and equals
//!   the committed golden `fixtures/site_table_golden.tsv.gz`
//!   (`ESCHALIGN_BLESS=1` rewrites it).
//! * `table_is_order_independent` — thread count, batch size and `--sort`
//!   leave the table byte-identical.
//! * `site_table_leaves_bam_unchanged` — the BAM records with and without the
//!   flag are the same bytes.

use flate2::read::MultiGzDecoder;
use noodles_bam as bam;
use noodles_sam::alignment::record::Flags;
use noodles_sam::alignment::record::cigar::op::Kind;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn input_bam() -> PathBuf {
    fixtures().join("trna_mappings_padded.bam")
}

fn reference() -> PathBuf {
    fixtures().join("trna_reference.fa")
}

/// Run `eschalign` on the fixture reads with `extra` flags.
fn align(out: &Path, extra: &[&str]) {
    let o = Command::new(env!("CARGO_BIN_EXE_eschalign"))
        .arg(input_bam())
        .arg("-r")
        .arg(reference())
        .arg("-o")
        .arg(out)
        .args(extra)
        .output()
        .expect("eschalign should launch");
    assert!(
        o.status.success(),
        "eschalign failed:\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
}

fn gunzip(path: &Path) -> String {
    let mut s = String::new();
    MultiGzDecoder::new(std::fs::File::open(path).unwrap())
        .read_to_string(&mut s)
        .expect("the site table is gzip");
    s
}

/// A table for `extra` flags, decompressed.
fn table(dir: &Path, tag: &str, extra: &[&str]) -> String {
    let (bam, sites) = (
        dir.join(format!("{tag}.bam")),
        dir.join(format!("{tag}.tsv.gz")),
    );
    let mut args = vec!["--site-table", sites.to_str().unwrap()];
    args.extend_from_slice(extra);
    align(&bam, &args);
    gunzip(&sites)
}

fn fasta() -> Vec<(String, Vec<u8>)> {
    let text = std::fs::read_to_string(reference()).unwrap();
    let mut out: Vec<(String, Vec<u8>)> = Vec::new();
    for line in text.lines() {
        if let Some(h) = line.strip_prefix('>') {
            out.push((h.split_whitespace().next().unwrap().to_string(), Vec::new()));
        } else if let Some(last) = out.last_mut() {
            last.1
                .extend(line.trim().bytes().map(|b| b.to_ascii_uppercase()));
        }
    }
    out
}

/// Per (reference name, 0-based position): [depth, match, mismatch, del, ins,
/// uniq_depth, tied_depth], from the BAM alone.
fn oracle(bam_path: &Path) -> BTreeMap<(String, usize), [u32; 7]> {
    let refs = fasta();
    let mut r = bam::io::Reader::new(std::fs::File::open(bam_path).unwrap());
    let header = r.read_header().unwrap();
    let mut t: BTreeMap<(String, usize), [u32; 7]> = BTreeMap::new();
    for rec in r.record_bufs(&header) {
        let rec = rec.unwrap();
        if rec.flags().is_unmapped() || rec.flags().intersects(Flags::SECONDARY) {
            continue;
        }
        let ri = rec.reference_sequence_id().unwrap();
        let (name, seq) = &refs[ri];
        let uniq = u8::from(rec.mapping_quality().unwrap()) == 60;
        let (mut rp, mut qp) = (usize::from(rec.alignment_start().unwrap()) - 1, 0);
        let mut last = None;
        let read = rec.sequence().as_ref().to_vec();
        for op in rec.cigar().as_ref() {
            for i in 0..op.len() {
                match op.kind() {
                    Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                        let c = t.entry((name.clone(), rp)).or_default();
                        let same = seq[rp].replace_u() == read[qp].to_ascii_uppercase().replace_u();
                        c[0] += 1;
                        c[if same { 1 } else { 2 }] += 1;
                        c[if uniq { 5 } else { 6 }] += 1;
                        last = Some(rp);
                        rp += 1;
                        qp += 1;
                    }
                    Kind::Deletion => {
                        let c = t.entry((name.clone(), rp)).or_default();
                        c[0] += 1;
                        c[3] += 1;
                        c[if uniq { 5 } else { 6 }] += 1;
                        last = Some(rp);
                        rp += 1;
                    }
                    Kind::Insertion => {
                        if let (0, Some(p)) = (i, last) {
                            t.entry((name.clone(), p)).or_default()[4] += 1;
                        }
                        qp += 1;
                    }
                    Kind::SoftClip => qp += 1,
                    _ => {}
                }
            }
        }
    }
    t
}

trait ReplaceU {
    fn replace_u(self) -> u8;
}
impl ReplaceU for u8 {
    fn replace_u(self) -> u8 {
        if self == b'U' { b'T' } else { self }
    }
}

#[test]
fn counts_match_pileup_golden() {
    let dir = tempfile::tempdir().unwrap();
    let got = table(dir.path(), "a", &[]);
    let mut lines = got.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let col = |n: &str| header.iter().position(|h| *h == n).unwrap();
    let mm: Vec<usize> = ["mm_A", "mm_C", "mm_G", "mm_T", "mm_N"]
        .iter()
        .map(|n| col(n))
        .collect();

    // Against the pileup rebuilt from the BAM.
    let want = oracle(&dir.path().join("a.bam"));
    let mut seen = 0;
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        let pos: usize = f[col("position")].parse().unwrap();
        let n = |c: usize| -> u32 { f[c].parse().unwrap() };
        let w = want
            .get(&(f[0].to_string(), pos - 1))
            .unwrap_or_else(|| panic!("row for an uncovered site: {l}"));
        let got = [
            n(col("depth")),
            n(col("match")),
            mm.iter().map(|&c| n(c)).sum(),
            n(col("del")),
            n(col("ins")),
            n(col("uniq_depth")),
            n(col("tied_depth")),
        ];
        assert_eq!(&got, w, "{l}");
        // The quality histograms partition matches and mismatches.
        let m: u32 = (col("m_q0_9")..=col("m_qna")).map(n).sum();
        let x: u32 = (col("x_q0_9")..=col("x_qna")).map(n).sum();
        assert_eq!((m, x), (got[1], got[2]), "{l}");
        seen += 1;
    }
    assert_eq!(seen, want.values().filter(|c| c[0] > 0).count());
    assert!(seen > 100, "the fixture should cover many sites: {seen}");
    assert!(
        want.values().any(|c| c[4] > 0) && want.values().any(|c| c[3] > 0),
        "the fixture should exercise insertions and deletions"
    );

    // Against the committed golden.
    let golden = fixtures().join("site_table_golden.tsv.gz");
    // A test-only switch, not a runtime knob: any value blesses.
    #[allow(clippy::disallowed_methods)]
    let bless = std::env::var_os("ESCHALIGN_BLESS").is_some();
    if bless {
        std::fs::copy(dir.path().join("a.tsv.gz"), &golden).unwrap();
        return;
    }
    assert_eq!(got, gunzip(&golden), "the site table changed");
}

#[test]
fn table_is_order_independent() {
    let dir = tempfile::tempdir().unwrap();
    let base = table(dir.path(), "base", &["-t", "1"]);
    for (tag, extra) in [
        ("t4", &["-t", "4", "--batch-size", "7"][..]),
        ("t8", &["-t", "8", "--batch-size", "3"]),
        ("sorted", &["-t", "4", "--sort", "coordinate"]),
        ("both", &["--strand", "both", "-t", "3"]),
    ] {
        if tag == "both" {
            // A different strand mode may place reads differently; compare it
            // with itself under another thread count instead.
            let again = table(dir.path(), "both2", &["--strand", "both", "-t", "1"]);
            assert_eq!(table(dir.path(), tag, extra), again, "{tag}");
            continue;
        }
        assert_eq!(table(dir.path(), tag, extra), base, "{tag}");
    }
}

#[test]
fn site_table_leaves_bam_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let plain = dir.path().join("plain.bam");
    align(&plain, &[]);
    table(dir.path(), "with", &[]);
    let records = |p: &Path| -> Vec<u8> {
        let mut r = bam::io::Reader::new(std::fs::File::open(p).unwrap());
        r.read_header().unwrap();
        let mut v = Vec::new();
        r.get_mut().read_to_end(&mut v).unwrap();
        v
    };
    assert_eq!(records(&plain), records(&dir.path().join("with.bam")));
}

#[test]
fn parquet_path_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let o = Command::new(env!("CARGO_BIN_EXE_eschalign"))
        .arg(input_bam())
        .arg("-r")
        .arg(reference())
        .arg("-o")
        .arg(dir.path().join("o.bam"))
        .arg("--site-table")
        .arg(dir.path().join("s.parquet"))
        .output()
        .unwrap();
    assert!(!o.status.success());
}
