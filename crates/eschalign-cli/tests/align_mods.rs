// SPDX-License-Identifier: MIT

//! `--mods`: `zk`/`zu` annotation over the fixture reads. The expected counts
//! come from an oracle in this file that rebuilds each read's events from the
//! output's own CIGAR/SEQ and the reference text, independently of the
//! library's classifier.

use noodles_bam as bam;
use noodles_sam as sam;
use sam::alignment::RecordBuf;
use sam::alignment::record::cigar::op::Kind;
use sam::alignment::record::data::field::Tag;
use sam::alignment::record_buf::data::field::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn run(out: &Path, extra: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_eschalign"))
        .arg(fixtures().join("trna_mappings_padded.bam"))
        .arg("-r")
        .arg(fixtures().join("trna_reference.fa"))
        .arg("-o")
        .arg(out)
        .args(extra)
        .output()
        .expect("eschalign should launch")
}

fn align(out: &Path, extra: &[&str]) {
    let o = run(out, extra);
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
}

fn read_bam(path: &Path) -> Vec<RecordBuf> {
    let mut r = bam::io::Reader::new(std::fs::File::open(path).unwrap());
    let header = r.read_header().unwrap();
    r.record_bufs(&header)
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

fn int(rec: &RecordBuf, t: &[u8; 2]) -> Option<i64> {
    rec.data().get(&Tag::new(t[0], t[1])).and_then(|v| match v {
        Value::Int8(x) => Some(*x as i64),
        Value::UInt8(x) => Some(*x as i64),
        Value::Int16(x) => Some(*x as i64),
        Value::UInt16(x) => Some(*x as i64),
        Value::Int32(x) => Some(*x as i64),
        Value::UInt32(x) => Some(*x as i64),
        _ => None,
    })
}

/// The panel as `(name, seq)`.
fn panel() -> Vec<(String, Vec<u8>)> {
    let text = std::fs::read_to_string(fixtures().join("trna_reference.fa")).unwrap();
    let mut v: Vec<(String, Vec<u8>)> = Vec::new();
    for l in text.lines() {
        if let Some(n) = l.strip_prefix('>') {
            v.push((n.split_whitespace().next().unwrap().to_string(), Vec::new()));
        } else {
            v.last_mut().unwrap().1.extend(l.trim().bytes());
        }
    }
    v
}

/// A site as the oracle sees it: reference, 1-based pos, mismatch bases,
/// deletion, insertion, window.
type OSite = (&'static str, usize, &'static str, bool, bool, (i64, i64));

/// The sites of `fixtures/mods_ecoli.toml`, transcribed by hand.
fn oracle_sites() -> Vec<OSite> {
    let r = "tRNA-Ala-GGC-1-1";
    vec![
        (r, 15, "C", false, false, (-1, 1)),
        (r, 18, "T", false, false, (-1, 1)),
        (r, 24, "AG", false, false, (0, 0)),
        (r, 40, "", true, true, (-2, 2)),
    ]
}

/// `(explained, unexplained)` for a mapped primary record, from CIGAR + SEQ +
/// the reference text (events found by comparing letters, not from MD).
fn oracle(rec: &RecordBuf, panel: &[(String, Vec<u8>)], sites: &[OSite]) -> (i64, i64) {
    let rid = rec.reference_sequence_id().unwrap();
    let (name, refseq) = &panel[rid];
    let seq: &[u8] = rec.sequence().as_ref();
    let mut j = usize::from(rec.alignment_start().unwrap()) - 1;
    let mut i = 0usize;
    let covers = |s: &OSite, ref0: usize| {
        s.0 == name && {
            let d = ref0 as i64 - (s.1 as i64 - 1);
            d >= s.5.0 && d <= s.5.1
        }
    };
    let (mut k, mut u) = (0, 0);
    let mut bump = |ok: bool| {
        if ok { k += 1 } else { u += 1 }
    };
    for op in rec.cigar().as_ref() {
        for _ in 0..op.len() {
            match op.kind() {
                Kind::SoftClip => i += 1,
                Kind::Match | Kind::SequenceMatch | Kind::SequenceMismatch => {
                    let (q, r) = (seq[i].to_ascii_uppercase(), refseq[j].to_ascii_uppercase());
                    if q != r {
                        bump(
                            sites
                                .iter()
                                .any(|s| covers(s, j) && s.2.as_bytes().contains(&q)),
                        );
                    }
                    i += 1;
                    j += 1;
                }
                Kind::Insertion => {
                    bump(sites.iter().any(|s| covers(s, j - 1) && s.4));
                    i += 1;
                }
                Kind::Deletion => {
                    bump(sites.iter().any(|s| covers(s, j) && s.3));
                    j += 1;
                }
                _ => unreachable!(),
            }
        }
    }
    (k, u)
}

/// The hand-checked read: `4S127M` at ref 13, NM 10.
fn is_hand(r: &RecordBuf) -> bool {
    let ops: Vec<_> = r
        .cigar()
        .as_ref()
        .iter()
        .map(|o| (o.kind(), o.len()))
        .collect();
    ops == [(Kind::SoftClip, 4), (Kind::Match, 127)] && int(r, b"NM") == Some(10)
}

fn sidecar() -> String {
    fixtures().join("mods_ecoli.toml").display().to_string()
}

/// `(sum zk, sum zu)` over the fixture, pinned.
const GOLDEN_TOTALS: (i64, i64) = (54, 1572);

#[test]
fn annotation_tags_match_golden() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("mods.bam");
    align(&out, &["--mods", &sidecar()]);
    let (panel, sites) = (panel(), oracle_sites());
    let (mut total_k, mut total_u) = (0, 0);
    let recs = read_bam(&out);
    assert!(!recs.is_empty());
    for r in &recs {
        if r.flags().is_unmapped() {
            assert!(int(r, b"zk").is_none() && int(r, b"zu").is_none());
            continue;
        }
        let want = oracle(r, &panel, &sites);
        let got = (int(r, b"zk").unwrap(), int(r, b"zu").unwrap());
        assert_eq!(got, want, "{:?}", r.name());
        total_k += got.0;
        total_u += got.1;
    }
    assert_eq!((total_k, total_u), GOLDEN_TOTALS);
    // The hand-checked read: `4S127M` at ref 13 has two explained events
    // (15 and 17) and, of its 10 NM, eight unexplained.
    let hand = recs
        .iter()
        .find(|r| is_hand(r))
        .expect("the hand-checked read");
    assert_eq!((int(hand, b"zk"), int(hand, b"zu")), (Some(2), Some(8)));
}

#[test]
fn shifted_sidecar_changes_classification() {
    let dir = tempfile::tempdir().unwrap();
    let text = std::fs::read_to_string(fixtures().join("mods_ecoli.toml")).unwrap();
    // Shift every site one base downstream.
    let shifted: String = text
        .lines()
        .map(|l| match l.strip_prefix("pos = ") {
            Some(p) => format!("pos = {}\n", p.trim().parse::<usize>().unwrap() + 1),
            None => format!("{l}\n"),
        })
        .collect();
    let (a, b, toml_b) = (
        dir.path().join("a.bam"),
        dir.path().join("b.bam"),
        dir.path().join("shifted.toml"),
    );
    std::fs::write(&toml_b, shifted).unwrap();
    align(&a, &["--mods", &sidecar()]);
    align(&b, &["--mods", toml_b.to_str().unwrap()]);
    let (ra, rb) = (read_bam(&a), read_bam(&b));
    let changed = ra
        .iter()
        .zip(&rb)
        .filter(|(x, y)| int(x, b"zk") != int(y, b"zk"))
        .count();
    assert!(
        changed > 0,
        "shifting the sites by one base changed nothing"
    );
    // The hand-checked read: site 18 -> 19 (window 18..20) loses the T at 17.
    let hand = |rs: &[RecordBuf]| {
        let r = rs.iter().find(|r| is_hand(r)).unwrap();
        (int(r, b"zk").unwrap(), int(r, b"zu").unwrap())
    };
    assert_eq!(hand(&ra), (2, 8));
    assert_eq!(hand(&rb), (1, 9));
    // Events never move, only the split.
    for (x, y) in ra.iter().zip(&rb) {
        let n = |r: &RecordBuf| int(r, b"zk").zip(int(r, b"zu")).map(|(k, u)| k + u);
        assert_eq!(n(x), n(y));
    }
}

#[test]
fn no_mods_output_unchanged() {
    // Without --mods there are no zk/zu tags, and with it everything else
    // (flags, MAPQ, CIGAR, AS/XS/XA/MD/NM and every input tag) is identical.
    let dir = tempfile::tempdir().unwrap();
    let (plain, with) = (dir.path().join("plain.bam"), dir.path().join("with.bam"));
    let flags = ["--secondary", "--strand", "both"];
    align(&plain, &flags);
    let mut f = flags.to_vec();
    let sc = sidecar();
    f.extend(["--mods", sc.as_str()]);
    align(&with, &f);
    let (p, w) = (read_bam(&plain), read_bam(&with));
    assert_eq!(p.len(), w.len());
    let (zk, zu) = (Tag::new(b'z', b'k'), Tag::new(b'z', b'u'));
    let mut annotated = 0;
    for (x, y) in p.iter().zip(&w) {
        assert!(x.data().get(&zk).is_none() && x.data().get(&zu).is_none());
        let mut y = y.clone();
        if y.data().get(&zk).is_some() {
            annotated += 1;
        }
        y.data_mut().remove(&zk);
        y.data_mut().remove(&zu);
        assert_eq!(x, &y);
    }
    assert!(annotated > 0);
}

#[test]
fn rejects_unknown_reference() {
    let dir = tempfile::tempdir().unwrap();
    let bad = dir.path().join("bad.toml");
    std::fs::write(
        &bad,
        "[[site]]\nref = \"nope\"\npos = 1\nexpect = [\"deletion\"]\n",
    )
    .unwrap();
    let o = run(
        &dir.path().join("o.bam"),
        &["--mods", bad.to_str().unwrap()],
    );
    assert!(!o.status.success());
    assert!(String::from_utf8_lossy(&o.stderr).contains("not in the panel"));
}

#[cfg(feature = "gpu")]
#[test]
fn device_gpu_tags_identical() {
    let dir = tempfile::tempdir().unwrap();
    let (c, g) = (dir.path().join("cpu.bam"), dir.path().join("gpu.bam"));
    align(&c, &["--device", "cpu", "--mods", &sidecar()]);
    let o = run(&g, &["--device", "gpu", "--mods", &sidecar()]);
    if !o.status.success() {
        let stderr = String::from_utf8_lossy(&o.stderr);
        assert!(stderr.contains("--device gpu cannot run"), "{stderr}");
        eprintln!("[align_mods] skipping device_gpu_tags_identical: {stderr}");
        return;
    }
    assert_eq!(read_bam(&c), read_bam(&g));
}
