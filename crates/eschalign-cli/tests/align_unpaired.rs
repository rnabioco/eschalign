// SPDX-License-Identifier: MIT

//! `unpaired_output_unchanged`: the unpaired path must write the same bytes it
//! wrote before paired-end support existed.
//!
//! The golden (`fixtures/unpaired_golden.tsv`) was measured with the binary at
//! the commit before `feat/paired-end`: for each flag combination, the record
//! count and an FNV-1a 64 digest of the *uncompressed* BAM record stream (the
//! header is skipped — its `@PG CL` carries the temporary paths of the run).
//! `ESCHALIGN_BLESS=1` rewrites it; do that only when an unpaired change is
//! intended, and say so in the commit.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

const CONFIGS: &[(&str, &[&str])] = &[
    ("default", &[]),
    (
        "semiglobal_both_secondary",
        &["--mode", "semiglobal", "--strand", "both", "--secondary"],
    ),
    ("sort_coordinate", &["--sort", "coordinate"]),
    (
        "min_score_max_ties",
        &["--min-score", "60", "--max-ties", "1"],
    ),
    (
        "scoring_bwa",
        &["--scoring", "1,-1,-2,-1", "--strand", "both"],
    ),
    ("max_read_len", &["--max-read-len", "60"]),
];

/// `(records, digest)` of a BAM's record stream, header skipped.
fn digest(path: &Path) -> (usize, u64) {
    let mut raw = Vec::new();
    noodles_bgzf::io::Reader::new(std::fs::File::open(path).unwrap())
        .read_to_end(&mut raw)
        .unwrap();
    let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap()) as usize;
    assert_eq!(&raw[..4], b"BAM\x01");
    let mut off = 8 + u32_at(4);
    let n_ref = u32_at(off);
    off += 4;
    for _ in 0..n_ref {
        off += 4 + u32_at(off) + 4;
    }
    let body = &raw[off..];
    let (mut n, mut p) = (0, 0);
    while p < body.len() {
        p += 4 + u32::from_le_bytes(body[p..p + 4].try_into().unwrap()) as usize;
        n += 1;
    }
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in body {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    (n, h)
}

#[test]
fn unpaired_output_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let mut got = String::new();
    for (label, extra) in CONFIGS {
        let out = dir.path().join(format!("{label}.bam"));
        let o = Command::new(env!("CARGO_BIN_EXE_eschalign"))
            .arg(fixtures().join("trna_mappings_padded.bam"))
            .arg("-r")
            .arg(fixtures().join("trna_reference.fa"))
            .arg("-o")
            .arg(&out)
            .args(*extra)
            .output()
            .expect("eschalign should launch");
        assert!(
            o.status.success(),
            "{label}: {}",
            String::from_utf8_lossy(&o.stderr)
        );
        let (n, h) = digest(&out);
        got.push_str(&format!("{label}\t{n}\t{h:016x}\n"));
    }
    let golden = fixtures().join("unpaired_golden.tsv");
    // A test-only switch, not a runtime knob: any value blesses.
    #[allow(clippy::disallowed_methods)]
    let bless = std::env::var_os("ESCHALIGN_BLESS").is_some();
    if bless {
        std::fs::write(&golden, &got).unwrap();
        return;
    }
    let want = std::fs::read_to_string(&golden).expect("fixtures/unpaired_golden.tsv");
    assert_eq!(got, want, "the unpaired output changed");
}
