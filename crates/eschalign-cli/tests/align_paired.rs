// SPDX-License-Identifier: MIT

//! Paired-end `eschalign` (rnabioco/eschalign#1) over the synthesized fixture
//! in `fixtures/pairs/` (regenerate with `gen_pair_fixture.py`): a four-
//! reference panel — two paralogs `P1`/`P2` that differ at one base, and two
//! unrelated references — and one read pair per behaviour, each named for what
//! it pins. No real data, no pod5.
//!
//! Every test aligns with `--min-score 40` unless it says otherwise: semi-global
//! scoring (the paired default) gives any read some positive score somewhere, so
//! a floor is what makes a mate that matches nothing "not aligned".

use noodles_bam as bam;
use noodles_sam as sam;
use sam::alignment::RecordBuf;
use sam::alignment::io::Write as _;
use sam::alignment::record::Flags;
use sam::alignment::record::data::field::Tag;
use sam::alignment::record_buf::data::field::Value;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fx() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pairs")
}

fn panel() -> PathBuf {
    fx().join("pairs_panel.fa")
}

const STD: &[&str] = &["--min-score", "40"];

fn eschalign(args: &[&std::ffi::OsStr]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_eschalign"))
        .args(args)
        .output()
        .expect("eschalign should launch")
}

fn run_ok(args: &[&std::ffi::OsStr]) -> String {
    let o = eschalign(args);
    let stderr = String::from_utf8_lossy(&o.stderr).into_owned();
    assert!(o.status.success(), "eschalign failed:\n{stderr}");
    stderr
}

/// `eschalign -1 R1 -2 R2 -r panel -o out <extra>`, returning the records.
fn align_files(
    dir: &Path,
    tag: &str,
    r1: &Path,
    r2: &Path,
    extra: &[&str],
) -> (sam::Header, Vec<RecordBuf>) {
    let out = dir.join(format!("{tag}.bam"));
    let panel_path = panel();
    let mut args: Vec<&std::ffi::OsStr> = vec![
        "-1".as_ref(),
        r1.as_os_str(),
        "-2".as_ref(),
        r2.as_os_str(),
        "-r".as_ref(),
        panel_path.as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
    ];
    args.extend(extra.iter().map(|s| std::ffi::OsStr::new(*s)));
    run_ok(&args);
    read_bam(&out)
}

fn align_fixture(dir: &Path, tag: &str, extra: &[&str]) -> (sam::Header, Vec<RecordBuf>) {
    let mut all = STD.to_vec();
    all.extend_from_slice(extra);
    align_files(
        dir,
        tag,
        &fx().join("pairs_R1.fq"),
        &fx().join("pairs_R2.fq"),
        &all,
    )
}

fn read_bam(path: &Path) -> (sam::Header, Vec<RecordBuf>) {
    let mut r = bam::io::Reader::new(std::fs::File::open(path).unwrap());
    let header = r.read_header().unwrap();
    let recs = r
        .record_bufs(&header)
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    (header, recs)
}

fn name(r: &RecordBuf) -> String {
    r.name().map(|n| n.to_string()).unwrap_or_default()
}

fn ref_name(h: &sam::Header, r: &RecordBuf) -> Option<String> {
    r.reference_sequence_id()
        .map(|i| h.reference_sequences().get_index(i).unwrap().0.to_string())
}

fn cigar(r: &RecordBuf) -> String {
    use sam::alignment::record::cigar::op::Kind;
    r.cigar()
        .as_ref()
        .iter()
        .map(|op| {
            let c = match op.kind() {
                Kind::Match => 'M',
                Kind::Insertion => 'I',
                Kind::Deletion => 'D',
                Kind::SoftClip => 'S',
                _ => '?',
            };
            format!("{}{c}", op.len())
        })
        .collect()
}

fn string_tag(r: &RecordBuf, tag: &[u8; 2]) -> Option<String> {
    match r.data().get(&Tag::from(*tag)) {
        Some(Value::String(s)) => Some(s.to_string()),
        _ => None,
    }
}

fn int_tag(r: &RecordBuf, tag: &[u8; 2]) -> Option<i64> {
    r.data().get(&Tag::from(*tag)).and_then(|v| v.as_int())
}

fn flag(r: &RecordBuf) -> u16 {
    u16::from(r.flags())
}

fn pos(r: &RecordBuf) -> Option<usize> {
    r.alignment_start().map(|p| p.get())
}

/// The two records of the pair called `pair`, mate 1 then mate 2, which must
/// be adjacent and in that order.
fn mates<'a>(recs: &'a [RecordBuf], pair: &str) -> [&'a RecordBuf; 2] {
    let i = recs
        .iter()
        .position(|r| name(r) == pair && r.flags().is_first_segment())
        .unwrap_or_else(|| panic!("no mate 1 named {pair}"));
    let (a, b) = (&recs[i], &recs[i + 1]);
    assert_eq!(name(b), pair, "mates are adjacent");
    assert!(b.flags().is_last_segment(), "mate 2 follows mate 1");
    [a, b]
}

fn fasta_seq(refname: &str) -> String {
    let text = std::fs::read_to_string(panel()).unwrap();
    let mut it = text.lines();
    while let Some(l) = it.next() {
        if l == format!(">{refname}") {
            return it.next().unwrap().to_string();
        }
    }
    panic!("no reference {refname}");
}

/// The two mates' alignments, flags and mate fields, as one comparable line
/// each (everything but the `@PG`).
fn lines(recs: &[RecordBuf]) -> Vec<String> {
    recs.iter()
        .map(|r| {
            let mut tags: Vec<String> = r
                .data()
                .iter()
                .map(|(t, v)| format!("{}{}={v:?}", t.as_ref()[0] as char, t.as_ref()[1] as char))
                .collect();
            tags.sort();
            format!(
                "{} {} {:?} {:?} {:?} {} {:?} {:?} {} {} {} {}",
                name(r),
                flag(r),
                r.reference_sequence_id(),
                r.alignment_start(),
                r.mapping_quality(),
                cigar(r),
                r.mate_reference_sequence_id(),
                r.mate_alignment_start(),
                r.template_length(),
                String::from_utf8_lossy(r.sequence().as_ref()),
                r.quality_scores()
                    .as_ref()
                    .iter()
                    .map(|q| (q + 33) as char)
                    .collect::<String>(),
                tags.join(",")
            )
        })
        .collect()
}

#[test]
fn pair_resolves_paralog_tie() {
    let dir = tempfile::tempdir().unwrap();
    let (h, recs) = align_fixture(dir.path(), "paired", &[]);
    let [a, b] = mates(&recs, "paralog");
    for r in [a, b] {
        assert_eq!(ref_name(&h, r).as_deref(), Some("P1"), "{}", name(r));
        assert_eq!(r.mapping_quality().map(|q| q.get()), Some(60));
        assert!(string_tag(r, b"XA").is_none(), "a unique pair has no XA");
    }
    // Mate 1 alone is a tie between the paralogs: the unpaired aligner says so.
    let out = dir.path().join("single.bam");
    run_ok(&[
        fx().join("pairs_R1.fq").as_os_str(),
        "-r".as_ref(),
        panel().as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
        "--min-score".as_ref(),
        "40".as_ref(),
    ]);
    let (_, single) = read_bam(&out);
    let alone = single.iter().find(|r| name(r) == "paralog").unwrap();
    assert_eq!(alone.mapping_quality().map(|q| q.get()), Some(0));
    assert!(string_tag(alone, b"XA").is_some());
    // The pair's mate 1 still reports the tie it would have had alone as XS.
    assert_eq!(int_tag(a, b"XS"), int_tag(a, b"AS"));
    // Flags: paired, proper, mate 1 forward / mate 2 reverse.
    assert_eq!(flag(a), 0x1 | 0x2 | 0x20 | 0x40);
    assert_eq!(flag(b), 0x1 | 0x2 | 0x10 | 0x80);
    assert_eq!((pos(a), pos(b)), (Some(21), Some(91)));
    assert_eq!(
        (
            a.mate_alignment_start().map(|p| p.get()),
            a.template_length()
        ),
        (Some(91), 130)
    );
    assert_eq!(b.template_length(), -130);
    assert_eq!(string_tag(a, b"MC").as_deref(), Some("60M"));
    // Mate 2 is stored reverse-complemented with its qualities reversed.
    let (_, input) = {
        let text = std::fs::read_to_string(fx().join("pairs_R2.fq")).unwrap();
        let l: Vec<&str> = text.lines().collect();
        ((), (l[1].to_string(), l[3].to_string()))
    };
    let rev_q: String = input.1.chars().rev().collect();
    let got_q: String = b
        .quality_scores()
        .as_ref()
        .iter()
        .map(|q| (q + 33) as char)
        .collect();
    assert_eq!(got_q, rev_q);
}

#[test]
fn pair_tie_is_reported_in_xa_and_secondaries() {
    let dir = tempfile::tempdir().unwrap();
    let (h, recs) = align_fixture(dir.path(), "tie", &[]);
    let [a, b] = mates(&recs, "pair_tie");
    for r in [a, b] {
        assert_eq!(r.mapping_quality().map(|q| q.get()), Some(0));
        assert_eq!(ref_name(&h, r).as_deref(), Some("P1"));
        let xa = string_tag(r, b"XA").expect("XA");
        assert!(xa.starts_with("P2,"), "{xa}");
    }
    // The k-th XA entry of each mate belongs to the same other pair.
    let (_, sec) = align_fixture(dir.path(), "tie_sec", &["--secondary"]);
    let secs: Vec<&RecordBuf> = sec
        .iter()
        .filter(|r| name(r) == "pair_tie" && r.flags().is_secondary())
        .collect();
    assert_eq!(secs.len(), 2, "one secondary per mate per further pair");
    assert!(secs[0].flags().is_first_segment() && secs[1].flags().is_last_segment());
    assert_eq!(ref_name(&h, secs[0]).as_deref(), Some("P2"));
    assert_eq!(
        secs[0].mate_reference_sequence_id(),
        secs[1].reference_sequence_id()
    );
    assert!(secs[0].flags().is_properly_segmented());
}

#[test]
fn pair_insert_shorter_than_read() {
    let dir = tempfile::tempdir().unwrap();
    let (h, recs) = align_fixture(dir.path(), "short", &[]);
    let [a, b] = mates(&recs, "short_insert");
    assert_eq!(ref_name(&h, a).as_deref(), Some("U1"));
    // Insert 40, reads 60: TLEN is the insert, and each read's 20 adapter
    // bases are a soft clip on its 3' end (the start of the reverse mate).
    assert_eq!((a.template_length(), b.template_length()), (40, -40));
    assert_eq!(cigar(a), "40M20S");
    assert_eq!(cigar(b), "20S40M");
    assert_eq!((pos(a), pos(b)), (Some(31), Some(31)));
    assert!(a.flags().is_properly_segmented());
    // MC is the *mate's* CIGAR.
    assert_eq!(string_tag(a, b"MC").as_deref(), Some("20S40M"));
    assert_eq!(string_tag(b, b"MC").as_deref(), Some("40M20S"));
    // Nothing discarded: SEQ keeps the adapter.
    assert_eq!(a.sequence().len(), 60);
    assert_eq!(b.sequence().len(), 60);
}

#[test]
fn pair_adapter_tag_counts() {
    let dir = tempfile::tempdir().unwrap();
    let (_, recs) = align_fixture(dir.path(), "ad", &[]);
    let ad = |pair: &str| {
        let [a, b] = mates(&recs, pair);
        (int_tag(a, b"ad"), int_tag(b, b"ad"))
    };
    assert_eq!(ad("short_insert"), (Some(20), Some(20)));
    // One mate reads through: its own count, and 0 (present) on the other.
    assert_eq!(ad("one_mate_reads_through"), (Some(15), Some(0)));
    assert_eq!(ad("plain"), (Some(0), Some(0)));
    let [a, b] = mates(&recs, "one_mate_reads_through");
    assert_eq!(cigar(a), "40M15S");
    assert_eq!(cigar(b), "35M");
    assert_eq!(a.template_length(), 40);
    // --no-adapter-trim: no detection, no tag, and the adapter is aligned (or
    // overhangs) rather than clipped by us.
    let (_, off) = align_fixture(dir.path(), "noad", &["--no-adapter-trim"]);
    assert!(off.iter().all(|r| int_tag(r, b"ad").is_none()));
    // Custom adapters: an adapter that is not in the data finds nothing.
    let (_, custom) = align_fixture(
        dir.path(),
        "custom",
        &[
            "--adapter-r1",
            "TTTTTTTTTTTTTTTT",
            "--adapter-r2",
            "TTTTTTTTTTTTTTTT",
        ],
    );
    let [a, _] = mates(&custom, "short_insert");
    assert_eq!(int_tag(a, b"ad"), Some(0));
    // A raised minimum overlap hides the 15-base read-through.
    let (_, hi) = align_fixture(dir.path(), "minov", &["--min-adapter-overlap", "16"]);
    let [a, _] = mates(&hi, "one_mate_reads_through");
    assert_eq!(int_tag(a, b"ad"), Some(0));
}

#[test]
fn pair_terminal_mismatch_not_clipped() {
    let dir = tempfile::tempdir().unwrap();
    let (_, recs) = align_fixture(dir.path(), "tm", &[]);
    let [a, b] = mates(&recs, "terminal_mismatch");
    let u1 = fasta_seq("U1");
    // Mate 1: last base is a mismatch, kept — MD ends in it, NM counts it.
    assert_eq!(cigar(a), "60M");
    assert_eq!(int_tag(a, b"NM"), Some(1));
    let last = u1.as_bytes()[69] as char;
    assert_eq!(string_tag(a, b"MD"), Some(format!("59{last}0")));
    // Mate 2: its first base (the *last* of the reverse-complemented
    // alignment) is a mismatch.
    assert_eq!(cigar(b), "60M");
    assert_eq!(int_tag(b, b"NM"), Some(1));
    assert!(string_tag(b, b"MD").unwrap().ends_with('0'));
    // The unpaired default (--mode local) would clip it; the paired default
    // keeps it, and an explicit --mode local gives the clip back.
    let (_, local) = align_fixture(dir.path(), "tm_local", &["--mode", "local"]);
    let [a, _] = mates(&local, "terminal_mismatch");
    assert_eq!(cigar(a), "59M1S");
}

#[test]
fn pair_over_max_insert_not_proper() {
    let dir = tempfile::tempdir().unwrap();
    let (h, recs) = align_fixture(dir.path(), "wide", &["--max-insert", "120"]);
    let [a, b] = mates(&recs, "over_max_insert");
    assert!(!a.flags().is_properly_segmented() && !b.flags().is_properly_segmented());
    // Each mate at its own best reference, with the mate fields still set.
    assert_eq!(ref_name(&h, a).as_deref(), Some("U1"));
    assert_eq!(ref_name(&h, b).as_deref(), Some("U1"));
    assert_eq!(flag(a), 0x1 | 0x20 | 0x40);
    assert_eq!(flag(b), 0x1 | 0x10 | 0x80);
    assert_eq!(a.template_length(), 150);
    // The default --max-insert is 500: the same pair is proper.
    let (_, recs) = align_fixture(dir.path(), "roomy", &[]);
    let [a, b] = mates(&recs, "over_max_insert");
    assert!(a.flags().is_properly_segmented() && b.flags().is_properly_segmented());
}

#[test]
fn pair_wrong_orientation_not_proper() {
    let dir = tempfile::tempdir().unwrap();
    let (h, recs) = align_fixture(dir.path(), "ff", &[]);
    let [a, b] = mates(&recs, "wrong_orientation");
    assert!(!a.flags().is_properly_segmented() && !b.flags().is_properly_segmented());
    // Both mates sit where they are, forward, at their own best reference.
    assert_eq!(ref_name(&h, a).as_deref(), Some("U2"));
    assert_eq!(ref_name(&h, b).as_deref(), Some("U2"));
    assert_eq!(flag(a), 0x1 | 0x40);
    assert_eq!(flag(b), 0x1 | 0x80);
    // A mate that matches nothing is unmapped and rides at its partner's place.
    let [a, b] = mates(&recs, "one_unmapped");
    assert_eq!(flag(a), 0x1 | 0x8 | 0x40);
    assert_eq!(flag(b), 0x1 | 0x4 | 0x80);
    assert_eq!(b.reference_sequence_id(), a.reference_sequence_id());
    assert_eq!(pos(b), pos(a));
    assert_eq!(b.mapping_quality().map(|q| q.get()), Some(0));
}

fn fastq_records(path: &Path) -> Vec<(String, String, String)> {
    let text = std::fs::read_to_string(path).unwrap();
    let l: Vec<&str> = text.lines().collect();
    l.chunks(4)
        .map(|c| (c[0][1..].to_string(), c[1].to_string(), c[3].to_string()))
        .collect()
}

/// The fixture as an interleaved FASTQ and an unaligned BAM.
fn other_forms(dir: &Path) -> (PathBuf, PathBuf) {
    let (r1, r2) = (
        fastq_records(&fx().join("pairs_R1.fq")),
        fastq_records(&fx().join("pairs_R2.fq")),
    );
    let il = dir.join("pairs_interleaved.fq");
    let mut text = String::new();
    let ubam = dir.join("pairs.ubam");
    let header = sam::Header::default();
    let mut w = bam::io::Writer::new(std::fs::File::create(&ubam).unwrap());
    w.write_header(&header).unwrap();
    for (a, b) in r1.iter().zip(&r2) {
        for (m, (n, s, q)) in [a, b].into_iter().enumerate() {
            text.push_str(&format!("@{n}\n{s}\n+\n{q}\n"));
            let mut flags = Flags::SEGMENTED | Flags::UNMAPPED | Flags::MATE_UNMAPPED;
            flags |= if m == 0 {
                Flags::FIRST_SEGMENT
            } else {
                Flags::LAST_SEGMENT
            };
            let rec = RecordBuf::builder()
                .set_name(n.clone().into_bytes())
                .set_flags(flags)
                .set_sequence(s.clone().into_bytes().into())
                .set_quality_scores(q.bytes().map(|c| c - 33).collect::<Vec<u8>>().into())
                .build();
            w.write_alignment_record(&header, &rec).unwrap();
        }
    }
    w.try_finish().unwrap();
    std::fs::write(&il, text).unwrap();
    (il, ubam)
}

fn align_single_input(dir: &Path, tag: &str, reads: &Path, extra: &[&str]) -> Vec<RecordBuf> {
    let out = dir.join(format!("{tag}.bam"));
    let panel_path = panel();
    let mut args: Vec<&std::ffi::OsStr> = vec![
        reads.as_os_str(),
        "-r".as_ref(),
        panel_path.as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
    ];
    args.extend(STD.iter().map(|s| std::ffi::OsStr::new(*s)));
    args.extend(extra.iter().map(|s| std::ffi::OsStr::new(*s)));
    run_ok(&args);
    read_bam(&out).1
}

#[test]
fn fastq_pair_matches_interleaved_and_ubam() {
    let dir = tempfile::tempdir().unwrap();
    let (il, ubam) = other_forms(dir.path());
    for extra in [&[][..], &["--strand", "both", "--secondary"][..]] {
        let (_, files) = align_fixture(dir.path(), "files", extra);
        let inter =
            align_single_input(dir.path(), "il", &il, &[&["--interleaved"], extra].concat());
        let from_bam = align_single_input(dir.path(), "ubam", &ubam, extra);
        assert!(files.len() >= 18);
        assert_eq!(
            lines(&files),
            lines(&inter),
            "interleaved differs {extra:?}"
        );
        assert_eq!(lines(&files), lines(&from_bam), "uBAM differs {extra:?}");
    }
    // Gzipped mate files, with /1 and /2 on the names, give the same records.
    let gz = |src: &Path, suffix: &str, to: &Path| {
        let mut text = String::new();
        for (n, s, q) in fastq_records(src) {
            text.push_str(&format!("@{n}/{suffix} extra words\n{s}\n+\n{q}\n"));
        }
        let mut e = flate2::write::GzEncoder::new(
            std::fs::File::create(to).unwrap(),
            flate2::Compression::fast(),
        );
        std::io::Write::write_all(&mut e, text.as_bytes()).unwrap();
        e.finish().unwrap();
    };
    let (g1, g2) = (dir.path().join("a_1.fq.gz"), dir.path().join("a_2.fq.gz"));
    gz(&fx().join("pairs_R1.fq"), "1", &g1);
    gz(&fx().join("pairs_R2.fq"), "2", &g2);
    let mut all = STD.to_vec();
    all.push("--no-adapter-trim");
    let (_, suffixed) = align_files(dir.path(), "suffixed", &g1, &g2, &all);
    let (_, plain) = align_fixture(dir.path(), "plain", &["--no-adapter-trim"]);
    assert_eq!(lines(&suffixed), lines(&plain));
}

fn write_fastq(p: &Path, recs: &[(String, String, String)]) {
    let t: String = recs
        .iter()
        .map(|(n, s, q)| format!("@{n}\n{s}\n+\n{q}\n"))
        .collect();
    std::fs::write(p, t).unwrap();
}

/// Run eschalign expecting failure; its stderr.
fn fail(args: &[&std::ffi::OsStr]) -> String {
    let o = eschalign(args);
    assert!(!o.status.success(), "this run must fail");
    String::from_utf8_lossy(&o.stderr).into_owned()
}

#[test]
fn mate_name_mismatch_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let r1 = fastq_records(&fx().join("pairs_R1.fq"));
    let mut r2 = fastq_records(&fx().join("pairs_R2.fq"));
    let (full, sw) = (dir.path().join("full.fq"), dir.path().join("swapped.fq"));
    write_fastq(&full, &r1);
    r2.swap(0, 1);
    write_fastq(&sw, &r2);
    let out = dir.path().join("o.bam");
    // The first mismatch is named, with both files.
    let err = fail(&[
        "-1".as_ref(),
        full.as_os_str(),
        "-2".as_ref(),
        sw.as_os_str(),
        "-r".as_ref(),
        panel().as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
    ]);
    assert!(err.contains("paralog") && err.contains("pair_tie"), "{err}");
    assert!(
        err.contains("full.fq") && err.contains("swapped.fq"),
        "{err}"
    );
    // Interleaved: the neighbour is not the mate.
    let il = dir.path().join("il.fq");
    let recs: Vec<_> = r1
        .iter()
        .zip(&r2)
        .flat_map(|(a, b)| [a.clone(), b.clone()])
        .collect();
    write_fastq(&il, &recs);
    let err = fail(&[
        il.as_os_str(),
        "--interleaved".as_ref(),
        "-r".as_ref(),
        panel().as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
    ]);
    assert!(err.contains("paralog") && err.contains("no mate"), "{err}");
}

#[test]
fn non_adjacent_bam_mates_say_to_collate() {
    let dir = tempfile::tempdir().unwrap();
    let (_, ubam) = other_forms(dir.path());
    let (header, mut recs) = read_bam(&ubam);
    // Mate 1s first, then mate 2s: valid pairs, but not adjacent.
    recs.sort_by_key(|r| r.flags().is_last_segment());
    let far = dir.path().join("far.bam");
    let mut w = bam::io::Writer::new(std::fs::File::create(&far).unwrap());
    w.write_header(&header).unwrap();
    for r in &recs {
        w.write_alignment_record(&header, r).unwrap();
    }
    w.try_finish().unwrap();
    let err = fail(&[
        far.as_os_str(),
        "-r".as_ref(),
        panel().as_os_str(),
        "-o".as_ref(),
        dir.path().join("o.bam").as_os_str(),
    ]);
    assert!(err.contains("samtools collate"), "{err}");
}

#[test]
fn unequal_counts_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let r1 = fastq_records(&fx().join("pairs_R1.fq"));
    let r2 = fastq_records(&fx().join("pairs_R2.fq"));
    let (short, full) = (dir.path().join("short.fq"), dir.path().join("full.fq"));
    write_fastq(&short, &r2[..r2.len() - 1]);
    write_fastq(&full, &r1);
    let out = dir.path().join("o.bam");
    let last = &r1.last().unwrap().0;
    // Either file may be the short one; the read left over is named.
    for (a, b) in [(&full, &short), (&short, &full)] {
        let err = fail(&[
            "-1".as_ref(),
            a.as_os_str(),
            "-2".as_ref(),
            b.as_os_str(),
            "-r".as_ref(),
            panel().as_os_str(),
            "-o".as_ref(),
            out.as_os_str(),
        ]);
        assert!(
            err.contains(last.as_str()) && err.contains("no mate"),
            "{err}"
        );
    }
    // Interleaved with a trailing orphan.
    let il = dir.path().join("il.fq");
    let mut recs: Vec<_> = r1
        .iter()
        .zip(&r2)
        .flat_map(|(a, b)| [a.clone(), b.clone()])
        .collect();
    recs.push((
        "lonely".into(),
        "ACGTACGTACGTACGTACGT".into(),
        "I".repeat(20),
    ));
    write_fastq(&il, &recs);
    let err = fail(&[
        il.as_os_str(),
        "--interleaved".as_ref(),
        "-r".as_ref(),
        panel().as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
    ]);
    assert!(err.contains("lonely") && err.contains("no mate"), "{err}");
    // A paired BAM whose last record has lost its mate.
    let (_, ubam) = other_forms(dir.path());
    let (header, mut recs) = read_bam(&ubam);
    recs.pop();
    let lost = dir.path().join("lost.bam");
    let mut w = bam::io::Writer::new(std::fs::File::create(&lost).unwrap());
    w.write_header(&header).unwrap();
    for r in &recs {
        w.write_alignment_record(&header, r).unwrap();
    }
    w.try_finish().unwrap();
    let err = fail(&[
        lost.as_os_str(),
        "-r".as_ref(),
        panel().as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
    ]);
    assert!(
        err.contains("discordant") && err.contains("no mate"),
        "{err}"
    );
}

#[test]
fn pair_ubam_tags_are_copied_on_both_mates() {
    let dir = tempfile::tempdir().unwrap();
    let (_, ubam) = other_forms(dir.path());
    let (header, mut recs) = read_bam(&ubam);
    for (k, r) in recs.iter_mut().enumerate() {
        r.data_mut().insert(
            Tag::new(b'Z', b'Z'),
            Value::String(format!("tag{k}").into()),
        );
        // A stale tag the aligner owns must be replaced, not kept.
        r.data_mut().insert(Tag::new(b'N', b'M'), Value::UInt8(99));
    }
    let tagged = dir.path().join("tagged.bam");
    let mut w = bam::io::Writer::new(std::fs::File::create(&tagged).unwrap());
    w.write_header(&header).unwrap();
    for r in &recs {
        w.write_alignment_record(&header, r).unwrap();
    }
    w.try_finish().unwrap();
    let out = align_single_input(dir.path(), "tagged_out", &tagged, &[]);
    assert_eq!(out.len(), recs.len());
    for (k, r) in out.iter().enumerate() {
        assert_eq!(string_tag(r, b"ZZ"), Some(format!("tag{k}")));
        assert_ne!(int_tag(r, b"NM"), Some(99));
    }
}

#[test]
fn sort_coordinate_matches_samtools_and_fixmate() {
    let dir = tempfile::tempdir().unwrap();
    for extra in [&[][..], &["--secondary", "--strand", "both"][..]] {
        let (_, unsorted) = align_fixture(dir.path(), "u", extra);
        let mut sorted_args = extra.to_vec();
        sorted_args.extend(["--sort", "coordinate"]);
        let (sh, sorted) = align_fixture(dir.path(), "s", &sorted_args);
        // The model: a stable sort on (tid, pos, reverse), no reference last.
        let mut model = unsorted.clone();
        model.sort_by_key(|r| {
            (
                r.reference_sequence_id().unwrap_or(usize::MAX),
                r.alignment_start().map_or(0, |p| p.get()),
                r.flags().is_reverse_complemented(),
            )
        });
        assert_eq!(lines(&sorted), lines(&model), "model order {extra:?}");
        // Mates of a pair stay consistent: each record's mate fields point at
        // its partner's place, wherever the sort put them.
        for r in sorted.iter().filter(|r| !r.flags().is_secondary()) {
            let partner = sorted
                .iter()
                .find(|p| {
                    name(p) == name(r)
                        && !p.flags().is_secondary()
                        && p.flags().is_first_segment() != r.flags().is_first_segment()
                })
                .expect("partner");
            if !r.flags().is_mate_unmapped() {
                assert_eq!(r.mate_alignment_start(), partner.alignment_start());
                assert_eq!(
                    r.mate_reference_sequence_id(),
                    partner.reference_sequence_id()
                );
            }
        }
        assert!(sh.header().is_some());
        // And samtools itself, when there is one.
        let Some(st) = samtools() else {
            eprintln!(
                "[align_paired] sort_coordinate_matches_samtools_and_fixmate: no samtools; model only"
            );
            continue;
        };
        let theirs = dir.path().join("theirs.bam");
        let o = Command::new(st)
            .args(["sort", "--no-PG", "-o"])
            .arg(&theirs)
            .arg(dir.path().join("u.bam"))
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let (_, st_recs) = read_bam(&theirs);
        assert_eq!(lines(&sorted), lines(&st_recs), "samtools order {extra:?}");
        // fixmate round trip: name-sort our coordinate-sorted output, let
        // samtools recompute every mate field, and compare. It must have
        // nothing to change.
        if extra.contains(&"--secondary") {
            continue; // fixmate is defined for primary records
        }
        let sorted_path = dir.path().join("s.bam");
        let (byname, fixed) = (dir.path().join("n.bam"), dir.path().join("f.bam"));
        let st = samtools().unwrap();
        let o = Command::new(&st)
            .args(["sort", "-n", "--no-PG", "-o"])
            .arg(&byname)
            .arg(&sorted_path)
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let o = Command::new(&st)
            .args(["fixmate", "-m", "--no-PG", "-O", "bam"])
            .arg(&byname)
            .arg(&fixed)
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let (_, fixed) = read_bam(&fixed);
        let mate_fields = |r: &RecordBuf| {
            (
                name(r),
                r.flags().is_first_segment(),
                flag(r) & (0x1 | 0x4 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80),
                r.mate_reference_sequence_id(),
                r.mate_alignment_start(),
                r.template_length(),
                string_tag(r, b"MC"),
            )
        };
        let mut ours: Vec<_> = sorted.iter().map(mate_fields).collect();
        let mut theirs: Vec<_> = fixed.iter().map(mate_fields).collect();
        ours.sort();
        theirs.sort();
        assert_eq!(ours, theirs, "samtools fixmate disagrees {extra:?}");
    }
}

// SAMTOOLS and PATH are third-party variables, not ESCHALIGN_* knobs.
#[allow(clippy::disallowed_methods)]
fn samtools() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("SAMTOOLS") {
        return Some(PathBuf::from(p));
    }
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|d| d.join("samtools"))
        .find(|p| p.is_file())
}

/// The GPU and CPU scorers feed the same pair resolution, tracebacks and
/// record building, so the records must be identical.
#[cfg(feature = "gpu")]
#[test]
fn device_gpu_output_is_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    for (k, extra) in [
        &[][..],
        &["--strand", "both", "--secondary"][..],
        &["--max-insert", "120", "--batch-size", "4"][..],
        &["--no-adapter-trim", "--mode", "local"][..],
    ]
    .iter()
    .enumerate()
    {
        let (r1, r2, panel_path) = (fx().join("pairs_R1.fq"), fx().join("pairs_R2.fq"), panel());
        let mk = |device: &str, tag: &str| {
            let out = dir.path().join(format!("{tag}{k}.bam"));
            let mut args: Vec<&std::ffi::OsStr> = vec![
                "-1".as_ref(),
                r1.as_os_str(),
                "-2".as_ref(),
                r2.as_os_str(),
                "-r".as_ref(),
                panel_path.as_os_str(),
                "-o".as_ref(),
                out.as_os_str(),
                "--device".as_ref(),
                device.as_ref(),
            ];
            args.extend(STD.iter().map(|s| std::ffi::OsStr::new(*s)));
            args.extend(extra.iter().map(|s| std::ffi::OsStr::new(*s)));
            (eschalign(&args), out.clone())
        };
        let (g, gout) = mk("gpu", "gpu");
        let stderr = String::from_utf8_lossy(&g.stderr);
        if !g.status.success() {
            assert!(
                stderr.contains("--device gpu cannot run"),
                "GPU run failed for a reason other than a missing device:\n{stderr}"
            );
            eprintln!("[align_paired] skipping device_gpu_output_is_byte_identical: {stderr}");
            return;
        }
        assert!(stderr.contains("on GPU"), "{stderr}");
        let (c, cout) = mk("cpu", "cpu");
        assert!(c.status.success(), "{}", String::from_utf8_lossy(&c.stderr));
        let (_, cpu) = read_bam(&cout);
        let (_, gpu) = read_bam(&gout);
        assert!(!cpu.is_empty());
        assert_eq!(cpu, gpu, "flags {extra:?}");
    }
}

/// Chunks and batches are cut between pairs, never inside one: a few hundred
/// pairs through tiny batches on several threads come out in input order, mates
/// adjacent, record for record what one big batch gives.
#[test]
fn pair_chunking_keeps_mates_together() {
    let dir = tempfile::tempdir().unwrap();
    let refs: Vec<String> = ["P1", "P2", "U1", "U2"].map(fasta_seq).to_vec();
    let rc = |s: &str| -> String {
        s.bytes()
            .rev()
            .map(|b| match b {
                b'A' => 'T',
                b'C' => 'G',
                b'G' => 'C',
                _ => 'A',
            })
            .collect()
    };
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = |n: usize| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % n as u64) as usize
    };
    let (mut t1, mut t2) = (String::new(), String::new());
    for k in 0..300 {
        let r = &refs[next(refs.len())];
        let start = next(40);
        let len = 30 + next(40);
        let insert = &r[start..start + len + next(30)];
        let q = "I".repeat(len);
        t1.push_str(&format!("@p{k}\n{}\n+\n{q}\n", &insert[..len]));
        t2.push_str(&format!(
            "@p{k}\n{}\n+\n{q}\n",
            rc(&insert[insert.len() - len..])
        ));
    }
    let (f1, f2) = (dir.path().join("r1.fq"), dir.path().join("r2.fq"));
    std::fs::write(&f1, t1).unwrap();
    std::fs::write(&f2, t2).unwrap();
    let (_, big) = align_files(dir.path(), "big", &f1, &f2, STD);
    let (_, small) = align_files(
        dir.path(),
        "small",
        &f1,
        &f2,
        &[STD, &["--batch-size", "7", "-t", "4", "--secondary"]].concat(),
    );
    assert_eq!(big.len(), 600);
    let primary: Vec<RecordBuf> = small
        .iter()
        .filter(|r| !r.flags().is_secondary())
        .cloned()
        .collect();
    assert_eq!(lines(&big), lines(&primary));
    for (k, pair) in big.chunks(2).enumerate() {
        assert_eq!(name(&pair[0]), format!("p{k}"));
        assert_eq!(name(&pair[1]), format!("p{k}"));
        assert!(pair[0].flags().is_first_segment() && pair[1].flags().is_last_segment());
    }
}

/// Mate 1 alone ties between the paralogs and mate 2 alone separates them
/// (`paralog` in the fixture: mate 1 sits where `P1` and `P2` are identical,
/// mate 2 covers the one base that differs). Together they are a unique pair.
#[test]
fn joint_score_breaks_tie_a_mate_cannot() {
    let dir = tempfile::tempdir().unwrap();
    let (h, recs) = align_fixture(dir.path(), "joint", &[]);
    let [a, b] = mates(&recs, "paralog");
    // Alone, each mate as the unpaired aligner sees it (both strands).
    let alone = |reads: &str, tag: &str| {
        let out = dir.path().join(format!("{tag}.bam"));
        run_ok(&[
            fx().join(reads).as_os_str(),
            "-r".as_ref(),
            panel().as_os_str(),
            "-o".as_ref(),
            out.as_os_str(),
            "--min-score".as_ref(),
            "40".as_ref(),
            "--mode".as_ref(),
            "semiglobal".as_ref(),
            "--strand".as_ref(),
            "both".as_ref(),
        ]);
        let (_, r) = read_bam(&out);
        r.into_iter().find(|r| name(r) == "paralog").unwrap()
    };
    let m1 = alone("pairs_R1.fq", "m1");
    let m2 = alone("pairs_R2.fq", "m2");
    assert_eq!(
        m1.mapping_quality().map(|q| q.get()),
        Some(0),
        "mate 1 ties"
    );
    assert!(string_tag(&m1, b"XA").is_some());
    assert_eq!(
        m2.mapping_quality().map(|q| q.get()),
        Some(60),
        "mate 2 decides"
    );
    assert_eq!(ref_name(&h, &m2).as_deref(), Some("P1"));
    // Jointly: both mates on P1, MAPQ 60, no XA, and each carries the other's
    // MAPQ in MQ and CIGAR in MC.
    for r in [a, b] {
        assert_eq!(ref_name(&h, r).as_deref(), Some("P1"));
        assert_eq!(r.mapping_quality().map(|q| q.get()), Some(60));
        assert!(string_tag(r, b"XA").is_none());
        assert_eq!(int_tag(r, b"MQ"), Some(60));
        assert_eq!(string_tag(r, b"MC").as_deref(), Some("60M"));
    }
}

#[test]
fn discordant_pair_flags_and_counts() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("d.bam");
    let (r1, r2) = (fx().join("pairs_R1.fq"), fx().join("pairs_R2.fq"));
    let stderr = run_ok(&[
        "-1".as_ref(),
        r1.as_os_str(),
        "-2".as_ref(),
        r2.as_os_str(),
        "-r".as_ref(),
        panel().as_os_str(),
        "-o".as_ref(),
        out.as_os_str(),
        "--min-score".as_ref(),
        "40".as_ref(),
    ]);
    let (h, recs) = read_bam(&out);
    let [a, b] = mates(&recs, "discordant");
    // Each mate at its own best reference, never forced onto one.
    assert_eq!(ref_name(&h, a).as_deref(), Some("U1"));
    assert_eq!(ref_name(&h, b).as_deref(), Some("U2"));
    assert_eq!(a.mate_reference_sequence_id(), b.reference_sequence_id());
    assert_eq!(b.mate_reference_sequence_id(), a.reference_sequence_id());
    assert_eq!(a.mate_alignment_start(), b.alignment_start());
    // Paired, mate 1 forward, mate 2 reverse, not proper, no TLEN.
    assert_eq!(flag(a), 0x1 | 0x20 | 0x40);
    assert_eq!(flag(b), 0x1 | 0x10 | 0x80);
    assert_eq!((a.template_length(), b.template_length()), (0, 0));
    assert_eq!(string_tag(a, b"MC").as_deref(), Some("60M"));
    assert_eq!(int_tag(a, b"MQ"), Some(60));
    // The run summary counts it (and only it: `wrong_orientation` is
    // non-proper too, but on one reference).
    assert!(
        stderr.contains("1 of the non-proper pairs are discordant"),
        "{stderr}"
    );
    let [c, d] = mates(&recs, "wrong_orientation");
    assert!(!c.flags().is_properly_segmented());
    assert_eq!(c.reference_sequence_id(), d.reference_sequence_id());
}

/// `wrong_orientation` is two mates on the same strand: not a pair under the
/// default `fr`, a proper one under `ff`.
#[test]
fn pair_orientation_ff_makes_same_strand_pair_proper() {
    let dir = tempfile::tempdir().unwrap();
    let (_, fr) = align_fixture(dir.path(), "fr", &[]);
    let [a, _] = mates(&fr, "wrong_orientation");
    assert!(!a.flags().is_properly_segmented());
    let (h, ff) = align_fixture(dir.path(), "ff", &["--pair-orientation", "ff"]);
    let [a, b] = mates(&ff, "wrong_orientation");
    assert_eq!(flag(a), 0x1 | 0x2 | 0x40);
    assert_eq!(flag(b), 0x1 | 0x2 | 0x80);
    assert_eq!(ref_name(&h, a), ref_name(&h, b));
    assert_eq!((a.template_length(), b.template_length()), (100, -100));
}

// ---------------------------------------------------------------------------
// Single-end input is unchanged
// ---------------------------------------------------------------------------

/// The single-end path must write the same bytes it wrote before paired-end
/// support existed. The golden (`fixtures/unpaired_golden.tsv`) was measured
/// with the binary at the commit before `feat/paired-end`: for each flag
/// combination, the record count and an FNV-1a 64 digest of the *uncompressed*
/// BAM record stream (the header is skipped; its `@PG CL` carries the
/// temporary paths of the run). `ESCHALIGN_BLESS=1` rewrites it; do that only
/// when a single-end change is intended, and say so in the commit.
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
fn single_end_output_unchanged() {
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
    assert_eq!(got, want, "the single-end output changed");
}
