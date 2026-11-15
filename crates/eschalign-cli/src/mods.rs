// SPDX-License-Identifier: MIT

//! The `--mods` sidecar: TOML parse and validation against the panel.
//!
//! ```toml
//! [[site]]
//! ref = "tRNA-Ala-GGC-1-1"
//! pos = 38                       # 1-based into the FASTA sequence as given
//! mod = "m1A"                    # free text, with the optional
//! sprinzl = "37"                 #   sprinzl, modomics_code, source
//! expect = ["mismatch:T", "deletion"]
//! window = [-1, 1]               # optional, default [-1, 1]
//! ```
//!
//! Every row states `expect` explicitly; nothing is inferred from `mod`.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use eschalign::Panel;
use eschalign::mods::{Expect, ModSite, ModTable};
use toml::Value;

const KEYS: [&str; 8] = [
    "ref",
    "pos",
    "sprinzl",
    "mod",
    "modomics_code",
    "expect",
    "window",
    "source",
];

/// Read `path` and validate every row against `panel`.
pub fn load(path: &Path, panel: &Panel) -> Result<ModTable> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading mods sidecar {}", path.display()))?;
    parse(&text, panel).with_context(|| format!("mods sidecar {}", path.display()))
}

pub fn parse(text: &str, panel: &Panel) -> Result<ModTable> {
    let doc: toml::Table = text.parse().context("not valid TOML")?;
    for k in doc.keys() {
        if k != "site" {
            bail!("unknown top-level key '{k}' (rows are `[[site]]` tables)");
        }
    }
    let rows = match doc.get("site") {
        Some(Value::Array(a)) => a.as_slice(),
        Some(_) => bail!("`site` must be an array of tables (`[[site]]`)"),
        None => &[],
    };
    let mut sites: Vec<ModSite> = Vec::with_capacity(rows.len());
    let mut seen = std::collections::HashSet::new();
    for (n, row) in rows.iter().enumerate() {
        let site = parse_row(row, panel).with_context(|| format!("site #{}", n + 1))?;
        if !seen.insert((site.reference, site.pos)) {
            bail!(
                "site #{}: duplicate row for {} position {}",
                n + 1,
                panel.get(site.reference).name,
                site.pos + 1
            );
        }
        sites.push(site);
    }
    Ok(ModTable::new(panel.len(), sites))
}

fn parse_row(row: &Value, panel: &Panel) -> Result<ModSite> {
    let t = row.as_table().ok_or_else(|| anyhow!("not a table"))?;
    for k in t.keys() {
        if !KEYS.contains(&k.as_str()) {
            bail!("unknown key '{k}'");
        }
    }
    let name = t
        .get("ref")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("`ref` (string) is required"))?;
    let reference = panel
        .references()
        .iter()
        .position(|r| r.name == name)
        .ok_or_else(|| anyhow!("reference '{name}' is not in the panel"))?;
    let len = panel.get(reference).seq.len();
    let pos = t
        .get("pos")
        .and_then(Value::as_integer)
        .ok_or_else(|| anyhow!("`pos` (integer, 1-based) is required"))?;
    if pos < 1 || pos as u64 > len as u64 {
        bail!("{name}: pos {pos} is outside 1..={len}");
    }
    let expect_raw = t
        .get("expect")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("`expect` (array of strings) is required"))?;
    if expect_raw.is_empty() {
        bail!("{name}:{pos}: `expect` is empty");
    }
    let mut expect = Expect::default();
    for e in expect_raw {
        let s = e
            .as_str()
            .ok_or_else(|| anyhow!("{name}:{pos}: `expect` entries are strings"))?;
        expect.add(s).map_err(|m| anyhow!("{name}:{pos}: {m}"))?;
    }
    let window = match t.get("window") {
        None => (-1, 1),
        Some(Value::Array(w)) => match w.as_slice() {
            [Value::Integer(lo), Value::Integer(hi)] if lo <= hi => (*lo, *hi),
            _ => bail!("{name}:{pos}: `window` must be [lo, hi] integers with lo <= hi"),
        },
        Some(_) => bail!("{name}:{pos}: `window` must be [lo, hi]"),
    };
    for k in ["sprinzl", "mod", "modomics_code", "source"] {
        if t.get(k).is_some_and(|v| !v.is_str()) {
            bail!("{name}:{pos}: `{k}` must be a string");
        }
    }
    Ok(ModSite {
        reference,
        pos: (pos - 1) as usize,
        expect,
        window,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panel() -> Panel {
        Panel::new([("r1", b"ACGTACGT".to_vec()), ("r2", b"AAAA".to_vec())]).unwrap()
    }

    #[test]
    fn parses_and_defaults_window() {
        let t = parse(
            "[[site]]\nref='r1'\npos=3\nmod='m1A'\nexpect=['mismatch:T']\n",
            &panel(),
        )
        .unwrap();
        assert_eq!(t.n_sites(), 1);
    }

    #[test]
    fn rejects_bad_rows() {
        let p = panel();
        for bad in [
            "[[site]]\nref='nope'\npos=1\nexpect=['deletion']",
            "[[site]]\nref='r2'\npos=5\nexpect=['deletion']",
            "[[site]]\nref='r2'\npos=0\nexpect=['deletion']",
            "[[site]]\nref='r2'\npos=1\nexpect=[]",
            "[[site]]\nref='r2'\npos=1",
            "[[site]]\nref='r2'\npos=1\nexpect=['mismatch:Z']",
            "[[site]]\nref='r2'\npos=1\nexpect=['deletion']\nwindow=[2,1]",
            "[[site]]\nref='r2'\npos=1\nexpect=['deletion']\ntypo=1",
            "[[site]]\nref='r2'\npos=1\nexpect=['deletion']\n[[site]]\nref='r2'\npos=1\nexpect=['deletion']",
        ] {
            assert!(parse(bad, &p).is_err(), "{bad}");
        }
    }
}
