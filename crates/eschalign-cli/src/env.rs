//! One rule for `ESCHALIGN_*` boolean switches.
//!
//! A boolean switch that tests `var_os(..).is_some()` turns `VAR=0` — meant to
//! switch it off — *on*. [`flag`] is the one parse: on for `1`, `true`, `yes`
//! or `on` (case-insensitive); off when the variable is unset, empty, or
//! `0`, `false`, `no`, `off` (case-insensitive). Anything else warns once per
//! name and counts as off, so a typo cannot silently turn a switch on.
//!
//! `clippy.toml` disallows raw `std::env::var`/`var_os` everywhere else;
//! backend-name caps (`*_BACKEND`, parsed in the library) and third-party
//! convention variables (`NO_COLOR`, `RUST_LOG`, `RAYON_NUM_THREADS`, …) are a
//! different shape and keep their own parse behind a narrow `allow`.

use std::collections::HashSet;
use std::sync::Mutex;

/// Every name this process has already warned about, so a switch read more
/// than once does not repeat the same warning.
static WARNED: Mutex<Option<HashSet<String>>> = Mutex::new(None);

fn warn_once(name: &str, message: std::fmt::Arguments<'_>) {
    let mut warned = WARNED.lock().unwrap_or_else(|e| e.into_inner());
    if warned
        .get_or_insert_with(HashSet::new)
        .insert(name.to_string())
    {
        tracing::warn!("{name}: {message}");
    }
}

/// A boolean `ESCHALIGN_*` switch. See the module docs for the contract.
// This module is the one home `clippy.toml` sends every switch to; the raw
// `std::env::var` call below is that parse, not a copy of it.
#[allow(clippy::disallowed_methods)]
pub fn flag(name: &str) -> bool {
    match std::env::var(name) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "false" | "no" | "off" => false,
            "1" | "true" | "yes" | "on" => true,
            _ => {
                warn_once(
                    name,
                    format_args!(
                        "{v:?} is not 1/true/yes/on or 0/false/no/off; treating it as off"
                    ),
                );
                false
            }
        },
        Err(std::env::VarError::NotPresent) => false,
        Err(std::env::VarError::NotUnicode(_)) => {
            warn_once(
                name,
                format_args!("value is not valid Unicode; treating it as off"),
            );
            false
        }
    }
}

/// [`flag`] for a switch that has a current name and a deprecated alias: the
/// current name wins whenever it is *set* (even to `0`), otherwise the alias
/// is read. Returns the value and the name that supplied it, when either is on.
#[allow(clippy::disallowed_methods)]
pub fn flag_with_alias(name: &str, deprecated: &str) -> bool {
    if std::env::var_os(name).is_some() {
        return flag(name);
    }
    let on = flag(deprecated);
    if on {
        warn_once(
            deprecated,
            format_args!("is a deprecated alias; use {name} instead"),
        );
    }
    on
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// `std::env` is process-global, so tests that set a variable serialize on
    /// this lock rather than racing each other under nextest's threads.
    static ENV_LOCK: StdMutex<()> = StdMutex::new(());

    // Test-only raw env access: saving/restoring the ambient values to test
    // `flag` itself, not a second parse of a switch.
    #[allow(clippy::disallowed_methods)]
    fn with_vars<T>(vars: &[(&str, Option<&str>)], f: impl FnOnce() -> T) -> T {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let previous: Vec<_> = vars.iter().map(|(n, _)| std::env::var(n).ok()).collect();
        for (name, value) in vars {
            match value {
                Some(v) => unsafe { std::env::set_var(name, v) },
                None => unsafe { std::env::remove_var(name) },
            }
        }
        let result = f();
        for ((name, _), prev) in vars.iter().zip(previous) {
            match prev {
                Some(v) => unsafe { std::env::set_var(name, v) },
                None => unsafe { std::env::remove_var(name) },
            }
        }
        result
    }

    fn with_var<T>(name: &str, value: Option<&str>, f: impl FnOnce() -> T) -> T {
        with_vars(&[(name, value)], f)
    }

    #[test]
    fn every_truthy_spelling_is_on() {
        for v in ["1", "true", "TRUE", "True", "yes", "YES", "on", "ON"] {
            assert!(
                with_var("ESCHALIGN_TEST_TRUE", Some(v), || flag(
                    "ESCHALIGN_TEST_TRUE"
                )),
                "{v:?} should be on"
            );
        }
    }

    #[test]
    fn every_falsy_spelling_is_off() {
        for v in ["0", "false", "FALSE", "False", "no", "NO", "off", "OFF"] {
            assert!(
                !with_var("ESCHALIGN_TEST_FALSE", Some(v), || flag(
                    "ESCHALIGN_TEST_FALSE"
                )),
                "{v:?} should be off"
            );
        }
    }

    #[test]
    fn unset_and_empty_are_off() {
        assert!(!with_var("ESCHALIGN_TEST_UNSET", None, || flag(
            "ESCHALIGN_TEST_UNSET"
        )));
        assert!(!with_var("ESCHALIGN_TEST_EMPTY", Some(""), || flag(
            "ESCHALIGN_TEST_EMPTY"
        )));
    }

    #[test]
    fn garbage_warns_and_is_off() {
        assert!(!with_var("ESCHALIGN_TEST_GARBAGE", Some("maybe"), || flag(
            "ESCHALIGN_TEST_GARBAGE"
        )));
    }

    /// The bug this module exists to prevent: `=0` must not count as "set".
    #[test]
    fn zero_is_off_not_on() {
        assert!(!with_var("ESCHALIGN_TEST_ZERO", Some("0"), || flag(
            "ESCHALIGN_TEST_ZERO"
        )));
    }

    #[test]
    fn current_name_wins_over_alias() {
        const NEW: &str = "ESCHALIGN_TEST_ALIAS_NEW";
        const OLD: &str = "ESCHALIGN_TEST_ALIAS_OLD";
        let run = |new, old| with_vars(&[(NEW, new), (OLD, old)], || flag_with_alias(NEW, OLD));
        // Alias alone is honoured.
        assert!(run(None, Some("1")));
        // Current name, even set to 0, beats an alias that is on.
        assert!(!run(Some("0"), Some("1")));
        assert!(run(Some("1"), Some("0")));
    }
}
