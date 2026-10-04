//! Styling for status output (the slice of escapepod-rs's `style` that
//! `eschalign` uses). ANSI escapes are suppressed automatically when:
//!
//! - `NO_COLOR` is set (https://no-color.org/)
//! - `CLICOLOR=0` is set
//! - stderr is not a TTY (e.g. piped or redirected)
//!
//! `CLICOLOR_FORCE=1` overrides the TTY check and always emits ANSI.
//!
//! Everything here gates on **stderr**'s terminal status, right for status
//! lines emitted via `tracing`; stdout carries only the BAM under `-o -`.

use owo_colors::OwoColorize;
use std::fmt::Display;
use std::io::IsTerminal;
use std::sync::OnceLock;

/// `CLICOLOR_FORCE`/`NO_COLOR`/`CLICOLOR=0` precedence shared by both the
/// stderr gate below and the stdout gate in [`stdout`]. Returns `Some` when an
/// env var settles the question outright; `None` leaves it to the caller's
/// own terminal check.
fn color_env_override() -> Option<bool> {
    // CLICOLOR_FORCE/NO_COLOR/CLICOLOR are third-party CLI conventions, not
    // ESCHALIGN_* switches, and keep their own parse.
    // Force-on wins over everything.
    #[allow(clippy::disallowed_methods)]
    if matches!(
        std::env::var("CLICOLOR_FORCE").as_deref(),
        Ok(v) if v != "0" && !v.is_empty()
    ) {
        return Some(true);
    }
    // NO_COLOR: any value, including empty, disables color.
    #[allow(clippy::disallowed_methods)]
    if std::env::var_os("NO_COLOR").is_some() {
        return Some(false);
    }
    // CLICOLOR=0 disables.
    #[allow(clippy::disallowed_methods)]
    if matches!(std::env::var("CLICOLOR").as_deref(), Ok("0")) {
        return Some(false);
    }
    None
}

static USE_COLOR: OnceLock<bool> = OnceLock::new();

fn use_color() -> bool {
    *USE_COLOR.get_or_init(|| {
        // Status prints go to stderr; gate on stderr's terminal status.
        color_env_override().unwrap_or_else(|| std::io::stderr().is_terminal())
    })
}

/// Style for file paths
pub fn path<T: Display>(s: T) -> String {
    if use_color() {
        format!("{}", s.cyan())
    } else {
        s.to_string()
    }
}

/// Style for important counts/numbers (matched reads, etc.)
pub fn count<T: Display>(n: T) -> String {
    if use_color() {
        format!("{}", n.green())
    } else {
        n.to_string()
    }
}

/// Style for labels like "Output:", "Filter:"
pub fn label<T: Display>(s: T) -> String {
    if use_color() {
        format!("{}", s.bold())
    } else {
        s.to_string()
    }
}
