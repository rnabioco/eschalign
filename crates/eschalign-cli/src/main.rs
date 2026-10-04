//! `eschalign` — all-vs-all alignment of reads to a small reference panel
//! (tRNA). One flat binary: `eschalign reads.ubam -r ref.fa -o out.bam`.
//! See `align.rs` for the pipeline and the library crate (`eschalign`) for the
//! DP and kernels.

use clap::Parser;
use clap::builder::styling::{AnsiColor, Effects, Styles};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::format::{self, FormatEvent, FormatFields};
use tracing_subscriber::registry::LookupSpan;

// mimalloc is per-thread by design and keeps Rust allocations off whatever
// libc malloc the binary links. escapepod-rs measured the static-musl release
// of this pipeline's parent at 2.35x the wall and 3.9x the CPU of a glibc
// build of the same code on an allocation-heavy command (musl's old allocator
// takes one global lock per call), and mimalloc worth a further ~10% of CPU on
// glibc. A feature, so the two allocators can still be measured against each
// other; on by default because that is what production runs.
#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod align;
mod device;
mod env;
mod progress;
mod style;
mod threads;
mod uuid_util;

/// Terse event formatter: `YYYY-MM-DD HH:MM:SS  LEVEL [target] message`.
///
/// The binary's own events (targets `eschalign_cli` / `eschalign_cli::*`) are
/// the primary status channel, so their target is omitted to keep status lines
/// clean. The library's (`eschalign::*`) are shown without the crate prefix;
/// all other (external) targets are printed verbatim so `-v` can attribute them.
struct EschalignFormatter;

impl<S, N> FormatEvent<S, N> for EschalignFormatter
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static,
{
    fn format_event(
        &self,
        ctx: &tracing_subscriber::fmt::FmtContext<'_, S, N>,
        mut writer: format::Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        let now = chrono::Local::now();
        write!(writer, "{}", now.format("%Y-%m-%d %H:%M:%S"))?;
        write!(writer, " {:>5}", event.metadata().level())?;

        let target = event.metadata().target();
        if target == "eschalign_cli" || target.starts_with("eschalign_cli::") {
            // The binary's own status output — no target label.
        } else if let Some(module) = target.strip_prefix("eschalign::") {
            write!(writer, " [{module}]")?;
        } else if target != "eschalign" {
            write!(writer, " [{target}]")?;
        }

        write!(writer, " ")?;
        ctx.field_format().format_fields(writer.by_ref(), event)?;
        writeln!(writer)
    }
}

const STYLES: Styles = Styles::styled()
    .header(AnsiColor::Yellow.on_default().effects(Effects::BOLD))
    .usage(AnsiColor::Yellow.on_default().effects(Effects::BOLD))
    .literal(AnsiColor::Green.on_default().effects(Effects::BOLD))
    .placeholder(AnsiColor::Cyan.on_default());

#[derive(Parser)]
#[command(name = "eschalign")]
#[command(author, version, styles = STYLES)]
#[command(about = "All-vs-all alignment of reads to a small reference panel (tRNA)")]
#[command(after_help = "\
Examples:
  eschalign reads.ubam -r trna.fa -o aligned.bam
  eschalign reads.fastq.gz -r trna.fa -o - --sort coordinate > sorted.bam
  eschalign reads.ubam -r trna.fa -o aligned.bam --scoring 1,-1,-2,-1 --device gpu
")]
struct Cli {
    /// Silence all output except errors. Overrides `--verbose`.
    #[arg(short = 'q', long, global = true)]
    quiet: bool,

    /// Increase log verbosity. Status messages show by default (info);
    /// `-v` = debug, `-vv` = trace. `RUST_LOG` takes precedence when set.
    #[arg(short = 'v', long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(flatten)]
    args: align::AlignArgs,
}

/// Restore the default `SIGPIPE` disposition.
///
/// The Rust runtime ignores `SIGPIPE`, which turns a closed downstream reader
/// into an `EPIPE` surfaced as `Error: Broken pipe`. Resetting to `SIG_DFL`
/// makes `eschalign … -o - | head` terminate quietly like any other Unix tool.
#[cfg(unix)]
fn reset_sigpipe() {
    // SAFETY: `signal(2)` with `SIG_DFL` is async-signal-safe and called once,
    // before any threads are spawned.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn reset_sigpipe() {}

/// The filter directive for a verbosity: `level` for this workspace's own
/// crates, `warn` for every dependency. `-q` is a flat global `error`.
///
/// Every workspace crate is named here, and a new one must be added: a crate
/// left off this list is held at `warn` like a dependency, and its `info!`
/// lines vanish with nothing to say so.
fn filter_directive(quiet: bool, verbose: u8) -> String {
    let level = match (quiet, verbose) {
        (true, _) => "error",
        (_, 0) => "info",
        (_, 1) => "debug",
        _ => "trace",
    };
    if quiet {
        level.to_string()
    } else {
        format!("warn,eschalign={level},eschalign_cli={level}")
    }
}

fn main() -> anyhow::Result<()> {
    reset_sigpipe();

    let cli = Cli::parse();

    // `RUST_LOG` always wins if set. Status/progress output is emitted at INFO,
    // so INFO is the default level; `-q` drops to errors only.
    // RUST_LOG is the `tracing` convention variable, not an ESCHALIGN_* switch.
    tracing_subscriber::fmt()
        // Status lines carry ANSI from the `style::*` helpers (already
        // TTY/NO_COLOR-gated). tracing-subscriber 0.3.20+ sanitizes ANSI in the
        // `message` field by default, which would render those escapes as
        // literal `\x1b[...]` text; opt out so trusted color passes through.
        // Must precede `event_format` — the setter is only exposed while the
        // event format is still the default `Format`.
        .with_ansi_sanitization(false)
        .event_format(EschalignFormatter)
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(filter_directive(cli.quiet, cli.verbose))),
        )
        .with_writer(std::io::stderr)
        .init();

    // Size the rayon pool before any command code runs. rayon's global pool
    // can only be built once, and only before anything touches it — doing it
    // here is what keeps `--threads` from being silently ignored.
    threads::init(cli.args.threads)?;

    align::run(cli.args)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_scopes_level_to_own_crates() {
        assert_eq!(
            filter_directive(false, 0),
            "warn,eschalign=info,eschalign_cli=info"
        );
        assert_eq!(
            filter_directive(false, 1),
            "warn,eschalign=debug,eschalign_cli=debug"
        );
        assert_eq!(
            filter_directive(false, 2),
            "warn,eschalign=trace,eschalign_cli=trace"
        );
        assert_eq!(filter_directive(true, 2), "error");
    }

    #[test]
    fn cli_definition_is_consistent() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }
}
