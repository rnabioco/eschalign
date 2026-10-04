//! `--device auto|cpu|gpu` — where panel scoring runs, and why.
//!
//! The slice of escapepod-rs's `device` module that `eschalign` needs: one
//! stage (the CUDA score kernel), so no per-stage enum.
//!
//! The old failure this exists to prevent is *silent CPU*: a user who forgot a
//! flag, or ran a binary with no GPU code at all, got the CPU path and no
//! indication that anything was slower than it had to be. So the placement is
//! decided and reported before anything is read, and a CPU placement that has
//! a GPU-shaped cost says what that cost is and why it happened.
//!
//! # This module compiles in every build
//!
//! Nothing here is behind `#[cfg(feature = "…")]`, including the warning path:
//! a build without GPU code is exactly where a user is most likely to wonder
//! why scoring is slow, and `--device gpu` on it must produce a sentence
//! explaining that the feature is not compiled in — not
//! `error: unexpected argument '--device'`. Feature detection uses `cfg!(…)`,
//! which evaluates in all builds, rather than `#[cfg(…)]`, which deletes the
//! code that would have explained itself.

use std::fmt;

/// Where the user wants GPU-capable work to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum Device {
    /// GPU when one is actually usable; CPU otherwise. Placing never fails.
    #[default]
    Auto,
    /// CPU, even with a device visible.
    Cpu,
    /// GPU or fail. Not a preference — a requirement.
    Gpu,
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        })
    }
}

// The `--device` flag plus its `--cpu` alias. `--cpu` conflicts with
// `--device`, so there is never a question of which one won. (A plain comment,
// not a doc comment: clap would take a doc comment on a flattened struct as the
// whole command's `about`.)
#[derive(Debug, Default, clap::Args)]
pub struct DeviceArgs {
    /// Where panel scoring runs: `auto` (default), `cpu`, or `gpu`.
    ///
    /// `auto` uses the GPU when the `gpu` Cargo feature is compiled in and a
    /// CUDA device is visible (~2.6x wall, ~4x less CPU on a 3.3 M-read run).
    ///
    /// `gpu` is a requirement, not a request: if the feature is missing or no
    /// device is visible, the run fails instead of silently continuing on the
    /// CPU. `cpu` forces the CPU even with a device present. The output is
    /// byte-identical either way.
    #[arg(
        long,
        value_enum,
        value_name = "auto|cpu|gpu",
        conflicts_with = "cpu",
        help_heading = "Advanced Options"
    )]
    pub device: Option<Device>,

    /// Alias for `--device cpu`.
    #[arg(long, help_heading = "Advanced Options")]
    pub cpu: bool,
}

impl DeviceArgs {
    /// The device this run asked for, folding in the `--cpu` alias.
    ///
    /// Call once, early — placement and the CPU-cost warning both hang off the
    /// result, and the point of the warning is that it arrives at second one
    /// rather than after the run.
    pub fn resolve(&self) -> Device {
        if self.cpu {
            return Device::Cpu;
        }
        self.device.unwrap_or_default()
    }
}

/// Human-readable name of the GPU-capable stage, used in every message.
const LABEL: &str = "read alignment scoring (`eschalign`)";

/// The Cargo feature that compiles the GPU path in.
const FEATURE: &str = "gpu";

/// What scoring on the CPU costs when a GPU is there.
///
/// Decided by the rule that the GPU arm must win on wall at equal or lower
/// CPU, on the interleaved measurement in `benchmarks/README.md`: 3.3 M reads
/// on one gpu node (`-c 16`, A30), 184 s and 2,900 CPU-s scoring on the CPU
/// against ~70 s and ~705 CPU-s on the GPU, output byte-identical. It wins on
/// both, so `auto` takes it. Should that change (a much smaller panel, say),
/// `None` here is the whole of turning it off.
const CPU_COST: Option<&str> =
    Some("~2.6x the wall and ~4x the CPU of scoring on the GPU end-to-end");

/// Whether this binary contains the GPU path.
///
/// `cfg!` and not `#[cfg]` on purpose — see the module docs.
const fn compiled_in() -> bool {
    cfg!(feature = "gpu")
}

/// Whether `--device auto` should send scoring to the GPU.
const fn auto_prefers_gpu() -> bool {
    CPU_COST.is_some()
}

/// Whether at least one CUDA device is visible to this process. Always `false`
/// in a build with no `gpu` feature — there is no cudarc to ask.
fn cuda_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        eschalign::cuda::device_visible()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Why scoring ended up on the CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuReason {
    /// `--device cpu`.
    Requested,
    /// This binary has no GPU path.
    NotCompiledIn,
    /// GPU path is compiled in, but no CUDA device is visible.
    NoDevice,
    /// A device is usable; the CPU is simply faster here.
    FasterOnCpu,
}

/// Where scoring will run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Gpu,
    Cpu(CpuReason),
}

impl Placement {
    /// Convenience for the `if gpu { … } else { … }` branches at the call sites.
    pub fn is_gpu(self) -> bool {
        matches!(self, Self::Gpu)
    }
}

/// Decide where scoring runs under `device`.
///
/// Errors only under `--device gpu`, and only for the two causes it can settle
/// up front: feature missing, or no device.
pub fn place(device: Device) -> anyhow::Result<Placement> {
    match device {
        Device::Cpu => Ok(Placement::Cpu(CpuReason::Requested)),
        Device::Gpu => {
            if !compiled_in() {
                anyhow::bail!(
                    "--device gpu cannot run {LABEL}: this binary was built without the \
                     `{FEATURE}` Cargo feature, so it contains no GPU code for that stage. \
                     Rebuild with `--features {FEATURE}` or use `--device auto` to run on \
                     the CPU instead.",
                );
            }
            if !cuda_available() {
                anyhow::bail!(
                    "--device gpu cannot run {LABEL}: no CUDA device is visible. Check \
                     `nvidia-smi` and `CUDA_VISIBLE_DEVICES`; under SLURM the job needs \
                     a GPU allocation (`--gres=gpu:1`). Use `--device auto` to fall \
                     back to the CPU automatically.",
                );
            }
            Ok(Placement::Gpu)
        }
        Device::Auto => {
            if !auto_prefers_gpu() {
                return Ok(Placement::Cpu(CpuReason::FasterOnCpu));
            }
            if !compiled_in() {
                return Ok(Placement::Cpu(CpuReason::NotCompiledIn));
            }
            if !cuda_available() {
                return Ok(Placement::Cpu(CpuReason::NoDevice));
            }
            Ok(Placement::Gpu)
        }
    }
}

/// [`place`], plus the log line that is the whole point of the module.
///
/// A scoring stage that lands on the CPU says so, in one line, with the cause
/// distinguished — "not compiled in" and "no device" need different fixes, and
/// conflating them is how you get someone rebuilding a binary that was already
/// correct.
pub fn place_and_report(device: Device) -> anyhow::Result<Placement> {
    let placement = place(device)?;
    match placement {
        Placement::Gpu => {
            tracing::info!("{} {LABEL} on GPU", crate::style::label("Device:"));
        }
        Placement::Cpu(reason) => report_cpu(reason),
    }
    Ok(placement)
}

/// The CPU-cost warning. Split out so the wording lives in one place.
fn report_cpu(reason: CpuReason) {
    let Some(cost) = CPU_COST else {
        // The CPU is the right answer; nothing to warn about.
        tracing::debug!("{LABEL} runs on the CPU");
        return;
    };
    match reason {
        // Asked for explicitly. Still worth a line — a wrapper script may be
        // passing `--device cpu` the user has forgotten about — but not a
        // warning, because nothing is wrong.
        CpuReason::Requested => tracing::info!(
            "{} {LABEL} on CPU ({cost}) — requested with `--device cpu`",
            crate::style::label("Device:"),
        ),
        CpuReason::NotCompiledIn => tracing::warn!(
            "{LABEL} is running on the CPU ({cost}); this build has no `{FEATURE}` feature. \
             Rebuild with `--features {FEATURE}` for GPU scoring, or pass `--device cpu` \
             to silence this.",
        ),
        CpuReason::NoDevice => tracing::warn!(
            "{LABEL} is running on the CPU ({cost}); no CUDA device is visible. Allocate a GPU \
             (SLURM: `--gres=gpu:1`) and check `nvidia-smi`, or pass `--device cpu` to \
             silence this.",
        ),
        CpuReason::FasterOnCpu => tracing::debug!("{LABEL} runs on the CPU"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_forces_cpu() {
        assert_eq!(
            place(Device::Cpu).unwrap(),
            Placement::Cpu(CpuReason::Requested)
        );
    }

    /// `auto` never errors, whatever the build or the node.
    #[test]
    fn auto_is_infallible() {
        assert!(place(Device::Auto).is_ok());
    }

    /// A build without the feature must *explain* itself rather than be
    /// missing the flag.
    #[test]
    fn gpu_on_an_uncompiled_build_names_the_feature() {
        if compiled_in() {
            return;
        }
        let err = place(Device::Gpu).unwrap_err().to_string();
        assert!(err.contains(FEATURE), "{err}");
        assert!(err.contains("--device gpu"), "{err}");
    }

    #[test]
    fn aliases_map_onto_device() {
        let cpu = DeviceArgs {
            device: None,
            cpu: true,
        };
        assert_eq!(cpu.resolve(), Device::Cpu);
        assert_eq!(DeviceArgs::default().resolve(), Device::Auto);
        let explicit = DeviceArgs {
            device: Some(Device::Cpu),
            cpu: false,
        };
        assert_eq!(explicit.resolve(), Device::Cpu);
    }

    /// A stage worth accelerating must be able to say what CPU costs.
    #[test]
    fn cpu_cost_tracks_auto_preference() {
        assert_eq!(auto_prefers_gpu(), CPU_COST.is_some());
    }
}
