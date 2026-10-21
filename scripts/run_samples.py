#!/usr/bin/env python3
"""Run generated Geant4 macros and collect ROOT files.

By default this script submits one CERN HTCondor job per generated macro. Use
--local to keep the old foreground execution mode for quick tests.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from common import config_dirs, load_config, repo_path, root_energy_label
from make_macros import iter_macro_samples, macro_sample_name


JOB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def config_job_name(config: dict) -> str:
    job_name = config.get("job_name")
    if not isinstance(job_name, str) or not job_name:
        raise SystemExit("Missing non-empty config value: job_name")
    if not JOB_NAME_PATTERN.fullmatch(job_name):
        raise SystemExit(
            "Invalid config job_name. Use only letters, numbers, '.', '_' and '-', "
            "and start with a letter or number."
        )
    return job_name


def output_root_pattern(sample: dict) -> str:
    if "energy_gev" in sample:
        energy_label = root_energy_label(sample["energy_gev"])
        return f"FiberDualReadoutCalo_*_{sample['particle']}_{energy_label}.root"
    # Geant4 names a GPS range run using its internal mono-energy value, which
    # is not the generated event energy. The per-job work directory keeps this
    # particle-only pattern unambiguous.
    return f"FiberDualReadoutCalo_*_{sample['particle']}_*.root"


def find_output_root(sample: dict, before: set[Path]) -> Path:
    pattern = output_root_pattern(sample)
    candidates = []
    for path in repo_path(".").glob(pattern):
        if path not in before:
            candidates.append(path)
    if not candidates:
        for path in repo_path(".").glob(pattern):
            candidates.append(path)
    if not candidates:
        raise SystemExit(f"Could not find ROOT output matching {pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_local(config: dict, executable: Path, dirs: dict[str, Path]) -> None:
    for sample in iter_macro_samples(config):
        name = macro_sample_name(sample)
        macro = dirs["macros"] / f"{name}.mac"
        if not macro.exists():
            raise SystemExit(f"Missing macro {macro}. Run make_macros.py first.")

        before = set(repo_path(".").glob("FiberDualReadoutCalo_*.root"))
        print(f"Running {name}: {macro}")
        subprocess.run([str(executable), str(macro)], cwd=str(repo_path(".")), check=True)
        produced = find_output_root(sample, before)
        destination = dirs["root"] / f"{name}.root"
        if destination.exists():
            destination.unlink()
        shutil.move(str(produced), str(destination))
        print(f"  wrote {destination.relative_to(repo_path('.'))}")


def quote_submit_value(value: str | Path) -> str:
    value = str(value)
    if any(char.isspace() for char in value):
        return f'"{value}"'
    return value


def write_condor_wrapper(path: Path, repo_root: Path, work_root: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "Usage: $0 SAMPLE PARTICLE OUTPUT_PATTERN MACRO EXECUTABLE CLUSTER_ID PROC_ID" >&2
  exit 2
fi

sample="$1"
particle="$2"
output_pattern="$3"
macro="$4"
executable="$5"
cluster_id="$6"
proc_id="$7"

repo_root="__REPO_ROOT__"
work_root="__WORK_ROOT__"
workdir="${work_root}/${sample}.${cluster_id}.${proc_id}"
stamp="${workdir}/dual_readout_${sample}_start.stamp"

echo "Host: $(hostname)"
echo "Start: $(date)"
echo "Repo: ${repo_root}"
echo "Workdir: ${workdir}"
echo "Sample: ${sample}"
echo "Macro: ${macro}"
echo "Executable: ${executable}"

mkdir -p "${workdir}"
cd "${workdir}"

set +u
source "${repo_root}/setup.sh"
set -u
echo "Source LCG and Geant4 end"

touch "${stamp}"

"${executable}" "${macro}"

mapfile -t candidates < <(
  find "${workdir}" -maxdepth 1 -type f \
    -name "${output_pattern}" \
    -newer "${stamp}" \
    -printf "%T@ %p\\n" | sort -n | awk '{print $2}'
)

if [[ ${#candidates[@]} -eq 0 ]]; then
  mapfile -t candidates < <(
    find "${workdir}" -maxdepth 1 -type f \
      -name "${output_pattern}" \
      -printf "%T@ %p\\n" | sort -n | awk '{print $2}'
  )
fi

if [[ ${#candidates[@]} -eq 0 ]]; then
  echo "Could not find ROOT output matching ${output_pattern} in ${workdir}" >&2
  exit 3
fi

echo "ROOT outputs left in ${workdir}:"
printf '  %s\\n' "${candidates[@]}"
echo "End: $(date)"
""".replace("__REPO_ROOT__", str(repo_root)).replace("__WORK_ROOT__", str(work_root))
    )
    path.chmod(path.stat().st_mode | 0o111)

def write_condor_submit(
    path: Path,
    wrapper: Path,
    jobs: list[dict[str, str]],
    log_dir: Path,
    request_cpus: int,
    request_memory: int,
    job_flavour: str,
    job_name: str,
) -> None:
    rows = "\n".join(
        " ".join(
            quote_submit_value(job[key])
            for key in ["sample", "particle", "output_pattern", "macro", "executable"]
        )
        for job in jobs
    )
    path.write_text(
        f"""executable = {wrapper}
arguments = $(sample) $(particle) $(output_pattern) $(macro) $(executable_path) $(ClusterId) $(ProcId)

initialdir = {repo_path(".")}
output = {log_dir}/$(sample).$(ClusterId).$(ProcId).out
error = {log_dir}/$(sample).$(ClusterId).$(ProcId).err
log = {log_dir}/{job_name}.$(ClusterId).log

request_cpus = {request_cpus}
request_memory = {request_memory}
+JobFlavour = "{job_flavour}"

queue sample, particle, output_pattern, macro, executable_path from (
{rows}
)
"""
    )


def submit_condor(
    config: dict,
    executable: Path,
    dirs: dict[str, Path],
    condor_dir: Path,
    request_cpus: int,
    request_memory: int,
    job_flavour: str,
    dry_run: bool,
) -> None:
    condor_dir.mkdir(parents=True, exist_ok=True)
    log_dir = condor_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    job_name = config_job_name(config)
    wrapper = condor_dir / f"{job_name}.sh"
    submit_file = condor_dir / f"{job_name}.sub"
    write_condor_wrapper(wrapper, repo_path("."), condor_dir / "work")

    jobs = []
    for sample in iter_macro_samples(config):
        name = macro_sample_name(sample)
        macro = dirs["macros"] / f"{name}.mac"
        if not macro.exists():
            raise SystemExit(f"Missing macro {macro}. Run make_macros.py first.")
        jobs.append({
            "sample": name,
            "particle": sample["particle"],
            "output_pattern": output_root_pattern(sample),
            "macro": str(macro),
            "executable": str(executable),
        })

    if not jobs:
        raise SystemExit("No samples found in config")

    write_condor_submit(
        submit_file,
        wrapper,
        jobs,
        log_dir,
        request_cpus,
        request_memory,
        job_flavour,
        job_name,
    )
    print(f"Wrote {wrapper.relative_to(repo_path('.'))}")
    print(f"Wrote {submit_file.relative_to(repo_path('.'))}")
    print(f"Prepared {len(jobs)} Condor jobs")

    if dry_run:
        print(f"Dry run: not submitting. Submit manually with: condor_submit {submit_file}")
        return

    subprocess.run(["condor_submit", str(submit_file)], cwd=str(repo_path(".")), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--executable", default="build/main")
    parser.add_argument("--local", action="store_true", help="Run samples locally instead of submitting HTCondor jobs")
    parser.add_argument("--condor-dir", default=None, help="Directory for generated submit files and Condor logs")
    parser.add_argument(
        "--request-cpus",
        type=int,
        default=None,
        help="CPU request per job. Defaults to config['threads'] so it matches /run/numberOfThreads.",
    )
    parser.add_argument("--request-memory", type=int, default=3000, help="Memory request in MB")
    parser.add_argument("--job-flavour", default="longlunch")
    parser.add_argument("--dry-run", action="store_true", help="Write Condor files but do not call condor_submit")
    args = parser.parse_args()

    config = load_config(args.config)
    dirs = config_dirs(config)
    executable = repo_path(args.executable)
    if not executable.exists():
        raise SystemExit(f"Missing executable: {executable}")

    if args.local:
        run_local(config, executable, dirs)
        return

    condor_dir = repo_path(args.condor_dir) if args.condor_dir else dirs["root"].parent / "condor"
    request_cpus = args.request_cpus if args.request_cpus is not None else int(config.get("threads", 1))
    submit_condor(
        config=config,
        executable=executable,
        dirs=dirs,
        condor_dir=condor_dir,
        request_cpus=request_cpus,
        request_memory=args.request_memory,
        job_flavour=args.job_flavour,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
