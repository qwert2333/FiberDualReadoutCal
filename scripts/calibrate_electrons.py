#!/usr/bin/env python3
"""Calibrate scintillation and Cherenkov channels with electron samples."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

cache_dir = tempfile.mkdtemp(prefix="dual-readout-calib-plot-cache-")
if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = cache_dir
if not os.environ.get("XDG_CACHE_HOME"):
    os.environ["XDG_CACHE_HOME"] = cache_dir

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    find_root_files,
    finite,
    linear_slope_through_origin,
    load_events,
    mean,
    require_columns,
    rms,
    stderr,
    write_csv,
    write_json,
    repo_path,
    ensure_dir,
)


def summarize_file(path: Path) -> dict:
    events = load_events(path)
    s_column = "counter_Scintillation_ScintFiber"
    c_column = "counter_Cerenkov_CherenkovFiber"
    require_columns(events, ["MCtruth_energy", s_column, c_column], path)
    fem_column = "truthFemScint" if "truthFemScint" in events[0] else "truthFemCrystal" if "truthFemCrystal" in events[0] else None
    valid = [
        row for row in events
        if float(row["MCtruth_energy"]) > 0
        and float(row[s_column]) > 0
        and float(row[c_column]) > 0
    ]
    if not valid:
        raise SystemExit(f"No valid electron calibration events in {path}")

    energies = [float(row["MCtruth_energy"]) for row in valid]
    s_values = [float(row[s_column]) for row in valid]
    c_values = [float(row[c_column]) for row in valid]
    fem_values = [float(row[fem_column]) for row in valid if fem_column and float(row[fem_column]) >= 0]
    energy = mean(energies)
    mean_s = mean(s_values)
    mean_c = mean(c_values)
    return {
        "file": str(path.relative_to(repo_path("."))),
        "energy_MeV": energy,
        "energy_GeV": energy / 1000.0,
        "entries": len(events),
        "valid_entries": len(valid),
        "mean_S_raw": mean_s,
        "rms_S_raw": rms(s_values),
        "stderr_S_raw": stderr(s_values),
        "mean_C_raw": mean_c,
        "rms_C_raw": rms(c_values),
        "stderr_C_raw": stderr(c_values),
        "mean_truthFemScint": mean(fem_values) if fem_values else math.nan,
        "kS_point": energy / mean_s,
        "kC_point": energy / mean_c,
    }


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Wrote {path}")


def plot_raw_linearity(rows: list[dict], out_dir: Path, slope_s: float, slope_c: float) -> None:
    energies_gev = [float(row["energy_GeV"]) for row in rows]
    energies_mev = [float(row["energy_MeV"]) for row in rows]
    x_fit_gev = [0.0, max(energies_gev)] if energies_gev else [0.0, 0.0]
    x_fit_mev = [x * 1000.0 for x in x_fit_gev]

    plt.figure(figsize=(7.2, 5.2))
    plt.errorbar(
        energies_gev,
        [float(row["mean_S_raw"]) for row in rows],
        yerr=[float(row["stderr_S_raw"]) for row in rows],
        fmt="o",
        capsize=4,
        label="S raw mean",
    )
    plt.plot(x_fit_gev, [slope_s * x for x in x_fit_mev], "-", label=f"Fit: {slope_s:.4g} counts/MeV")
    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("S_raw [counts]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out_dir / "electron_S_raw_linearity.png")

    plt.figure(figsize=(7.2, 5.2))
    plt.errorbar(
        energies_gev,
        [float(row["mean_C_raw"]) for row in rows],
        yerr=[float(row["stderr_C_raw"]) for row in rows],
        fmt="s",
        capsize=4,
        label="C raw mean",
    )
    plt.plot(x_fit_gev, [slope_c * x for x in x_fit_mev], "-", label=f"Fit: {slope_c:.4g} counts/MeV")
    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("C_raw [counts]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out_dir / "electron_C_raw_linearity.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Directory containing electron ROOT files")
    parser.add_argument("--out", required=True, help="Output table directory")
    args = parser.parse_args()

    input_dir = repo_path(args.input)
    out_dir = ensure_dir(args.out)
    files = find_root_files(input_dir, role="electron")
    if not files:
        files = find_root_files(input_dir, particle="e-")
    if not files:
        raise SystemExit(f"No electron ROOT files found in {input_dir}")

    rows = [summarize_file(path) for path in files]
    rows.sort(key=lambda row: row["energy_MeV"])
    energies = [float(row["energy_MeV"]) for row in rows]
    mean_s = [float(row["mean_S_raw"]) for row in rows]
    mean_c = [float(row["mean_C_raw"]) for row in rows]

    slope_s = linear_slope_through_origin(energies, mean_s)
    slope_c = linear_slope_through_origin(energies, mean_c)
    a_s = 1.0 / slope_s if finite(slope_s) and slope_s > 0 else math.nan
    a_c = 1.0 / slope_c if finite(slope_c) and slope_c > 0 else math.nan

    for row in rows:
        row["S_cal_over_E_mean"] = a_s * float(row["mean_S_raw"]) / float(row["energy_MeV"])
        row["C_cal_over_E_mean"] = a_c * float(row["mean_C_raw"]) / float(row["energy_MeV"])

    calibration = {
        "signal_definitions": {
            "S_raw": "counter_Scintillation_ScintFiber",
            "C_raw": "counter_Cerenkov_CherenkovFiber",
        },
        "calibration": {
            "aS_MeV_per_count": a_s,
            "aC_MeV_per_count": a_c,
            "slope_S_counts_per_MeV": slope_s,
            "slope_C_counts_per_MeV": slope_c,
        },
        "points": rows,
    }

    write_csv(out_dir / "electron_calibration.csv", rows)
    write_json(out_dir / "electron_calibration.json", calibration)
    plot_raw_linearity(rows, out_dir, slope_s, slope_c)
    print(f"Wrote {out_dir / 'electron_calibration.csv'}")
    print(f"Wrote {out_dir / 'electron_calibration.json'}")
    print(f"aS = {a_s:.8g} MeV/count, aC = {a_c:.8g} MeV/count")


if __name__ == "__main__":
    main()
