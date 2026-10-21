#!/usr/bin/env python3
"""Compute S- and C-fiber sampling fractions, <Edep_fiber> / E_beam, per sample."""

from __future__ import annotations

import argparse
import os
import tempfile

cache_dir = tempfile.mkdtemp(prefix="dual-readout-sampling-plot-cache-")
if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = cache_dir
if not os.environ.get("XDG_CACHE_HOME"):
    os.environ["XDG_CACHE_HOME"] = cache_dir

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (
    ensure_dir,
    find_root_files,
    load_events,
    mean,
    repo_path,
    require_columns,
    rms,
    stderr,
    write_csv,
    write_json,
)


S_EDEP_COLUMN = "EdepScintFiber"
C_EDEP_COLUMN = "EdepCherenkovFiber"


def summarize_file(path) -> dict:
    events = load_events(path)
    require_columns(events, ["particle", "MCtruth_energy", S_EDEP_COLUMN, C_EDEP_COLUMN], path)
    energies = [float(row["MCtruth_energy"]) for row in events]
    f_s = [float(row[S_EDEP_COLUMN]) / energy for row, energy in zip(events, energies) if energy > 0]
    f_c = [float(row[C_EDEP_COLUMN]) / energy for row, energy in zip(events, energies) if energy > 0]
    energy = mean(energies)
    return {
        "file": str(path.relative_to(repo_path("."))),
        "particle": str(events[0]["particle"]),
        "energy_MeV": energy,
        "energy_GeV": energy / 1000.0,
        "entries": len(events),
        "f_S": mean(f_s),
        "f_S_rms": rms(f_s),
        "f_S_err": stderr(f_s),
        "f_C": mean(f_c),
        "f_C_rms": rms(f_c),
        "f_C_err": stderr(f_c),
    }


def plot_sampling_fraction(rows: list[dict], out_dir) -> None:
    plt.figure(figsize=(7.2, 5.2))
    for particle, marker in [("e-", "o"), ("pi-", "s")]:
        sample = [row for row in rows if row["particle"] == particle]
        if not sample:
            continue
        energies = [row["energy_GeV"] for row in sample]
        for channel, fill in [("S", "full"), ("C", "none")]:
            plt.errorbar(
                energies,
                [100.0 * row[f"f_{channel}"] for row in sample],
                yerr=[100.0 * row[f"f_{channel}_err"] for row in sample],
                fmt=f"{marker}-",
                fillstyle=fill,
                capsize=4,
                label=f"{particle} {channel} fiber",
            )
    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("<Edep_fiber> / E_beam [%]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "sampling_fraction.png"
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Directory containing ROOT files")
    parser.add_argument("--out", required=True, help="Output table directory")
    parser.add_argument("--plots", default=None, help="Output plot directory (default: --out)")
    args = parser.parse_args()

    input_dir = repo_path(args.input)
    out_dir = ensure_dir(args.out)
    plot_dir = ensure_dir(args.plots) if args.plots else out_dir
    files = find_root_files(input_dir)
    if not files:
        raise SystemExit(f"No ROOT files found in {input_dir}")

    rows = [summarize_file(path) for path in files]
    rows.sort(key=lambda row: (row["particle"], row["energy_MeV"]))

    write_csv(out_dir / "sampling_fraction.csv", rows)
    write_json(out_dir / "sampling_fraction.json", {
        "definition": "f_X = mean over events of Edep_X / MCtruth_energy, X = EdepScintFiber (S), EdepCherenkovFiber (C)",
        "points": rows,
    })
    plot_sampling_fraction(rows, plot_dir)
    print(f"Wrote {out_dir / 'sampling_fraction.csv'}")
    print(f"Wrote {out_dir / 'sampling_fraction.json'}")
    for row in rows:
        print(
            f"{row['particle']:>3} {row['energy_GeV']:g} GeV: "
            f"f_S = {100*row['f_S']:.4f} +/- {100*row['f_S_err']:.4f} %, "
            f"f_C = {100*row['f_C']:.4f} +/- {100*row['f_C_err']:.4f} %"
        )


if __name__ == "__main__":
    main()
