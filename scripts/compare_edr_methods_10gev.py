#!/usr/bin/env python3
"""Compare three dual-readout E_DR constructions on one beam-energy sample."""

from __future__ import annotations

import argparse
import math
import os
import tempfile
from pathlib import Path

cache_dir = tempfile.mkdtemp(prefix="dual-readout-edr-methods-plot-cache-")
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
    finite,
    fit_response_vs_fem,
    load_events,
    mean,
    read_json,
    require_columns,
    rms,
    write_csv,
    write_json,
    repo_path,
)

S_COLUMN = "counter_Scintillation_ScintFiber"
C_COLUMN = "counter_Cerenkov_CherenkovFiber"
FEM_COLUMN = "truthFemScint"


def resolution(values: list[float]) -> float:
    clean = [value for value in values if finite(value)]
    mu = mean(clean)
    return rms(clean) / abs(mu) if clean and mu != 0 else math.nan


def e_dr_over_e(s_cal: float, c_cal: float, energy: float, chi: float) -> float:
    if not finite(chi) or abs(1.0 - chi) < 1e-12 or energy <= 0:
        return math.nan
    return ((s_cal - chi * c_cal) / (1.0 - chi)) / energy


def find_sample_file(input_dir: Path, particle: str, energy_gev: float) -> Path:
    energy_label = f"{int(energy_gev)}GeV" if float(energy_gev).is_integer() else f"{energy_gev:g}GeV"
    matches = [path for path in find_root_files(input_dir) if f"_{particle}_{energy_label}" in path.name]
    if not matches:
        raise SystemExit(f"No ROOT file found for particle={particle}, energy={energy_label} in {input_dir}")
    if len(matches) > 1:
        print("Multiple matching files found; using the newest:")
        for path in matches:
            print(f"  {path}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def load_sample(path: Path, a_s: float, a_c: float, cache_dir: Path) -> list[dict]:
    events = load_events(path, cache_dir=cache_dir)
    require_columns(events, ["eventID", "particle", "MCtruth_energy", S_COLUMN, C_COLUMN, FEM_COLUMN], path)
    rows = []
    for row in events:
        energy = float(row["MCtruth_energy"])
        fem = float(row[FEM_COLUMN])
        s_raw = float(row[S_COLUMN])
        c_raw = float(row[C_COLUMN])
        if energy <= 0 or fem < 0 or s_raw <= 0 or c_raw <= 0:
            continue
        s_cal = a_s * s_raw
        c_cal = a_c * c_raw
        rows.append({
            "eventID": int(row["eventID"]),
            "particle": row["particle"],
            "energy_MeV": energy,
            "energy_GeV": energy / 1000.0,
            "truthFemScint": fem,
            "S_raw": s_raw,
            "C_raw": c_raw,
            "S_cal": s_cal,
            "C_cal": c_cal,
            "S_cal_over_E": s_cal / energy,
            "C_cal_over_E": c_cal / energy,
            "C_over_S": c_cal / s_cal if s_cal > 0 else math.nan,
        })
    if not rows:
        raise SystemExit(f"No valid events after quality cuts in {path}")
    return rows


def method_truth_fem_fit(rows: list[dict]) -> dict:
    fem = [row["truthFemScint"] for row in rows]
    s_response = [row["S_cal_over_E"] for row in rows]
    c_response = [row["C_cal_over_E"] for row in rows]
    fit_s = fit_response_vs_fem(fem, s_response)
    fit_c = fit_response_vs_fem(fem, c_response)
    h_s = fit_s["h_over_e"]
    h_c = fit_c["h_over_e"]
    chi = (1.0 - h_s) / (1.0 - h_c) if abs(1.0 - h_c) > 1e-12 else math.nan
    values = [e_dr_over_e(row["S_cal"], row["C_cal"], row["energy_MeV"], chi) for row in rows]
    return {
        "method": "truth_fem_fit",
        "chi": chi,
        "resolution": resolution(values),
        "mean_E_DR_over_E": mean([value for value in values if finite(value)]),
        "h_over_e_S": h_s,
        "h_over_e_C": h_c,
        "fit_rms_S": fit_s["fit_rms"],
        "fit_rms_C": fit_c["fit_rms"],
        "values": values,
    }


def method_scan_chi(rows: list[dict], chi_min: float, chi_max: float, chi_steps: int) -> dict:
    best = None
    for index in range(chi_steps + 1):
        chi = chi_min + (chi_max - chi_min) * index / chi_steps
        if abs(1.0 - chi) < 1e-12:
            continue
        values = [e_dr_over_e(row["S_cal"], row["C_cal"], row["energy_MeV"], chi) for row in rows]
        res = resolution(values)
        if finite(res) and (best is None or res < best["resolution"]):
            best = {"chi": chi, "resolution": res, "values": values}
    if best is None:
        return {"method": "scan_chi", "chi": math.nan, "resolution": math.nan, "mean_E_DR_over_E": math.nan, "values": []}
    clean = [value for value in best["values"] if finite(value)]
    return {
        "method": "scan_chi",
        "chi": best["chi"],
        "resolution": best["resolution"],
        "mean_E_DR_over_E": mean(clean),
        "values": best["values"],
    }


def solve_2x2(m00: float, m01: float, m11: float, v0: float, v1: float) -> tuple[float, float]:
    det = m00 * m11 - m01 * m01
    if abs(det) < 1e-18:
        return math.nan, math.nan
    return (v0 * m11 - v1 * m01) / det, (m00 * v1 - m01 * v0) / det


def method_linear_energy_fit(rows: list[dict]) -> dict:
    m00 = sum(row["S_cal"] * row["S_cal"] for row in rows)
    m01 = sum(row["S_cal"] * row["C_cal"] for row in rows)
    m11 = sum(row["C_cal"] * row["C_cal"] for row in rows)
    v0 = sum(row["S_cal"] * row["energy_MeV"] for row in rows)
    v1 = sum(row["C_cal"] * row["energy_MeV"] for row in rows)
    a, b = solve_2x2(m00, m01, m11, v0, v1)
    chi = -b / a if finite(a) and abs(a) > 1e-18 else math.nan
    values = [e_dr_over_e(row["S_cal"], row["C_cal"], row["energy_MeV"], chi) for row in rows]
    return {
        "method": "linear_energy_fit",
        "chi": chi,
        "resolution": resolution(values),
        "mean_E_DR_over_E": mean([value for value in values if finite(value)]),
        "fit_a": a,
        "fit_b": b,
        "values": values,
    }


def write_outputs(rows: list[dict], methods: list[dict], out_dir: Path, energy_gev: float) -> None:
    event_rows = []
    for idx, row in enumerate(rows):
        out = dict(row)
        for method in methods:
            values = method.get("values", [])
            out[f"E_DR_over_E_{method['method']}"] = values[idx] if idx < len(values) else math.nan
            out[f"E_DR_{method['method']}"] = out[f"E_DR_over_E_{method['method']}"] * row["energy_MeV"] if finite(out[f"E_DR_over_E_{method['method']}"]) else math.nan
        event_rows.append(out)

    summary_rows = []
    for method in methods:
        summary_rows.append({key: value for key, value in method.items() if key != "values"})

    write_csv(out_dir / "edr_methods_10GeV_events.csv", event_rows)
    write_csv(out_dir / "edr_methods_10GeV_summary.csv", summary_rows)
    write_json(out_dir / "edr_methods_10GeV_summary.json", {"energy_GeV": energy_gev, "methods": summary_rows})

    plt.figure(figsize=(7.2, 5.2))
    bins = [0.0 + 0.02 * i for i in range(101)]
    for method in methods:
        values = [value for value in method.get("values", []) if finite(value)]
        plt.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            label=f"{method['method']} chi={method['chi']:.4g}, res={100*method['resolution']:.2f}%",
        )
    plt.axvline(1.0, color="black", linewidth=1.0)
    plt.xlabel("E_DR / E_beam")
    plt.ylabel("Events")
    plt.title(f"{energy_gev:g} GeV dual-readout E_DR methods")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / "edr_methods_10GeV.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()

    plt.figure(figsize=(7.2, 5.2))
    c_over_s = [row["C_over_S"] for row in rows]
    for method in methods:
        values = method.get("values", [])
        points = [
            (x, y)
            for x, y in zip(c_over_s, values)
            if finite(x) and finite(y)
        ]
        if not points:
            continue
        xs, ys = zip(*points)
        plt.scatter(
            xs,
            ys,
            s=16,
            alpha=0.55,
            edgecolors="none",
            label=f"{method['method']} chi={method['chi']:.4g}",
        )
    plt.axhline(1.0, color="black", linewidth=1.0)
    plt.xlabel("C / S")
    plt.ylabel("E_DR / E_beam")
    plt.title(f"{energy_gev:g} GeV E_DR response vs C/S")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ratio_plot_path = out_dir / "edr_methods_10GeV_vs_c_over_s.png"
    plt.savefig(ratio_plot_path, dpi=160)
    plt.close()

    print(f"Wrote {out_dir / 'edr_methods_10GeV_events.csv'}")
    print(f"Wrote {out_dir / 'edr_methods_10GeV_summary.csv'}")
    print(f"Wrote {out_dir / 'edr_methods_10GeV_summary.json'}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {ratio_plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/root", help="Directory containing ROOT files")
    parser.add_argument("--calib", default="outputs/tables/electron_calibration.json", help="electron_calibration.json")
    parser.add_argument("--out", default="outputs/tables", help="Output directory for tables and plot")
    parser.add_argument("--energy-gev", type=float, default=10.0)
    parser.add_argument("--particle", default="pi-")
    parser.add_argument("--chi-min", type=float, default=-2.0)
    parser.add_argument("--chi-max", type=float, default=2.0)
    parser.add_argument("--chi-steps", type=int, default=4000)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out)
    calibration = read_json(args.calib)["calibration"]
    a_s = float(calibration["aS_MeV_per_count"])
    a_c = float(calibration["aC_MeV_per_count"])
    if not finite(a_s) or not finite(a_c):
        raise SystemExit("Calibration constants are not finite")

    sample_file = find_sample_file(repo_path(args.input), args.particle, args.energy_gev)
    cache = ensure_dir(out_dir / "_csv_cache_edr_methods")
    rows = load_sample(sample_file, a_s, a_c, cache)
    methods = [
        method_truth_fem_fit(rows),
        method_scan_chi(rows, args.chi_min, args.chi_max, args.chi_steps),
        method_linear_energy_fit(rows),
    ]
    write_outputs(rows, methods, out_dir, args.energy_gev)

    print("Method summary:")
    for method in methods:
        print(
            f"  {method['method']}: chi={method['chi']:.8g}, "
            f"resolution={100*method['resolution']:.4g}%, mean E_DR/E={method['mean_E_DR_over_E']:.6g}"
        )


if __name__ == "__main__":
    main()
