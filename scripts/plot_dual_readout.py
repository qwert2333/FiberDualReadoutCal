#!/usr/bin/env python3
"""Create standard plots for the dual-readout study."""

from __future__ import annotations

import argparse
import math
import os
import tempfile

cache_dir = tempfile.mkdtemp(prefix="dual-readout-plot-cache-")
if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = cache_dir
if not os.environ.get("XDG_CACHE_HOME"):
    os.environ["XDG_CACHE_HOME"] = cache_dir

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ensure_dir, read_csv, read_json, repo_path, rms


def finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def resolution(values):
    clean = [float(value) for value in values if finite(value)]
    mu = sum(clean) / len(clean) if clean else math.nan
    return rms(clean) / mu if clean and mu != 0 else math.nan


def fit_resolution_ab(points):
    valid = [(float(e), float(r)) for e, r in points if finite(e) and finite(r) and float(e) > 0 and float(r) >= 0]
    if len(valid) < 2:
        return math.nan, math.nan
    x = [1.0 / e for e, _ in valid]
    y = [r * r for _, r in valid]
    n = len(valid)
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xx * xx for xx in x)
    sxy = sum(xx * yy for xx, yy in zip(x, y))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-18:
        return math.nan, math.nan
    a2 = (n * sxy - sx * sy) / denom
    b2 = (sy - a2 * sx) / n
    return math.sqrt(max(a2, 0.0)), math.sqrt(max(b2, 0.0))


def resolution_model(energy, a, b):
    if not finite(a) or not finite(b) or energy <= 0:
        return math.nan
    return math.sqrt((a * a) / energy + b * b)


def fit_label(name, a, b):
    if not finite(a) or not finite(b):
        return f"{name} fit unavailable"
    return f"{name}: a={100*a:.2f}% sqrt(GeV), b={100*b:.2f}%"


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print(f"wrote {path}")


def plot_electron_linearity(tables, out):
    rows = read_csv(tables / "electron_calibration.csv")
    energies = [float(row["energy_MeV"]) / 1000.0 for row in rows]
    s = [float(row["mean_S_raw"]) for row in rows]
    c = [float(row["mean_C_raw"]) for row in rows]
    plt.figure()
    plt.plot(energies, s, "o-", label="Scintillation")
    plt.plot(energies, c, "s-", label="Cherenkov")
    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("Mean raw signal [counts]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out / "electron_raw_linearity.png")

    plt.figure()
    plt.plot(energies, [float(row["S_cal_over_E_mean"]) for row in rows], "o-", label="S calibrated")
    plt.plot(energies, [float(row["C_cal_over_E_mean"]) for row in rows], "s-", label="C calibrated")
    plt.axhline(1.0, color="black", linewidth=1)
    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("Mean calibrated response / E")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out / "electron_calibrated_closure.png")


def plot_response_vs_fem(tables, out):
    rows = read_csv(tables / "event_reco.csv")
    response = read_csv(tables / "hadron_response.csv")
    fit = {(row["subset"], row["channel"]): row for row in response}
    fem = [float(row["truthFemCrystal"]) for row in rows]
    s = [float(row["S_cal_over_E"]) for row in rows]
    c = [float(row["C_cal_over_E"]) for row in rows]
    xline = [0.0, 1.0]

    plt.figure()
    plt.scatter(fem, s, s=12, alpha=0.65, label="S events")
    plt.scatter(fem, c, s=12, alpha=0.65, label="C events")
    if ("all", "S") in fit:
        h = float(fit[("all", "S")]["h_over_e"])
        plt.plot(xline, [x + h * (1 - x) for x in xline], label=f"S fit h/e={h:.3g}")
    if ("all", "C") in fit:
        h = float(fit[("all", "C")]["h_over_e"])
        plt.plot(xline, [x + h * (1 - x) for x in xline], label=f"C fit h/e={h:.3g}")
    plt.xlabel("truth f_em in scintillator")
    plt.ylabel("Calibrated response / E")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out / "hadron_response_vs_truth_fem.png")


def plot_h_over_e_vs_energy(tables, out):
    path = tables / "h_over_e_vs_energy.csv"
    if not path.exists():
        print(f"skip {path}: run fit_h_over_e_by_energy.py first")
        return
    rows = read_csv(path)
    energies = [float(row["energy_GeV"]) for row in rows]
    s = [float(row["h_over_e_S"]) for row in rows]
    s_err = [float(row["h_over_e_S_err"]) for row in rows]
    c = [float(row["h_over_e_C"]) for row in rows]
    c_err = [float(row["h_over_e_C_err"]) for row in rows]

    plt.figure(figsize=(7.2, 5.2))
    plt.errorbar(energies, s, yerr=s_err, fmt="o-", capsize=4, label="(h/e)_S")
    plt.errorbar(energies, c, yerr=c_err, fmt="s-", capsize=4, label="(h/e)_C")
    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("h/e")
    plt.xticks(energies)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(out / "h_over_e_vs_energy.png")


def plot_energy_distributions(tables, out, energy_gev=None):
    rows = read_csv(tables / "event_reco.csv")
    if energy_gev is not None:
        rows = [row for row in rows if abs(float(row["energy_GeV"]) - energy_gev) < 1e-9]
        if not rows:
            print(f"skip energy_response_distributions: no events at {energy_gev:g} GeV")
            return
    values = {
        "S": [float(row["S_cal"]) for row in rows],
        "C": [float(row["C_cal"]) for row in rows],
        "DR": [float(row["E_DR"]) for row in rows],
    }
    plt.figure()
    for label, data in values.items():
        data = [x for x in data if x == x]
        plt.hist(data, bins=30, histtype="step", linewidth=1.8, label=label)
    if energy_gev is not None:
        plt.title(f"{energy_gev:g} GeV")
    plt.xlabel("Reconstructed energy / beam energy")
    plt.ylabel("Events")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if energy_gev is None:
        savefig(out / "energy_response_distributions.png")
    else:
        savefig(out / f"energy_response_distributions_{energy_gev:g}GeV.png")


def plot_fem_reco(tables, out):
    rows = read_csv(tables / "event_reco.csv")
    truth = [float(row["truthFemCrystal"]) for row in rows if float(row["fem_DR"]) == float(row["fem_DR"])]
    reco = [float(row["fem_DR"]) for row in rows if float(row["fem_DR"]) == float(row["fem_DR"])]
    plt.figure()
    plt.scatter(truth, reco, s=12, alpha=0.7)
    plt.plot([0, 1], [0, 1], color="black", linewidth=1)
    plt.xlabel("truth f_em in scintillator")
    plt.ylabel("dual-readout reconstructed f_em")
    plt.grid(True, alpha=0.3)
    savefig(out / "reco_fem_vs_truth_fem.png")


def plot_fem_distributions_by_sample(tables, out):
    rows = read_csv(tables / "event_reco.csv")
    energies = sorted({float(row["energy_GeV"]) for row in rows})
    bins = [(-0.5 + i * 0.025) for i in range(81)]

    for energy in energies:
        sample = [
            row for row in rows
            if float(row["energy_GeV"]) == energy
            and float(row["fem_DR"]) == float(row["fem_DR"])
        ]
        truth = [float(row["truthFemCrystal"]) for row in sample]
        reco = [float(row["fem_DR"]) for row in sample]

        plt.figure(figsize=(7.2, 5.2))
        plt.hist(
            truth,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            label=f"truth f_em (RMS={rms(truth):.3f})",
        )
        plt.hist(
            reco,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            label=f"reco f_em (RMS={rms(reco):.3f})",
        )
        plt.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
        plt.axvline(1.0, color="black", linewidth=0.8, alpha=0.5)
        plt.xlabel("f_em")
        plt.ylabel("Events")
        plt.title(f"pi- {energy:g} GeV, truth f_em > 0 (N={len(sample)})")
        plt.xlim(-0.5, 1.5)
        plt.legend()
        plt.grid(True, alpha=0.3)
        savefig(out / f"fem_truth_reco_pim_{energy:g}GeV.png")


def plot_leakage_residual(tables, out):
    rows = read_csv(tables / "event_reco.csv")
    leakage = [float(row["leakageFraction"]) for row in rows]
    residual = [float(row["E_DR_over_E"]) - 1.0 for row in rows]
    plt.figure()
    plt.scatter(leakage, residual, s=12, alpha=0.7)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("Leakage energy / beam energy")
    plt.ylabel("E_DR / E - 1")
    plt.grid(True, alpha=0.3)
    savefig(out / "leakage_fraction_vs_dr_residual.png")


def plot_em_resolution(tables, out):
    path = tables / "electron_calibration.csv"
    if not path.exists():
        print(f"skip {path}: run calibrate_electrons.py first")
        return
    rows = read_csv(path)
    energies = [float(row["energy_GeV"]) for row in rows]
    s_res = [float(row["rms_S_raw"]) / float(row["mean_S_raw"]) for row in rows]
    c_res = [float(row["rms_C_raw"]) / float(row["mean_C_raw"]) for row in rows]
    s_a, s_b = fit_resolution_ab(zip(energies, s_res))
    c_a, c_b = fit_resolution_ab(zip(energies, c_res))
    xfit = [min(energies), max(energies)] if len(energies) >= 2 else energies
    if len(xfit) == 2:
        lo, hi = xfit
        xfit = [lo + (hi - lo) * i / 100.0 for i in range(101)]

    plt.figure(figsize=(7.2, 5.2))
    plt.plot(energies, [100.0 * r for r in s_res], "o", label="e- S")
    plt.plot(energies, [100.0 * r for r in c_res], "s", label="e- C")
    if xfit:
        plt.plot(xfit, [100.0 * resolution_model(e, s_a, s_b) for e in xfit], "-", label=fit_label("S", s_a, s_b))
        plt.plot(xfit, [100.0 * resolution_model(e, c_a, c_b) for e in xfit], "-", label=fit_label("C", c_a, c_b))
    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("EM energy resolution [%]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out / "electron_energy_resolution.png")


def plot_hadronic_resolution(tables, out):
    path = tables / "event_reco.csv"
    if not path.exists():
        print(f"skip {path}: run analyze_hadrons.py first")
        return
    rows = read_csv(path)
    energies = sorted({float(row["energy_GeV"]) for row in rows})
    points = []
    for energy in energies:
        sample = [row for row in rows if float(row["energy_GeV"]) == energy]
        points.append({
            "energy_GeV": energy,
            "resolution_S": resolution(row["S_cal_over_E"] for row in sample),
            "resolution_C": resolution(row["C_cal_over_E"] for row in sample),
            "resolution_DR": resolution(row["E_DR_over_E"] for row in sample),
            "n_events": len(sample),
        })
    s_a, s_b = fit_resolution_ab((row["energy_GeV"], row["resolution_S"]) for row in points)
    c_a, c_b = fit_resolution_ab((row["energy_GeV"], row["resolution_C"]) for row in points)
    dr_a, dr_b = fit_resolution_ab((row["energy_GeV"], row["resolution_DR"]) for row in points)
    xfit = [min(energies), max(energies)] if len(energies) >= 2 else energies
    if len(xfit) == 2:
        lo, hi = xfit
        xfit = [lo + (hi - lo) * i / 100.0 for i in range(101)]

    plt.figure(figsize=(7.2, 5.2))
    series = [
        ("S", "resolution_S", "o", s_a, s_b),
        ("C", "resolution_C", "s", c_a, c_b),
        ("DR", "resolution_DR", "^", dr_a, dr_b),
    ]
    for name, key, marker, a, b in series:
        y = [100.0 * float(row[key]) for row in points]
        plt.plot(energies, y, marker, label=f"pi- {name}")
        if xfit:
            plt.plot(xfit, [100.0 * resolution_model(e, a, b) for e in xfit], "-", label=fit_label(name, a, b))
    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("Hadronic energy resolution [%]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out / "hadron_energy_resolution.png")


def plot_edr_linearity(tables, out):
    path = tables / "event_reco.csv"
    if not path.exists():
        print(f"skip {path}: run analyze_hadrons.py first")
        return
    rows = read_csv(path)
    energies = sorted({float(row["energy_GeV"]) for row in rows})
    points = []
    for energy in energies:
        values = [float(row["E_DR"]) / 1000.0 for row in rows if float(row["energy_GeV"]) == energy and finite(row["E_DR"])]
        if not values:
            continue
        mu = sum(values) / len(values)
        points.append({
            "energy_GeV": energy,
            "mean_E_DR_GeV": mu,
            "rms_E_DR_GeV": rms(values),
            "n_events": len(values),
        })
    if not points:
        print("skip hadron_edr_linearity: no finite E_DR values")
        return

    x = [row["energy_GeV"] for row in points]
    y = [row["mean_E_DR_GeV"] for row in points]
    yerr = [row["rms_E_DR_GeV"] for row in points]
    lo = min(min(x), min(y))
    hi = max(max(x), max(y))

    plt.figure(figsize=(7.2, 5.2))
    plt.errorbar(x, y, yerr=yerr, fmt="o", capsize=4, label="pi- mean E_DR")
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, label="E_DR = E_beam")
    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("Mean E_DR [GeV]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(out / "hadron_edr_linearity.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--energy-gev", type=float, default=None, help="Only plot energy distributions for this beam energy")
    args = parser.parse_args()

    tables = repo_path(args.tables)
    out = ensure_dir(args.out)
    # plot_electron_linearity(tables, out)
    # plot_response_vs_fem(tables, out)
    # plot_h_over_e_vs_energy(tables, out)
    plot_energy_distributions(tables, out, args.energy_gev)
    # plot_fem_reco(tables, out)
    # plot_fem_distributions_by_sample(tables, out)
    plot_em_resolution(tables, out)
    plot_hadronic_resolution(tables, out)
    plot_edr_linearity(tables, out)
    #plot_leakage_residual(tables, out)


if __name__ == "__main__":
    main()
