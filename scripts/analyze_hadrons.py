#!/usr/bin/env python3
"""Analyze hadron samples with electron calibration constants.

Follows S.Y. Jang et al., NIM A 1090 (2026) 171598, Sec. 4.4.1, extended to
an energy-dependent h/e:
  1. at each beam energy, (h/e)_S and (h/e)_C from the profile of S/E and C/E
     versus the f_em of Eq. 11, fitted with R = p1 + p0 f_em;
  2. fit h/e(E) = a + b ln(E/GeV) (or a + b E) for each channel;
  3. chi(E) = (1 - (h/e)_S(E)) / (1 - (h/e)_C(E)) at the beam energy,
     E_DR = (S - chi C) / (1 - chi).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from collections import defaultdict

from common import (
    chi_from_h_over_e,
    fem_from_ratio,
    find_root_files,
    finite,
    load_events,
    mean,
    paper_h_over_e,
    read_json,
    require_columns,
    rms,
    stderr,
    weighted_line_fit,
    write_csv,
    write_json,
    repo_path,
    ensure_dir,
)


def reco_events(path: Path, a_s: float, a_c: float) -> tuple[list[dict], dict]:
    events = load_events(path)
    s_column = "counter_Scintillation_ScintFiber"
    c_column = "counter_Cerenkov_CherenkovFiber"
    fem_column = "truthFem"
    require_columns(events, ["MCtruth_energy", s_column, c_column, fem_column, "truthFemScint"], path)

    rows = []
    rejected = {
        "S_raw_le_0": 0,
        "C_raw_le_0": 0,
        "energy_le_0": 0,
    }
    for row in events:
        energy = float(row["MCtruth_energy"])
        s_raw = float(row[s_column])
        c_raw = float(row[c_column])
        if energy <= 0:
            rejected["energy_le_0"] += 1
            continue
        if s_raw <= 0:
            rejected["S_raw_le_0"] += 1
            continue
        if c_raw <= 0:
            rejected["C_raw_le_0"] += 1
            continue

        s_cal = a_s * s_raw
        c_cal = a_c * c_raw
        leakage = float(row["leakageEnergy"]) if "leakageEnergy" in row else math.nan
        leakage_fraction = leakage / energy if finite(leakage) and energy > 0 else math.nan
        rows.append({
            "source_file": str(path.relative_to(repo_path("."))),
            "eventID": int(row["eventID"]),
            "particle": row["particle"],
            "energy_MeV": energy,
            "energy_GeV": energy / 1000.0,
            "S_raw": s_raw,
            "C_raw": c_raw,
            "S_cal": s_cal,
            "C_cal": c_cal,
            "S_cal_over_E": s_cal / energy,
            "C_cal_over_E": c_cal / energy,
            "truthFem": float(row[fem_column]),
            "truthFemCrystal": float(row["truthFemScint"]),
            "truthFemScint": float(row["truthFemScint"]),
            "leakageEnergy": leakage,
            "leakageFraction": leakage_fraction,
            "EdepScintFiber": float(row.get("EdepScintFiber", math.nan)),
            "EdepCherenkovFiber": float(row.get("EdepCherenkovFiber", math.nan)),
            "EdepAbsorber": float(row.get("EdepAbsorber", math.nan)),
            "truthEdepScintTotal": float(row.get("truthEdepScintTotal", math.nan)),
            "truthEdepCherenkovTotal": float(row.get("truthEdepCherenkovTotal", math.nan)),
        })
    return rows, rejected


def group_by_energy(rows: list[dict]) -> dict[float, list[dict]]:
    groups: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        groups[float(row["energy_GeV"])].append(row)
    return dict(sorted(groups.items()))


def h_over_e_by_energy(groups: dict[float, list[dict]]) -> list[dict]:
    """Paper method (Sec. 4.4.1) applied separately at each beam energy."""
    points = []
    for energy, sample in groups.items():
        result = paper_h_over_e(
            [float(row["S_cal_over_E"]) for row in sample],
            [float(row["C_cal_over_E"]) for row in sample],
            [float(row["truthFem"]) for row in sample],
        )
        fit_s, fit_c = result["S"], result["C"]
        points.append({
            "energy_GeV": energy,
            "n_events": len(sample),
            "h_over_e_S": fit_s["h_over_e"],
            "h_over_e_S_err": fit_s["h_over_e_err"],
            "p1_S": fit_s["p1"],
            "p0_S": fit_s["p0"],
            "h_over_e_C": fit_c["h_over_e"],
            "h_over_e_C_err": fit_c["h_over_e_err"],
            "p1_C": fit_c["p1"],
            "p0_C": fit_c["p0"],
            "chi_point": chi_from_h_over_e(fit_s["h_over_e"], fit_c["h_over_e"]),
            "seed_h_over_e_S_truthFem": result["seed_S"]["h_over_e"],
            "seed_h_over_e_C_truthFem": result["seed_C"]["h_over_e"],
        })
    return points


def hoe_variable(energy_gev: float, model: str) -> float:
    return math.log(energy_gev) if model == "log" else energy_gev


def fit_h_over_e_model(points: list[dict], model: str, emin: float) -> dict:
    """Fit h/e(E) = intercept + slope * x, x = ln(E/GeV) (log) or E/GeV (linear)."""
    used = [row for row in points if row["energy_GeV"] >= emin]
    out = {"model": model, "variable": "ln(E/GeV)" if model == "log" else "E/GeV", "fit_emin_GeV": emin,
           "fit_energies_GeV": [row["energy_GeV"] for row in used]}
    for channel in ("S", "C"):
        fit = weighted_line_fit(
            [hoe_variable(row["energy_GeV"], model) for row in used],
            [row[f"h_over_e_{channel}"] for row in used],
            [row[f"h_over_e_{channel}_err"] for row in used],
        )
        out[channel] = {
            "intercept": fit["intercept"],
            "intercept_err": math.sqrt(fit["var_intercept"]),
            "slope": fit["slope"],
            "slope_err": math.sqrt(fit["var_slope"]),
            "chi2": fit["chi2"],
            "ndf": fit["ndf"],
        }
    return out


def h_over_e_at(model_fit: dict, channel: str, energy_gev: float) -> float:
    params = model_fit[channel]
    return params["intercept"] + params["slope"] * hoe_variable(energy_gev, model_fit["model"])


def reconstruct(groups: dict[float, list[dict]], model_fit: dict) -> list[dict]:
    """E_DR = (S - chi C)/(1 - chi) with chi from h/e(E) at the beam energy (paper Eq. 9)."""
    summaries = []
    for energy, sample in groups.items():
        h_s = h_over_e_at(model_fit, "S", energy)
        h_c = h_over_e_at(model_fit, "C", energy)
        chi = chi_from_h_over_e(h_s, h_c)
        for row in sample:
            s_cal = float(row["S_cal"])
            c_cal = float(row["C_cal"])
            e_dr = (s_cal - chi * c_cal) / (1.0 - chi)
            row["h_over_e_S"] = h_s
            row["h_over_e_C"] = h_c
            row["chi"] = chi
            row["E_DR"] = e_dr
            row["E_DR_over_E"] = e_dr / float(row["energy_MeV"])
            row["fem_DR"] = fem_from_ratio(float(row["S_cal_over_E"]), float(row["C_cal_over_E"]), h_s, h_c)

        summary = {"energy_GeV": energy, "n_events": len(sample),
                   "h_over_e_S_model": h_s, "h_over_e_C_model": h_c, "chi": chi}
        for name, key in (("S", "S_cal_over_E"), ("C", "C_cal_over_E"), ("DR", "E_DR_over_E")):
            values = [float(row[key]) for row in sample]
            mu = mean(values)
            summary[f"mean_{name}_over_E"] = mu
            summary[f"mean_{name}_over_E_err"] = stderr(values)
            summary[f"resolution_{name}"] = rms(values) / mu
        fem_delta = [float(row["fem_DR"]) - float(row["truthFem"]) for row in sample if finite(float(row["fem_DR"]))]
        summary["mean_fem_DR_minus_truthFem"] = mean(fem_delta)
        summary["rms_fem_DR_minus_truthFem"] = rms(fem_delta)
        summaries.append(summary)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Directory containing hadron ROOT files")
    parser.add_argument("--calib", required=True, help="electron_calibration.json")
    parser.add_argument("--out", required=True, help="Output table directory")
    parser.add_argument("--hoe-model", choices=["log", "linear"], default="log",
                        help="h/e(E) = a + b ln(E/GeV) (log, default) or a + b E (linear)")
    parser.add_argument("--hoe-emin", type=float, default=0.0, help="Lowest beam energy [GeV] in the h/e(E) fit")
    args = parser.parse_args()

    calibration = read_json(args.calib)["calibration"]
    a_s = float(calibration["aS_MeV_per_count"])
    a_c = float(calibration["aC_MeV_per_count"])
    if not finite(a_s) or not finite(a_c):
        raise SystemExit("Calibration constants are not finite")

    input_dir = repo_path(args.input)
    out_dir = ensure_dir(args.out)
    files = [
        path for path in find_root_files(input_dir)
        if "electron_" not in path.name and "_e-_" not in path.name
    ]
    if not files:
        raise SystemExit(f"No hadron ROOT files found in {input_dir}")

    all_rows = []
    rejected_by_file = {}
    for path in files:
        rows, rejected = reco_events(path, a_s, a_c)
        all_rows.extend(rows)
        rejected_by_file[str(path.relative_to(repo_path(".")))] = rejected
    if not all_rows:
        raise SystemExit("No valid hadron events after quality cuts")

    groups = group_by_energy(all_rows)
    hoe_points = h_over_e_by_energy(groups)
    model_fit = fit_h_over_e_model(hoe_points, args.hoe_model, args.hoe_emin)
    for row in hoe_points:
        for channel in ("S", "C"):
            row[f"h_over_e_{channel}_model"] = h_over_e_at(model_fit, channel, row["energy_GeV"])
        row["chi_model"] = chi_from_h_over_e(row["h_over_e_S_model"], row["h_over_e_C_model"])
    summaries = reconstruct(groups, model_fit)

    summary = {
        "calibration_file": str(repo_path(args.calib).relative_to(repo_path("."))),
        "reference": "S.Y. Jang et al., NIM A 1090 (2026) 171598, Sec. 4.4.1, Eqs. 9-13",
        "signal_definitions": {
            "S_raw": "counter_Scintillation_ScintFiber",
            "C_raw": "counter_Cerenkov_CherenkovFiber",
            "truth_fem": "truthFem (seed for Eq. 11 only)",
            "h_over_e": "per energy: profile of S/E, C/E vs f_em(Eq. 11), fit R = p1 + p0 f_em, h/e = p1/(p0+p1)",
            "E_DR": "chi(E_beam) from the h/e(E) fit, E_DR = (S - chi*C)/(1 - chi)",
        },
        "h_over_e_model": model_fit,
        "rejected_by_file": rejected_by_file,
        "per_energy": summaries,
    }

    write_csv(out_dir / "h_over_e_vs_energy.csv", hoe_points)
    write_csv(out_dir / "hadron_response.csv", summaries)
    write_csv(out_dir / "event_reco.csv", all_rows)
    write_json(out_dir / "dual_readout_summary.json", summary)
    print(f"Wrote {out_dir / 'h_over_e_vs_energy.csv'}")
    print(f"Wrote {out_dir / 'hadron_response.csv'}")
    print(f"Wrote {out_dir / 'event_reco.csv'}")
    print(f"Wrote {out_dir / 'dual_readout_summary.json'}")
    for channel in ("S", "C"):
        params = model_fit[channel]
        print(f"(h/e)_{channel}(E) = {params['intercept']:.4f} + {params['slope']:.4f} * {model_fit['variable']}"
              f"   chi2/ndf = {params['chi2']:.1f}/{params['ndf']}")
    for row, summ in zip(hoe_points, summaries):
        print(f"{row['energy_GeV']:>4g} GeV: h/e S={row['h_over_e_S']:.4f}+-{row['h_over_e_S_err']:.4f} "
              f"C={row['h_over_e_C']:.4f}+-{row['h_over_e_C_err']:.4f}  chi(model)={summ['chi']:.4f}  "
              f"<DR>/E={summ['mean_DR_over_E']:.4f}  res DR={100*summ['resolution_DR']:.2f}%")


if __name__ == "__main__":
    main()
