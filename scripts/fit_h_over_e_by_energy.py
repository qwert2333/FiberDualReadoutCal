#!/usr/bin/env python3
"""Fit hadron h/e independently at each beam energy."""

from __future__ import annotations

import argparse
from collections import defaultdict

from common import (
    ensure_dir,
    find_root_files,
    finite,
    fit_response_vs_fem,
    load_events,
    read_json,
    repo_path,
    require_columns,
    write_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Directory containing hadron ROOT files")
    parser.add_argument("--calib", required=True, help="electron_calibration.json")
    parser.add_argument("--out", required=True, help="Output table directory")
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

    by_energy = defaultdict(list)
    rejected = defaultdict(lambda: defaultdict(int))
    for path in files:
        events = load_events(path)
        s_column = "counter_Scintillation_ScintFiber"
        c_column = "counter_Cerenkov_CherenkovFiber"
        fem_column = "truthFemScint"
        require_columns(events, ["MCtruth_energy", s_column, c_column, fem_column], path)
        for event in events:
            energy = float(event["MCtruth_energy"])
            fem = float(event[fem_column])
            s_raw = float(event[s_column])
            c_raw = float(event[c_column])
            if energy <= 0:
                rejected[energy]["energy_le_0"] += 1
                continue
            if fem < 0:
                rejected[energy]["truthFem_lt_0"] += 1
                continue
            if s_raw <= 0:
                rejected[energy]["S_raw_le_0"] += 1
                continue
            if c_raw <= 0:
                rejected[energy]["C_raw_le_0"] += 1
                continue
            by_energy[energy].append({
                "truthFemCrystal": fem,
                "S_cal_over_E": a_s * s_raw / energy,
                "C_cal_over_E": a_c * c_raw / energy,
            })

    rows = []
    for energy in sorted(by_energy):
        sample = by_energy[energy]
        fem = [row["truthFemCrystal"] for row in sample]
        fit_s = fit_response_vs_fem(fem, [row["S_cal_over_E"] for row in sample])
        fit_c = fit_response_vs_fem(fem, [row["C_cal_over_E"] for row in sample])
        rows.append({
            "energy_MeV": energy,
            "energy_GeV": energy / 1000.0,
            "n_events": len(sample),
            "n_truthFem_eq_0": sum(value == 0 for value in fem),
            "h_over_e_S": fit_s["h_over_e"],
            "h_over_e_S_err": fit_s["h_over_e_err"],
            "fit_rms_S": fit_s["fit_rms"],
            "h_over_e_C": fit_c["h_over_e"],
            "h_over_e_C_err": fit_c["h_over_e_err"],
            "fit_rms_C": fit_c["fit_rms"],
        })

    payload = {
        "selection": "truthFemScint >= 0, counter_Scintillation_ScintFiber > 0, counter_Cerenkov_CherenkovFiber > 0; includes truthFemScint == 0",
        "calibration_file": str(repo_path(args.calib).relative_to(repo_path("."))),
        "signal_definitions": {
            "S_raw": "counter_Scintillation_ScintFiber",
            "C_raw": "counter_Cerenkov_CherenkovFiber",
            "truth_fem": "truthFemScint",
        },
        "points": rows,
    }
    write_csv(out_dir / "h_over_e_vs_energy.csv", rows)
    write_json(out_dir / "h_over_e_vs_energy.json", payload)
    print(f"Wrote {out_dir / 'h_over_e_vs_energy.csv'}")
    print(f"Wrote {out_dir / 'h_over_e_vs_energy.json'}")
    for row in rows:
        print(
            f"{row['energy_GeV']:g} GeV: "
            f"h/e S={row['h_over_e_S']:.5f} +/- {row['h_over_e_S_err']:.5f}, "
            f"C={row['h_over_e_C']:.5f} +/- {row['h_over_e_C_err']:.5f}"
        )


if __name__ == "__main__":
    main()
