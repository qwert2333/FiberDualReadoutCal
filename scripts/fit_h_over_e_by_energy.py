#!/usr/bin/env python3
"""Fit hadron h/e independently at each beam energy (paper method) and the h/e(E) relation.

Same per-energy h/e and h/e(E) fit as analyze_hadrons.py, without the energy
reconstruction.  Method: S.Y. Jang et al., NIM A 1090 (2026) 171598, Sec. 4.4.1.
"""

from __future__ import annotations

import argparse

from analyze_hadrons import fit_h_over_e_model, group_by_energy, h_over_e_at, h_over_e_by_energy, reco_events
from common import (
    chi_from_h_over_e,
    ensure_dir,
    find_root_files,
    finite,
    read_json,
    repo_path,
    write_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Directory containing hadron ROOT files")
    parser.add_argument("--calib", required=True, help="electron_calibration.json")
    parser.add_argument("--out", required=True, help="Output table directory")
    parser.add_argument("--hoe-model", choices=["log", "linear"], default="log")
    parser.add_argument("--hoe-emin", type=float, default=0.0)
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

    rows = []
    for path in files:
        rows.extend(reco_events(path, a_s, a_c)[0])
    points = h_over_e_by_energy(group_by_energy(rows))
    model_fit = fit_h_over_e_model(points, args.hoe_model, args.hoe_emin)
    for row in points:
        for channel in ("S", "C"):
            row[f"h_over_e_{channel}_model"] = h_over_e_at(model_fit, channel, row["energy_GeV"])
        row["chi_model"] = chi_from_h_over_e(row["h_over_e_S_model"], row["h_over_e_C_model"])

    payload = {
        "selection": "counter_Scintillation_ScintFiber > 0, counter_Cerenkov_CherenkovFiber > 0",
        "calibration_file": str(repo_path(args.calib).relative_to(repo_path("."))),
        "method": "profile of S/E, C/E vs f_em(Eq. 11, seeded with truthFem), fit R = p1 + p0 f_em, h/e = p1/(p0+p1)",
        "h_over_e_model": model_fit,
        "points": points,
    }
    write_csv(out_dir / "h_over_e_vs_energy.csv", points)
    write_json(out_dir / "h_over_e_vs_energy.json", payload)
    print(f"Wrote {out_dir / 'h_over_e_vs_energy.csv'}")
    print(f"Wrote {out_dir / 'h_over_e_vs_energy.json'}")
    for row in points:
        print(
            f"{row['energy_GeV']:g} GeV: "
            f"h/e S={row['h_over_e_S']:.5f} +/- {row['h_over_e_S_err']:.5f}, "
            f"C={row['h_over_e_C']:.5f} +/- {row['h_over_e_C_err']:.5f}, chi(model)={row['chi_model']:.4f}"
        )


if __name__ == "__main__":
    main()
