#!/usr/bin/env python3
"""Dual-readout analysis following S.Y. Jang et al., NIM A 1090 (2026) 171598.

Steps (paper section in brackets):
  1. Calibration with 20 GeV electrons: k = E_beam / <N>            [Sec. 3]
  2. Light yield <N>/E per electron energy                          [Sec. 4.1, Tables 2-3]
  3. EM resolution rms/mean, fit sigma/E = a/sqrt(E) + c            [Sec. 4.3.1, Eq. 7]
     and the combined (S+C)/2 signal                                [Eq. 8]
  4. (h/e)_S, (h/e)_C from 20 GeV pions: profile of R = S/E, C/E
     versus f_em, fit R = p1 + p0 f_em, e/h = 1 + p0/p1             [Sec. 4.4.1, Eq. 11-13]
     f_em is the event-by-event reconstructed value of Eq. 11 (one pass,
     seeded with the truth-f_em h/e); truth f_em is the cross-check
  5. chi = (1 - (h/e)_S) / (1 - (h/e)_C), E = (S - chi C)/(1 - chi) [Eq. 9]
  6. Hadron response and resolution versus energy for S, C, DR      [Sec. 4.4.3]

Not applied: the light-attenuation correction of Eq. 14 (the simulation
has no optical transport), and photosensor PDE.  The S and C signals are the
numbers of produced optical photons, not photoelectrons.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

cache_dir = tempfile.mkdtemp(prefix="paper-dr-plot-cache-")
os.environ.setdefault("MPLCONFIGDIR", cache_dir)
os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import ensure_dir, repo_path, write_csv, write_json


S_COLUMN = "counter_Scintillation_ScintFiber"
C_COLUMN = "counter_Cerenkov_CherenkovFiber"
COLUMNS = [
    ("eventID", "Int_t"),
    ("MCtruth_energy", "Double_t"),
    (S_COLUMN, "Int_t"),
    (C_COLUMN, "Int_t"),
    ("EdepScintFiber", "Double_t"),
    ("EdepCherenkovFiber", "Double_t"),
    ("EdepAbsorber", "Double_t"),
    ("truthFem", "Double_t"),
    ("truthFemScint", "Double_t"),
    ("truthFemCherenkov", "Double_t"),
]

CALIB_ENERGY_GEV = 20.0     # Sec. 3
HOVERE_ENERGY_GEV = 20.0    # Sec. 4.4.1, Fig. 8
PAPER_CU = {
    "calib_S_pe": 22653.4, "calib_C_pe": 1468.81,           # Table 1
    "LY_S_pe_per_GeV": 1136.21, "LY_C_pe_per_GeV": 73.80,   # Tables 2-3
    "em_S_a": 0.132, "em_S_c": 0.015, "em_C_a": 0.193, "em_C_c": 0.005,  # Table 7
    "h_over_e_S": 0.85, "h_over_e_C": 0.35, "chi": 0.23,    # Table 8
    "DR_response": 0.92,                                    # Sec. 4.4.3
}


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------

def export_columns(root_file: Path, out_csv: Path) -> None:
    """Dump scalar branches to CSV.  Only the needed branches are enabled, so
    the large per-fiber vectors are not read."""
    decl = "\n".join(f"  {t} v_{i} = 0;" for i, (_, t) in enumerate(COLUMNS))
    status = "\n".join(f'  t->SetBranchStatus("{n}", 1); t->SetBranchAddress("{n}", &v_{i});'
                       for i, (n, _) in enumerate(COLUMNS))
    line = ' << "," << '.join(f"v_{i}" for i in range(len(COLUMNS)))
    macro = f"""
#include <fstream>
#include <iomanip>
void export_paper_dr() {{
  auto f = TFile::Open("{root_file}");
  if (!f || f->IsZombie()) gSystem->Exit(2);
  auto t = (TTree*)f->Get("eventTree");
  if (!t) gSystem->Exit(3);
  t->SetBranchStatus("*", 0);
{decl}
{status}
  std::ofstream out("{out_csv}");
  out << "{','.join(n for n, _ in COLUMNS)}\\n" << std::setprecision(12);
  for (Long64_t i = 0; i < t->GetEntries(); ++i) {{ t->GetEntry(i); out << {line} << "\\n"; }}
}}
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "export_paper_dr.C"
        path.write_text(macro)
        print(f"+ export {root_file.name}")
        subprocess.run(["root", "-l", "-b", "-q", str(path)], check=True, stdout=subprocess.DEVNULL)


def load_sample(root_file: Path, cache: Path) -> dict[str, np.ndarray]:
    csv = cache / f"{root_file.stem}.csv"
    if not csv.exists() or csv.stat().st_mtime < root_file.stat().st_mtime:
        export_columns(root_file, csv)
    data = np.genfromtxt(csv, delimiter=",", names=True)
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names}


def find_samples(input_dir: Path, particle: str) -> dict[float, Path]:
    out = {}
    for path in sorted(input_dir.glob(f"*_{particle}_*GeV.root")):
        match = re.search(rf"_{re.escape(particle)}_([0-9.]+)GeV\.root$", path.name)
        if match:
            out[float(match.group(1))] = path
    return dict(sorted(out.items()))


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------

def mean_rms(x: np.ndarray) -> tuple[float, float, float, float]:
    """mean, rms, and their statistical errors."""
    n = len(x)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return mu, sd, sd / math.sqrt(n), sd / math.sqrt(2.0 * (n - 1))


def resolution_point(x: np.ndarray) -> tuple[float, float]:
    """rms/mean and its error (paper: rms width divided by mean)."""
    mu, sd, emu, esd = mean_rms(x)
    r = sd / mu
    return r, r * math.hypot(esd / sd, emu / mu)


def fit_linear_sum(energy: np.ndarray, res: np.ndarray, err: np.ndarray) -> dict:
    """Weighted fit of sigma/E = a/sqrt(E) + c (paper Eq. 7)."""
    design = np.vstack([1.0 / np.sqrt(energy), np.ones_like(energy)]).T
    w = 1.0 / err
    coef, *_ = np.linalg.lstsq(design * w[:, None], res * w, rcond=None)
    cov = np.linalg.inv((design * w[:, None]).T @ (design * w[:, None]))
    chi2 = float(np.sum(((design @ coef - res) * w) ** 2))
    return {"a": float(coef[0]), "a_err": float(math.sqrt(cov[0, 0])),
            "c": float(coef[1]), "c_err": float(math.sqrt(cov[1, 1])),
            "chi2": chi2, "ndf": len(energy) - 2}


def fit_quadrature(energy: np.ndarray, res: np.ndarray, err: np.ndarray) -> dict:
    """Weighted fit of (sigma/E)^2 = a^2/E + c^2, for reference."""
    design = np.vstack([1.0 / energy, np.ones_like(energy)]).T
    w = 1.0 / (2.0 * res * err)
    coef, *_ = np.linalg.lstsq(design * w[:, None], res ** 2 * w, rcond=None)
    return {"a": float(math.sqrt(max(coef[0], 0.0))), "c": float(math.sqrt(max(coef[1], 0.0)))}


def profile_line_fit(x: np.ndarray, y: np.ndarray, lo: float, hi: float, nbins: int) -> dict:
    """Fit y = p1 + p0 x to the bin means of y in bins of x (TProfile + pol1)."""
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.digitize(x, edges) - 1
    cx, cy, ce = [], [], []
    for b in range(nbins):
        sel = idx == b
        if sel.sum() < 10:
            continue
        cx.append(float(np.mean(x[sel])))
        cy.append(float(np.mean(y[sel])))
        ce.append(float(np.std(y[sel]) / math.sqrt(sel.sum())))
    cx, cy, ce = map(np.asarray, (cx, cy, ce))
    design = np.vstack([np.ones_like(cx), cx]).T
    w = 1.0 / ce
    coef, *_ = np.linalg.lstsq(design * w[:, None], cy * w, rcond=None)
    cov = np.linalg.inv((design * w[:, None]).T @ (design * w[:, None]))
    p1, p0 = float(coef[0]), float(coef[1])
    h_over_e = p1 / (p0 + p1)            # e/h - 1 = p0/p1
    # error propagation for h/e = p1/(p0+p1)
    g = np.array([p0 / (p0 + p1) ** 2, -p1 / (p0 + p1) ** 2])
    h_err = float(math.sqrt(g @ cov @ g))
    return {"p1": p1, "p0": p0, "h_over_e": h_over_e, "h_over_e_err": h_err,
            "bin_x": cx, "bin_y": cy, "bin_err": ce}


def fem_from_ratio(s: np.ndarray, c: np.ndarray, h_s: float, h_c: float) -> np.ndarray:
    """Paper Eq. 11."""
    q = c / s
    return (h_c - q * h_s) / (q * (1.0 - h_s) - (1.0 - h_c))


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------

def calibrate(electrons: dict[float, dict]) -> dict:
    ev = electrons[CALIB_ENERGY_GEV]
    e_mev = CALIB_ENERGY_GEV * 1000.0
    mean_s, mean_c = float(np.mean(ev[S_COLUMN])), float(np.mean(ev[C_COLUMN]))
    edep = ev["EdepScintFiber"] + ev["EdepCherenkovFiber"] + ev["EdepAbsorber"]
    return {
        "energy_GeV": CALIB_ENERGY_GEV,
        "n_events": len(ev[S_COLUMN]),
        "mean_S_photons": mean_s,
        "mean_C_photons": mean_c,
        "kS_MeV_per_photon": e_mev / mean_s,
        "kC_MeV_per_photon": e_mev / mean_c,
        "mean_Edep_total_over_E": float(np.mean(edep)) / e_mev,
    }


def em_analysis(electrons: dict[float, dict], calib: dict) -> tuple[list[dict], dict]:
    rows = []
    for energy, ev in electrons.items():
        e_mev = energy * 1000.0
        s = calib["kS_MeV_per_photon"] * ev[S_COLUMN]
        c = calib["kC_MeV_per_photon"] * ev[C_COLUMN]
        sc = 0.5 * (s + c)
        edep = ev["EdepScintFiber"] + ev["EdepCherenkovFiber"] + ev["EdepAbsorber"]
        row = {"energy_GeV": energy, "n_events": len(s),
               "LY_S_per_GeV": float(np.mean(ev[S_COLUMN])) / energy,
               "LY_S_per_GeV_err": float(np.std(ev[S_COLUMN]) / math.sqrt(len(s))) / energy,
               "LY_C_per_GeV": float(np.mean(ev[C_COLUMN])) / energy,
               "LY_C_per_GeV_err": float(np.std(ev[C_COLUMN]) / math.sqrt(len(s))) / energy,
               "containment": float(np.mean(edep)) / e_mev}
        for name, x in (("S", s), ("C", c), ("SC", sc)):
            mu, _, emu, _ = mean_rms(x)
            r, er = resolution_point(x)
            row[f"response_{name}"] = mu / e_mev
            row[f"response_{name}_err"] = emu / e_mev
            row[f"res_{name}"] = r
            row[f"res_{name}_err"] = er
        # Eq. 8 expectation for (S+C)/2 when S and C are independent
        row["res_SC_eq8"] = math.sqrt(row["res_S"] ** 2 + row["res_C"] ** 2) / 2.0
        row["rho_SC"] = float(np.corrcoef(s, c)[0, 1])
        rows.append(row)
    return rows, fit_resolutions(rows, ("S", "C", "SC"))


def fit_resolutions(rows: list[dict], names, emin: float = 0.0) -> dict:
    sel = [r for r in rows if r["energy_GeV"] >= emin]
    energy = np.array([r["energy_GeV"] for r in sel])
    fits = {}
    for name in names:
        res = np.array([r[f"res_{name}"] for r in sel])
        err = np.array([r[f"res_{name}_err"] for r in sel])
        fits[name] = {"linear_sum": fit_linear_sum(energy, res, err),
                      "quadrature": fit_quadrature(energy, res, err),
                      "energies_GeV": [float(e) for e in energy]}
    return fits


def hoe_analysis(pion: dict, calib: dict) -> dict:
    e_mev = HOVERE_ENERGY_GEV * 1000.0
    s = calib["kS_MeV_per_photon"] * pion[S_COLUMN] / e_mev
    c = calib["kC_MeV_per_photon"] * pion[C_COLUMN] / e_mev
    truth = pion["truthFem"]

    fit_truth_s = profile_line_fit(truth, s, 0.0, 1.0, 20)
    fit_truth_c = profile_line_fit(truth, c, 0.0, 1.0, 20)

    # Paper: f_em reconstructed event by event with Eq. 11, then one profile fit.
    # Eq. 11 needs (h/e) as input; the truth-f_em values are used as the seed.
    # Iterating to a fixed point is not used: f_em(Eq. 11) depends on C/S only,
    # so only chi is constrained and the individual h/e values drift along a
    # degenerate direction (chi stays put).  The seed dependence is recorded.
    def one_pass(h_s0: float, h_c0: float):
        f = fem_from_ratio(s, c, h_s0, h_c0)
        ok = np.isfinite(f)
        return f, profile_line_fit(f[ok], s[ok], 0.0, 1.0, 20), profile_line_fit(f[ok], c[ok], 0.0, 1.0, 20)

    fem, fit_s, fit_c = one_pass(fit_truth_s["h_over_e"], fit_truth_c["h_over_e"])
    h_s, h_c = fit_s["h_over_e"], fit_c["h_over_e"]
    seeds = []
    for seed in ((fit_truth_s["h_over_e"], fit_truth_c["h_over_e"]),
                 (PAPER_CU["h_over_e_S"], PAPER_CU["h_over_e_C"]), (0.9, 0.5), (0.7, 0.2)):
        _, fs, fc = one_pass(*seed)
        seeds.append({"seed_S": seed[0], "seed_C": seed[1], "h_over_e_S": fs["h_over_e"],
                      "h_over_e_C": fc["h_over_e"],
                      "chi": (1.0 - fs["h_over_e"]) / (1.0 - fc["h_over_e"])})
    return {
        "energy_GeV": HOVERE_ENERGY_GEV,
        "reco": {"S": fit_s, "C": fit_c, "h_over_e_S": h_s, "h_over_e_C": h_c,
                 "chi": (1.0 - h_s) / (1.0 - h_c), "seed_scan": seeds,
                 "fem_reco_mean": float(np.mean(fem)), "fem_reco_minus_truth_rms": float(np.std(fem - truth))},
        "truth": {"S": fit_truth_s, "C": fit_truth_c,
                  "h_over_e_S": fit_truth_s["h_over_e"], "h_over_e_C": fit_truth_c["h_over_e"],
                  "chi": (1.0 - fit_truth_s["h_over_e"]) / (1.0 - fit_truth_c["h_over_e"])},
        "arrays": {"S": s, "C": c, "truth": truth, "fem_reco": fem},
    }


def hadron_analysis(pions: dict[float, dict], calib: dict, chi: float) -> tuple[list[dict], dict]:
    rows = []
    for energy, ev in pions.items():
        e_mev = energy * 1000.0
        s = calib["kS_MeV_per_photon"] * ev[S_COLUMN]
        c = calib["kC_MeV_per_photon"] * ev[C_COLUMN]
        dr = (s - chi * c) / (1.0 - chi)
        edep = ev["EdepScintFiber"] + ev["EdepCherenkovFiber"] + ev["EdepAbsorber"]
        row = {"energy_GeV": energy, "n_events": len(s),
               "containment": float(np.mean(edep)) / e_mev,
               "truthFem_mean": float(np.mean(ev["truthFem"])),
               "rho_SC": float(np.corrcoef(s, c)[0, 1])}
        for name, x in (("S", s), ("C", c), ("DR", dr)):
            mu, _, emu, _ = mean_rms(x)
            r, er = resolution_point(x)
            row[f"response_{name}"] = mu / e_mev
            row[f"response_{name}_err"] = emu / e_mev
            row[f"res_{name}"] = r
            row[f"res_{name}_err"] = er
        rows.append(row)
    return rows, fit_resolutions(rows, ("S", "C", "DR"))


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"wrote {path}")


def plot_resolution(rows, fits, names, labels, title, path: Path, eq8=False) -> None:
    energy = np.array([r["energy_GeV"] for r in rows])
    grid = np.linspace(min(energy) * 0.9, max(energy) * 1.05, 200)
    plt.figure(figsize=(7.0, 5.0))
    for (name, label), color in zip(zip(names, labels), ("tab:red", "tab:blue", "black")):
        y = np.array([r[f"res_{name}"] for r in rows]) * 100
        e = np.array([r[f"res_{name}_err"] for r in rows]) * 100
        f = fits[name]["linear_sum"]
        plt.errorbar(energy, y, yerr=e, fmt="o", color=color, capsize=3)
        plt.plot(grid, 100 * (f["a"] / np.sqrt(grid) + f["c"]), "-", color=color,
                 label=f"{label}: {100*f['a']:.1f}%/√E + {100*f['c']:.2f}%")
    if eq8:
        plt.plot(energy, [100 * r["res_SC_eq8"] for r in rows], "x--", color="gray",
                 label="Eq. 8 expectation for (S+C)/2")
    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("σ/E (rms/mean) [%]")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(path)


def plot_response(rows, names, labels, title, path: Path, ylim=None) -> None:
    energy = np.array([r["energy_GeV"] for r in rows])
    plt.figure(figsize=(7.0, 5.0))
    for (name, label), color, marker in zip(zip(names, labels), ("tab:red", "tab:blue", "black"), "o^s"):
        y = [r[f"response_{name}"] for r in rows]
        e = [r[f"response_{name}_err"] for r in rows]
        plt.errorbar(energy, y, yerr=e, fmt=marker, color=color, capsize=3, label=label)
    plt.axhline(1.0, color="gray", lw=0.8)
    plt.xlabel("Beam energy [GeV]")
    plt.ylabel("Response <E_reco>/E_beam")
    if ylim:
        plt.ylim(*ylim)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    savefig(path)


def plot_hoe(hoe: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for ax, key, title in ((axes[0], "reco", "reconstructed f_em (Eq. 11)"), (axes[1], "truth", "truth f_em")):
        for ch, color in (("C", "tab:blue"), ("S", "tab:red")):
            fit = hoe[key][ch]
            ax.errorbar(fit["bin_x"], fit["bin_y"], yerr=fit["bin_err"], fmt="o", ms=4, color=color)
            xx = np.linspace(0, 1, 50)
            ax.plot(xx, fit["p1"] + fit["p0"] * xx, color=color,
                    label=f"{ch}/E: p1={fit['p1']:.3f}, p0={fit['p0']:.3f}, h/e={fit['h_over_e']:.3f}")
        ax.set_xlabel(f"f_em ({title})")
        ax.set_ylabel("Signal / E_beam")
        ax.set_xlim(0, 1)
        ax.set_title(f"π⁻ {hoe['energy_GeV']:g} GeV, χ = {hoe[key]['chi']:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    savefig(out / "h_over_e_profiles_20GeV.png")


def plot_sc_scatter(pions: dict[float, dict], calib: dict, chi: float, out: Path) -> None:
    energies = [e for e in (10.0, 20.0, 50.0) if e in pions]
    fig, axes = plt.subplots(1, len(energies), figsize=(4.6 * len(energies), 4.4))
    for ax, energy in zip(np.atleast_1d(axes), energies):
        ev = pions[energy]
        s = calib["kS_MeV_per_photon"] * ev[S_COLUMN] / (energy * 1000)
        c = calib["kC_MeV_per_photon"] * ev[C_COLUMN] / (energy * 1000)
        ax.scatter(s, c, s=3, alpha=0.4, c=ev["truthFem"], cmap="viridis", vmin=0, vmax=1)
        # line of constant E_DR = E_beam: C = (S - (1-chi))/chi, rotation point (1, 1)
        xx = np.linspace(min(s.min(), c.min()), 1.05, 50)
        ax.plot(xx, (xx - (1 - chi)) / chi, "k--", lw=1, label=f"E_DR = E, χ = {chi:.3f}")
        ax.plot([0, 1.05], [0, 1.05], color="gray", lw=0.8, label="C = S (f_em = 1)")
        ax.set_xlabel("S / E_beam")
        ax.set_ylabel("C / E_beam")
        ax.set_title(f"π⁻ {energy:g} GeV (colour: truth f_em)")
        ax.set_xlim(left=min(s.min(), 0.5))
        ax.set_ylim(bottom=max(c.min() - 0.05, 0))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")
    savefig(out / "pion_S_vs_C_scatter.png")


def plot_hadron_distributions(pions, calib, chi, out: Path) -> None:
    energies = list(pions)
    ncol = 3
    nrow = math.ceil(len(energies) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.6 * nrow))
    for ax, energy in zip(axes.ravel(), energies):
        ev = pions[energy]
        e_mev = energy * 1000
        s = calib["kS_MeV_per_photon"] * ev[S_COLUMN] / e_mev
        c = calib["kC_MeV_per_photon"] * ev[C_COLUMN] / e_mev
        dr = (s - chi * c) / (1 - chi)
        bins = np.linspace(0, 1.5, 120)
        for x, label, color in ((c, "C", "tab:blue"), (s, "S", "tab:red"), (dr, "DR", "black")):
            ax.hist(x, bins=bins, histtype="step", color=color, lw=1.4, label=label)
        ax.set_title(f"π⁻ {energy:g} GeV")
        ax.set_xlabel("E_reco / E_beam")
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(energies):]:
        ax.axis("off")
    savefig(out / "pion_energy_distributions.png")


# ----------------------------------------------------------------------------

def strip_arrays(obj):
    if isinstance(obj, dict):
        return {k: strip_arrays(v) for k, v in obj.items() if k != "arrays"}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data", help="Directory with e- and pi- ROOT files")
    parser.add_argument("--out", default="outputs/paper_dr", help="Output directory")
    args = parser.parse_args()

    input_dir = repo_path(args.input)
    out = ensure_dir(args.out)
    cache = ensure_dir(out / "cache")

    electrons = {e: load_sample(p, cache) for e, p in find_samples(input_dir, "e-").items()}
    pions = {e: load_sample(p, cache) for e, p in find_samples(input_dir, "pi-").items()}

    calib = calibrate(electrons)
    em_rows, em_fits = em_analysis(electrons, calib)
    em_fits_5 = fit_resolutions(em_rows, ("S", "C", "SC"), emin=5.0)
    hoe = hoe_analysis(pions[HOVERE_ENERGY_GEV], calib)
    chi = hoe["reco"]["chi"]
    had_rows, had_fits = hadron_analysis(pions, calib, chi)
    had_fits_5 = fit_resolutions(had_rows, ("S", "C", "DR"), emin=5.0)
    chi_truth = hoe["truth"]["chi"]
    had_rows_truth, had_fits_truth_5 = None, None
    if abs(chi_truth - chi) > 1e-9:
        had_rows_truth, _ = hadron_analysis(pions, calib, chi_truth)
        had_fits_truth_5 = fit_resolutions(had_rows_truth, ("DR",), emin=5.0)

    write_csv(out / "electron_points.csv", em_rows)
    write_csv(out / "pion_points.csv", had_rows)
    if had_rows_truth:
        write_csv(out / "pion_points_chi_truthfem.csv", had_rows_truth)
    summary = {
        "reference": "S.Y. Jang et al., NIM A 1090 (2026) 171598",
        "signals": {"S": S_COLUMN, "C": C_COLUMN, "unit": "produced optical photons (not p.e.)"},
        "calibration": calib,
        "em_resolution_fits_all": em_fits,
        "em_resolution_fits_E_ge_5GeV": em_fits_5,
        "h_over_e": strip_arrays(hoe),
        "chi_used": chi,
        "hadron_resolution_fits_all": had_fits,
        "hadron_resolution_fits_E_ge_5GeV": had_fits_5,
        "hadron_DR_resolution_with_truthfem_chi_E_ge_5GeV": had_fits_truth_5,
        "paper_Cu": PAPER_CU,
    }
    write_json(out / "summary.json", summary)

    plots = ensure_dir(out / "plots")
    plot_resolution(em_rows, em_fits_5, ("S", "C", "SC"), ("S", "C", "(S+C)/2"),
                    "e⁻ resolution (fit E ≥ 5 GeV)", plots / "electron_resolution.png", eq8=True)
    plot_response(em_rows, ("S", "C"), ("S", "C"), "e⁻ response (calibrated at 20 GeV)",
                  plots / "electron_response.png", ylim=(0.95, 1.05))
    plot_hoe(hoe, plots)
    plot_sc_scatter(pions, calib, chi, plots)
    plot_hadron_distributions(pions, calib, chi, plots)
    plot_response(had_rows, ("S", "C", "DR"), ("S", "C", "DR"), f"π⁻ response, χ = {chi:.3f}",
                  plots / "pion_response.png")
    plot_resolution(had_rows, had_fits_5, ("S", "C", "DR"), ("S", "C", "DR"),
                    f"π⁻ resolution, χ = {chi:.3f} (fit E ≥ 5 GeV)", plots / "pion_resolution.png")

    print(f"\nkS = {calib['kS_MeV_per_photon']:.5g} MeV/photon, kC = {calib['kC_MeV_per_photon']:.5g} MeV/photon")
    print(f"h/e reco: S={hoe['reco']['h_over_e_S']:.4f} C={hoe['reco']['h_over_e_C']:.4f} chi={chi:.4f} "
          "")
    for seed in hoe["reco"]["seed_scan"]:
        print(f"  seed ({seed['seed_S']:.3f}, {seed['seed_C']:.3f}) -> h/e S={seed['h_over_e_S']:.4f} "
              f"C={seed['h_over_e_C']:.4f} chi={seed['chi']:.4f}")
    print(f"h/e truth: S={hoe['truth']['h_over_e_S']:.4f} C={hoe['truth']['h_over_e_C']:.4f} chi={chi_truth:.4f}")
    for title, rows, names in (("e-", em_rows, ("S", "C", "SC")), ("pi-", had_rows, ("S", "C", "DR"))):
        for r in rows:
            parts = "  ".join(f"{n}: R={r[f'response_{n}']:.4f} res={100*r[f'res_{n}']:.2f}%" for n in names)
            print(f"{title} {r['energy_GeV']:>4g} GeV  {parts}  cont={r['containment']:.4f}")
    for title, fits in (("e- (E>=5)", em_fits_5), ("pi- (E>=5)", had_fits_5)):
        for n, f in fits.items():
            ls, q = f["linear_sum"], f["quadrature"]
            print(f"{title} {n}: a={100*ls['a']:.2f}+-{100*ls['a_err']:.2f}% c={100*ls['c']:.2f}+-{100*ls['c_err']:.2f}% "
                  f"chi2/ndf={ls['chi2']:.1f}/{ls['ndf']}  | quad a={100*q['a']:.2f}% c={100*q['c']:.2f}%")


if __name__ == "__main__":
    main()
