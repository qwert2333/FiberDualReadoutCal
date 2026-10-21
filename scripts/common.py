#!/usr/bin/env python3
"""Shared utilities for the dual-readout analysis workflow."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

EVENT_COLUMNS = [
    "eventID",
    "particle",
    "MCtruth_energy",
    "EdepScintFiber",
    "EdepCherenkovFiber",
    "EdepAbsorber",
    "Nph_Cherenkov",
    "Nph_Scintillation",
    "counter_Cerenkov_ScintFiber",
    "counter_Scintillation_ScintFiber",
    "counter_Cerenkov_CherenkovFiber",
    "counter_Scintillation_CherenkovFiber",
    "truthEdepScintTotal",
    "truthEdepScintEM",
    "truthEdepScintNonEM",
    "truthFemScint",
    "truthEdepCherenkovTotal",
    "truthEdepCherenkovEM",
    "truthEdepCherenkovNonEM",
    "truthFemCherenkov",
    "truthEdepAbsorberTotal",
    "truthEdepAbsorberEM",
    "truthEdepAbsorberNonEM",
    "truthFemAbsorber",
    "truthEdepTotal",
    "truthEdepEM",
    "truthEdepNonEM",
    "truthFem",
]

NUMERIC_COLUMNS = set(EVENT_COLUMNS) - {"particle"}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    path = repo_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load the project configs.

    The files use JSON syntax in .yaml files. JSON is valid YAML, and this keeps
    the workflow independent of PyYAML in the current physics environment.
    """

    path = repo_path(path)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Could not parse {path}. These YAML files intentionally use JSON "
            "syntax so no PyYAML dependency is required."
        ) from exc


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = repo_path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(repo_path(path).read_text())


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path = repo_path(path)
    ensure_dir(path.parent)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> List[Dict[str, Any]]:
    path = repo_path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            out: Dict[str, Any] = {}
            for key, value in row.items():
                out[key] = parse_value(value)
            rows.append(out)
    return rows


def parse_value(value: str) -> Any:
    if value is None:
        return value
    value = value.strip()
    if value == "":
        return value
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        return float(value)
    except ValueError:
        return value


def particle_tag(particle: str) -> str:
    return particle.replace("+", "p").replace("-", "m")


def energy_tag(energy_gev: float) -> str:
    if float(energy_gev).is_integer():
        return f"{int(energy_gev)}GeV"
    return f"{energy_gev:g}GeV"


def root_energy_label(energy_gev: float) -> str:
    if float(energy_gev).is_integer():
        return f"{int(energy_gev)}GeV"
    return f"{energy_gev:g}GeV"


def sample_name(role: str, particle: str, energy_gev: float) -> str:
    return f"{role}_{particle_tag(particle)}_{energy_tag(energy_gev)}"


def iter_samples(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for group in config["samples"]:
        for energy in group["energies_gev"]:
            yield {
                "role": group["role"],
                "particle": group["particle"],
                "energy_gev": float(energy),
                "events": int(group["events"]),
            }


def config_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    dirs = config.get("output_dirs", {})
    return {
        "macros": ensure_dir(dirs.get("macros", "outputs/dual_readout/macros")),
        "root": ensure_dir(dirs.get("root", "outputs/dual_readout/root")),
        "tables": ensure_dir(dirs.get("tables", "outputs/dual_readout/tables")),
        "plots": ensure_dir(dirs.get("plots", "outputs/dual_readout/plots")),
    }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def rms(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))


def stderr(values: Sequence[float]) -> float:
    return rms(values) / math.sqrt(len(values)) if values else math.nan


def linear_slope_through_origin(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    denom = sum(x * x for x in x_values)
    if denom <= 0:
        return math.nan
    return sum(x * y for x, y in zip(x_values, y_values)) / denom


def fit_response_vs_fem(fem: Sequence[float], response: Sequence[float]) -> Dict[str, float]:
    """Fit response = fem + h_over_e * (1 - fem)."""

    x = [1.0 - f for f in fem]
    y = [r - f for f, r in zip(fem, response)]
    h_over_e = linear_slope_through_origin(x, y)
    residuals = [yy - h_over_e * xx for xx, yy in zip(x, y)]
    dof = max(len(x) - 1, 1)
    sigma2 = sum(r * r for r in residuals) / dof
    denom = sum(xx * xx for xx in x)
    err = math.sqrt(sigma2 / denom) if denom > 0 else math.nan
    return {
        "h_over_e": h_over_e,
        "h_over_e_err": err,
        "fit_rms": math.sqrt(sum(r * r for r in residuals) / len(residuals)) if residuals else math.nan,
        "n_fit": len(x),
    }


def run_command(command: Sequence[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=str(cwd or REPO_ROOT), check=True)


def export_root_to_csv(root_file: str | Path, out_csv: str | Path, columns: Sequence[str] = EVENT_COLUMNS) -> None:
    """Export eventTree branches to CSV through ROOT CLI.

    PyROOT import is not reliable in the current conda env, but the `root`
    executable is available. This keeps the public scripts Python-driven while
    using ROOT for file I/O.
    """

    root_file = repo_path(root_file)
    out_csv = repo_path(out_csv)
    ensure_dir(out_csv.parent)
    macro = f"""
#include <fstream>
#include <iomanip>
#include <vector>
void export_tree() {{
  auto f = TFile::Open("{root_file}");
  if (!f || f->IsZombie()) {{ std::cerr << "Could not open ROOT file\\n"; gSystem->Exit(2); }}
  auto t = (TTree*)f->Get("eventTree");
  if (!t) {{ std::cerr << "Missing eventTree\\n"; gSystem->Exit(3); }}
  Int_t eventID = 0, Nph_Cherenkov = 0, Nph_Scintillation = 0, leakageNParticles = 0;
  Char_t particle[64];
  Double_t MCtruth_energy = 0, EdepCrystal = 0, EdepFiberCore = 0, EdepFiberClad = 0, EdepCarbonFrame = 0;
  Double_t truthEdepCrystalTotal = 0, truthEdepCrystalEM = 0, truthEdepCrystalNonEM = 0, truthFemCrystal = 0;
  Double_t leakageEnergy = 0;
  t->SetBranchAddress("eventID", &eventID);
  t->SetBranchAddress("particle", particle);
  t->SetBranchAddress("MCtruth_energy", &MCtruth_energy);
  t->SetBranchAddress("EdepCrystal", &EdepCrystal);
  t->SetBranchAddress("EdepFiberCore", &EdepFiberCore);
  t->SetBranchAddress("EdepFiberClad", &EdepFiberClad);
  t->SetBranchAddress("EdepCarbonFrame", &EdepCarbonFrame);
  t->SetBranchAddress("Nph_Cherenkov", &Nph_Cherenkov);
  t->SetBranchAddress("Nph_Scintillation", &Nph_Scintillation);
  t->SetBranchAddress("truthEdepCrystalTotal", &truthEdepCrystalTotal);
  t->SetBranchAddress("truthEdepCrystalEM", &truthEdepCrystalEM);
  t->SetBranchAddress("truthEdepCrystalNonEM", &truthEdepCrystalNonEM);
  t->SetBranchAddress("truthFemCrystal", &truthFemCrystal);
  t->SetBranchAddress("leakageEnergy", &leakageEnergy);
  t->SetBranchAddress("leakageNParticles", &leakageNParticles);
  std::ofstream out("{out_csv}");
  out << "{",".join(columns)}\\n";
  out << std::setprecision(17);
  for (Long64_t i = 0; i < t->GetEntries(); ++i) {{
    t->GetEntry(i);
    out << eventID << "," << particle << "," << MCtruth_energy << "," << EdepCrystal << ","
        << EdepFiberCore << "," << EdepFiberClad << "," << EdepCarbonFrame << ","
        << Nph_Cherenkov << "," << Nph_Scintillation << ","
        << truthEdepCrystalTotal << "," << truthEdepCrystalEM << "," << truthEdepCrystalNonEM << ","
        << truthFemCrystal << "," << leakageEnergy << "," << leakageNParticles << "\\n";
  }}
}}
"""
    with tempfile.TemporaryDirectory() as tmp:
        macro_path = Path(tmp) / "export_tree.C"
        macro_path.write_text(macro)
        run_command(["root", "-l", "-b", "-q", f"{macro_path}"], cwd=REPO_ROOT)


def load_events(root_file: str | Path, cache_dir: str | Path | None = None) -> List[Dict[str, Any]]:
    root_file = repo_path(root_file)
    cache_dir = ensure_dir(cache_dir or root_file.parent / "_csv_cache")
    csv_file = cache_dir / f"{root_file.stem}.csv"
    if not csv_file.exists() or csv_file.stat().st_mtime < root_file.stat().st_mtime:
        export_root_to_csv(root_file, csv_file)
    return read_csv(csv_file)


def find_root_files(input_dir: str | Path, role: str | None = None, particle: str | None = None) -> List[Path]:
    input_dir = repo_path(input_dir)
    files = sorted(input_dir.glob("*.root"))
    if particle:
        tag = f"_{particle}_"
        files = [path for path in files if tag in path.name]
    if role:
        role_tag = f"{role}_"
        files = [path for path in files if path.name.startswith(role_tag)]
    return files


def require_columns(events: Sequence[Dict[str, Any]], columns: Sequence[str], source: Path) -> None:
    if not events:
        raise SystemExit(f"No events found in {source}")
    missing = [col for col in columns if col not in events[0]]
    if missing:
        raise SystemExit(f"{source} is missing required columns: {', '.join(missing)}")


def finite(value: float) -> bool:
    return math.isfinite(float(value))


# Override the legacy exporter above for the current FiberDualReadoutCal ntuple.
# The logical default photon columns intentionally map to fiber-type-specific counters.
BRANCH_ALIASES = {
    "Nph_Cherenkov": "counter_Cerenkov_CherenkovFiber",
    "Nph_Scintillation": "counter_Scintillation_ScintFiber",
}


def _cpp_var_name(column: str) -> str:
    return re.sub(r"\W", "_", column)


def export_root_to_csv(root_file: str | Path, out_csv: str | Path, columns: Sequence[str] = EVENT_COLUMNS) -> None:
    root_file = repo_path(root_file)
    out_csv = repo_path(out_csv)
    ensure_dir(out_csv.parent)

    int_columns = {
        "eventID",
        "Nph_Cherenkov",
        "Nph_Scintillation",
        "counter_Cerenkov_ScintFiber",
        "counter_Scintillation_ScintFiber",
        "counter_Cerenkov_CherenkovFiber",
        "counter_Scintillation_CherenkovFiber",
    }
    declarations = []
    branch_addresses = []
    output_terms = []
    for column in columns:
        branch = BRANCH_ALIASES.get(column, column)
        var = _cpp_var_name(column)
        if column == "particle":
            declarations.append(f"  Char_t {var}[64];")
            branch_addresses.append(f'  t->SetBranchAddress("{branch}", {var});')
        elif column in int_columns:
            declarations.append(f"  Int_t {var} = 0;")
            branch_addresses.append(f'  t->SetBranchAddress("{branch}", &{var});')
        else:
            declarations.append(f"  Double_t {var} = 0;")
            branch_addresses.append(f'  t->SetBranchAddress("{branch}", &{var});')
        output_terms.append(var)

    output_line = ' << "," << '.join(output_terms)
    macro = f"""
#include <fstream>
#include <iomanip>
#include <vector>
void export_tree() {{
  auto f = TFile::Open("{root_file}");
  if (!f || f->IsZombie()) {{ std::cerr << "Could not open ROOT file\\n"; gSystem->Exit(2); }}
  auto t = (TTree*)f->Get("eventTree");
  if (!t) {{ std::cerr << "Missing eventTree\\n"; gSystem->Exit(3); }}
{chr(10).join(declarations)}
{chr(10).join(branch_addresses)}
  std::ofstream out("{out_csv}");
  out << "{','.join(columns)}\\n";
  out << std::setprecision(17);
  for (Long64_t i = 0; i < t->GetEntries(); ++i) {{
    t->GetEntry(i);
    out << {output_line} << "\\n";
  }}
}}
"""
    with tempfile.TemporaryDirectory() as tmp:
        macro_path = Path(tmp) / "export_tree.C"
        macro_path.write_text(macro)
        run_command(["root", "-l", "-b", "-q", f"{macro_path}"], cwd=REPO_ROOT)
