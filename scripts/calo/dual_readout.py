"""Dual-readout cell aggregation and reconstructed event features.

The full detector grid is used; no geometric crop is applied.

Fiber-calorimeter readout (src/detector.cc, src/MyEventAction.cc): one entry
per fiber segment with

    cellID = (iz * N + iy) * N + ix,   N = geo_NFiberPerSide,

and the fiber kind in the parallel vecFiberType branch (0 = S, 1 = C). Fibers
form a checkerboard: S fibers have (ix + iy) even, C fibers odd. The S signal
is vecNScint of S fibers and the C signal is vecNChren of C fibers, matching
counter_Scintillation_ScintFiber / counter_Cerenkov_CherenkovFiber used by
RunFullDualReadout.cpp.
"""

import csv
import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FIBER_TYPE_S = 0
FIBER_TYPE_C = 1
GEOMETRY_BRANCHES = ("geo_NFiberPerSide", "geo_ZSegNum")
REQUIRED_TEST_ENERGIES_GEV = {5, 10, 20, 30, 40, 50}
AUX_NAMES = (
    "log1p_S_total_GeV",
    "log1p_C_total_GeV",
    "log1p_n_active_S",
    "log1p_n_active_C",
    "C_over_S",
    "S_minus_C_over_sum",
    "h_over_e_S_reco",
    "h_over_e_C_reco",
    "dualreadout_chi_reco",
    "signed_log1p_E_DR_reco_GeV",
    "E_DR_over_S",
)


@dataclass(frozen=True)
class DualReadoutGeometry:
    # nx = ny = fibers per side (geo_NFiberPerSide), nz = z segments (geo_ZSegNum).
    nx: int
    ny: int
    nz: int

    def __post_init__(self):
        if self.nx != self.ny:
            raise ValueError("Fiber cellID encoding requires nx == ny (fibers per side)")


@dataclass(frozen=True)
class DualReadoutCalibration:
    s_gev_per_count: float
    c_gev_per_count: float
    h_s_intercept: float
    h_s_slope_per_gev: float
    h_c_intercept: float
    h_c_slope_per_gev: float
    max_reco_energy_gev: float = 200.0


@dataclass(frozen=True)
class BaselineProfile:
    name: str
    geometry: DualReadoutGeometry
    calibration: DualReadoutCalibration
    calibration_table: str
    hovere_fit_table: str


def _read_channel_rows(path, value_columns):
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for channel in ("S", "C"):
        selected = [row for row in rows if row.get("channel", "").upper() == channel]
        if not selected:
            raise RuntimeError(f"No {channel} row in {path}")
        result[channel] = {name: float(selected[0][name]) for name in value_columns}
        for row in selected[1:]:
            for name, expected in result[channel].items():
                if not np.isclose(float(row[name]), expected, rtol=0.0, atol=5.0e-7):
                    raise RuntimeError(f"Inconsistent {channel} {name} values in {path}")
    return result


def read_fiber_geometry(paths):
    """Read (fibers per side, z segments) from the geo_* branches of ROOT files."""
    import uproot

    dimensions = set()
    for path in paths:
        with uproot.open(path) as source:
            arrays = source["eventTree"].arrays(GEOMETRY_BRANCHES, library="np")
        for name in GEOMETRY_BRANCHES:
            if np.unique(arrays[name]).size != 1:
                raise RuntimeError(f"{name} is not constant within {path}")
        dimensions.add((
            int(arrays["geo_NFiberPerSide"][0]), int(arrays["geo_ZSegNum"][0])
        ))
    if len(dimensions) != 1:
        raise RuntimeError(f"Mixed detector geometries: {sorted(dimensions)}")
    return dimensions.pop()


def load_baseline_profile(input_dir, reference_dir=None):
    """Read geometry from geo_* branches and load calibration tables."""
    input_dir = Path(input_dir)
    pion_files = sorted(input_dir.glob("*_pi-_*.root"))
    if not pion_files:
        raise RuntimeError(f"No pion ROOT files found in {input_dir}")
    n_fiber, nz = read_fiber_geometry(pion_files)
    geometry = DualReadoutGeometry(nx=n_fiber, ny=n_fiber, nz=nz)

    if reference_dir is None:
        reference_dir = input_dir.parent / "full_dualreadout" / input_dir.name
    reference_dir = Path(reference_dir)
    calibration_table = reference_dir / "tables" / "electron_em_calibration.csv"
    hovere_fit_table = reference_dir / "tables" / "pion_hovere_energy_fit.csv"
    calibration_rows = _read_channel_rows(
        calibration_table, ("calibration_MeV_per_readout",)
    )
    hovere_rows = _read_channel_rows(
        hovere_fit_table, ("intercept", "slope_per_GeV")
    )
    calibration = DualReadoutCalibration(
        s_gev_per_count=calibration_rows["S"]["calibration_MeV_per_readout"] / 1000.0,
        c_gev_per_count=calibration_rows["C"]["calibration_MeV_per_readout"] / 1000.0,
        h_s_intercept=hovere_rows["S"]["intercept"],
        h_s_slope_per_gev=hovere_rows["S"]["slope_per_GeV"],
        h_c_intercept=hovere_rows["C"]["intercept"],
        h_c_slope_per_gev=hovere_rows["C"]["slope_per_GeV"],
    )
    return BaselineProfile(
        name=input_dir.name,
        geometry=geometry,
        calibration=calibration,
        calibration_table=str(calibration_table.resolve()),
        hovere_fit_table=str(hovere_fit_table.resolve()),
    )


def discover_file_roles(input_dir):
    train_files = sorted(glob.glob(os.path.join(
        input_dir, "*_pi-_1-60GeV_Train.root"
    )))
    validation_files = sorted(glob.glob(os.path.join(
        input_dir, "*_pi-_1-60GeV_Valid.root"
    )))
    if not train_files:
        raise RuntimeError(f"No *_pi-_1-60GeV_Train.root files found in {input_dir}")
    if not validation_files:
        raise RuntimeError(f"No *_pi-_1-60GeV_Valid.root files found in {input_dir}")

    test_by_energy = {}
    for path in sorted(glob.glob(os.path.join(input_dir, "*_pi-_*.root"))):
        match = re.search(r"_pi-_([0-9]+)GeV\.root$", os.path.basename(path))
        if not match:
            continue
        energy = int(match.group(1))
        if energy in test_by_energy:
            raise RuntimeError(
                f"Multiple fixed-energy pion files found at {energy} GeV"
            )
        test_by_energy[energy] = path
    missing = sorted(REQUIRED_TEST_ENERGIES_GEV - set(test_by_energy))
    if missing:
        raise RuntimeError(f"Missing required fixed-energy pion samples: {missing}")
    return train_files, validation_files, [
        test_by_energy[energy] for energy in sorted(test_by_energy)
    ]


def decode_fiber_cell_ids(cell_ids, fiber_type, geo):
    """Return ix, iy, iz and a validity mask for fiber-segment cellIDs.

    An entry is valid when the cellID lies inside the grid, the fiber type is
    S or C, and the type agrees with the checkerboard parity of (ix, iy).
    """
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    fiber_type = np.asarray(fiber_type, dtype=np.int64)
    n = geo.nx
    ix = cell_ids % n
    iy = (cell_ids // n) % n
    iz = cell_ids // (n * n)
    valid = (
        (cell_ids >= 0) & (iz < geo.nz)
        & ((fiber_type == FIBER_TYPE_S) | (fiber_type == FIBER_TYPE_C))
        & ((ix + iy) % 2 == fiber_type)
    )
    return ix, iy, iz, valid


def _fill_channel(cell_ids, fiber_type, counts, fiber_type_id, scale, geo, out):
    ix, iy, iz, valid = decode_fiber_cell_ids(cell_ids, fiber_type, geo)
    positive = counts > 0
    if np.any(positive & ~valid):
        raise RuntimeError("Positive S/C signal has an invalid fiber cellID")
    selected = (np.asarray(fiber_type) == fiber_type_id) & positive
    if not np.any(selected):
        return
    values = counts[selected].astype(np.float64, copy=False) * float(scale)
    # Layout is [x transverse, z longitudinal, y transverse].
    np.add.at(out, (ix[selected], iz[selected], iy[selected]), values)


def reconstruct_standard_dr(s_total, c_total, calibration, n_iter=6):
    """Evaluate h/e(E), chi(E), and DR using only reconstructed S and C."""
    energy = float(np.clip(s_total, 0.0, calibration.max_reco_energy_gev))
    h_s = h_c = chi = 0.0
    e_dr = energy
    for _ in range(int(n_iter)):
        h_s = calibration.h_s_intercept + calibration.h_s_slope_per_gev * energy
        h_c = calibration.h_c_intercept + calibration.h_c_slope_per_gev * energy
        chi = (1.0 - h_s) / max(1.0 - h_c, 1.0e-8)
        denom = 1.0 - chi
        if abs(denom) < 1.0e-8:
            raise RuntimeError("Invalid reconstructed dual-readout denominator")
        e_dr = (float(s_total) - chi * float(c_total)) / denom
        energy = float(np.clip(e_dr, 0.0, calibration.max_reco_energy_gev))
    return float(h_s), float(h_c), float(chi), float(e_dr)


def build_aux(mode, s_voxel, c_voxel, calibration):
    s_total = float(np.sum(s_voxel, dtype=np.float64))
    n_s = int(np.count_nonzero(s_voxel))
    aux = np.zeros(len(AUX_NAMES), dtype=np.float32)
    aux[0] = np.log1p(max(s_total, 0.0))
    aux[2] = np.log1p(n_s)
    diagnostics = {"S_total_GeV": s_total, "n_active_S": n_s}

    if mode == "s_only":
        diagnostics.update({"C_total_GeV": 0.0, "n_active_C": 0, "E_DR_reco_GeV": 0.0})
        return aux, diagnostics
    if mode != "sc_full":
        raise ValueError(f"Unknown dual-readout mode: {mode}")

    c_total = float(np.sum(c_voxel, dtype=np.float64))
    n_c = int(np.count_nonzero(c_voxel))
    h_s, h_c, chi, e_dr = reconstruct_standard_dr(s_total, c_total, calibration)
    eps = 1.0e-8
    aux[1] = np.log1p(max(c_total, 0.0))
    aux[3] = np.log1p(n_c)
    aux[4] = c_total / max(s_total, eps)
    aux[5] = (s_total - c_total) / max(s_total + c_total, eps)
    aux[6:9] = (h_s, h_c, chi)
    aux[9] = np.sign(e_dr) * np.log1p(abs(e_dr))
    aux[10] = e_dr / max(s_total, eps)
    diagnostics.update({
        "C_total_GeV": c_total,
        "n_active_C": n_c,
        "h_over_e_S_reco": h_s,
        "h_over_e_C_reco": h_c,
        "chi_reco": chi,
        "E_DR_reco_GeV": e_dr,
    })
    return aux, diagnostics


def build_event_input(cell_ids, fiber_type, n_scint, n_cherenkov, mode, geo, calibration,
                      cell_signal_scale_gev=1.0e-3):
    cell_ids = np.asarray(cell_ids, dtype=np.int64)
    fiber_type = np.asarray(fiber_type, dtype=np.int64)
    n_scint = np.asarray(n_scint, dtype=np.int64)
    if cell_ids.size != n_scint.size:
        raise RuntimeError("vecCellID and vecNScint lengths differ")
    if cell_ids.size != fiber_type.size:
        raise RuntimeError("vecCellID and vecFiberType lengths differ")

    s_voxel = np.zeros((geo.nx, geo.nz, geo.ny), dtype=np.float32)
    c_voxel = np.zeros_like(s_voxel)
    _fill_channel(cell_ids, fiber_type, n_scint, FIBER_TYPE_S,
                  calibration.s_gev_per_count, geo, s_voxel)

    if mode == "sc_full":
        if n_cherenkov is None:
            raise RuntimeError("sc_full requires vecNChren")
        n_cherenkov = np.asarray(n_cherenkov, dtype=np.int64)
        if cell_ids.size != n_cherenkov.size:
            raise RuntimeError("vecCellID and vecNChren lengths differ")
        _fill_channel(cell_ids, fiber_type, n_cherenkov, FIBER_TYPE_C,
                      calibration.c_gev_per_count, geo, c_voxel)
    elif mode != "s_only":
        raise ValueError(f"Unknown dual-readout mode: {mode}")

    aux, diagnostics = build_aux(mode, s_voxel, c_voxel, calibration)
    stacked = np.stack([s_voxel, c_voxel])
    stacked = np.log1p(stacked / float(cell_signal_scale_gev)).astype(np.float32)
    return stacked, aux, diagnostics


def event_uid(source_file, entry_index, event_id):
    """Return a stable event key when ROOT eventID values are sample-local."""
    source_name = Path(source_file).name
    return f"{source_name}#{int(entry_index)}:{int(event_id)}"
