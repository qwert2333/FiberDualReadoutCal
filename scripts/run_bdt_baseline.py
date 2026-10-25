#!/usr/bin/env python3
"""Train non-CNN BDT control regressors for the dual-readout energy study.

Each feature set is a gradient-boosted regressor on per-event global and
shower-shape summaries computed from the same full-grid, cell-aggregated S/C
signal as the CNN input.  The data roles, loss and resolution metric are the
CNN ones: Train fits the trees, Valid selects the number of boosting
iterations, fixed-energy pion files are used only for evaluation, the loss is
the report-style relative MSE, and resolution is sigma68/MPV.

Feature sets whose names start with ``s_`` contain no Cherenkov-derived input.
The ladder separates what simple global quantities (S, C, active-cell counts,
a few shape summaries) already achieve from what the CNN adds.

The continuous Train spectrum defines the regression prior.  Fixed-energy
points close to its edges (for 1-60 GeV training, the lowest test energies) are
compressed toward the interior; ``distance_to_train_edge_GeV`` and
``corr_pred_S_fixed_E`` in the metrics table flag this.
"""

import argparse
import json
import re
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calo.dr_metrics import energy_metrics
from calo.dual_readout import (
    FIBER_TYPE_C,
    FIBER_TYPE_S,
    decode_fiber_cell_ids,
    discover_file_roles,
    event_uid,
    load_baseline_profile,
)


TREE = "eventTree"
BRANCHES = ("eventID", "MCtruth_energy", "vecCellID", "vecFiberType", "vecNScint", "vecNChren")
CELL_ID_DECODING = "fiber: cellID=(iz*N+iy)*N+ix, vecFiberType 0=S 1=C"
C_FEATURES = {"C_GeV", "n_active_C"}
S_SHAPE = (
    "z_cog_layer",
    "z_start_layer",
    "late_fraction_S",
    "r_rms_cells",
    "dense_fraction_S",
    "n_dense_S",
)
FEATURE_SETS = {
    "s_total": ("S_GeV",),
    "sc_total": ("S_GeV", "C_GeV"),
    "s_hits": ("S_GeV", "n_active_S"),
    "sc_hits": ("S_GeV", "C_GeV", "n_active_S", "n_active_C"),
    "s_shape": ("S_GeV", "n_active_S") + S_SHAPE,
    "sc_shape": ("S_GeV", "C_GeV", "n_active_S", "n_active_C") + S_SHAPE,
}
FEATURE_COLUMNS = ("S_GeV", "C_GeV", "n_active_S", "n_active_C") + S_SHAPE
IDENTITY_COLUMNS = ("source_file", "entry_index", "event_id", "true_energy_GeV")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("outputs_dual_readout/standard_dual_readout/data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs_dual_readout/bdt_baseline/Sapphire_5-5-5mm_N30x30x100"),
    )
    parser.add_argument(
        "--feature-sets", nargs="+", choices=tuple(FEATURE_SETS),
        default=list(FEATURE_SETS),
    )
    parser.add_argument("--max-events-per-file", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--events-per-task", type=int, default=10000)
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument("--dense-threshold-gev", type=float, default=0.02)
    parser.add_argument("--late-layers", type=int, default=20)
    parser.add_argument("--start-fraction", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--l2-regularization", type=float, default=0.0)
    parser.add_argument("--denom-min", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--smoke-split-fixed-energy", action="store_true",
        help="Smoke test only: split every fixed-energy pion file by event into "
             "Train/Valid/Test instead of using *_1-60GeV_Train/Valid files. "
             "Training on discrete energies is not a physics result.",
    )
    parser.add_argument("--split-fractions", type=float, nargs=2, default=(0.6, 0.2),
                        metavar=("TRAIN", "VALID"),
                        help="Per-file Train and Valid fractions; the rest is Test")
    parser.add_argument("--split-seed", type=int, default=170510363)
    return parser.parse_args()


def _aggregate_cells(event_index, local_cell, values, n_cells):
    """Sum repeated entries of one physical cell, keeping positive cells only."""
    positive = values > 0
    key = event_index[positive].astype(np.int64) * n_cells + local_cell[positive]
    unique, inverse = np.unique(key, return_inverse=True)
    return unique // n_cells, unique % n_cells, np.bincount(inverse, values[positive])


def event_features(event_index, cell_ids, fiber_type, n_scint, n_cherenkov, n_events, geo,
                   calibration, dense_threshold_gev=0.02, late_layers=20,
                   start_fraction=0.05):
    """Per-event summaries of the full-grid, cell-aggregated S and C signal.

    Inputs are flat hit-level arrays; ``event_index`` maps each entry to its
    event in ``range(n_events)``.  No geometric crop is applied.
    S is vecNScint of S fibers and C is vecNChren of C fibers.
    """
    fiber_type = np.asarray(fiber_type, dtype=np.int64)
    ix, iy, iz, valid = decode_fiber_cell_ids(cell_ids, fiber_type, geo)
    n_scint = np.where(fiber_type == FIBER_TYPE_S, np.asarray(n_scint, dtype=np.float64), 0.0)
    n_cherenkov = np.where(
        fiber_type == FIBER_TYPE_C, np.asarray(n_cherenkov, dtype=np.float64), 0.0
    )
    if np.any(~valid & ((n_scint > 0) | (n_cherenkov > 0))):
        raise RuntimeError("Positive S/C signal has an invalid fiber cellID")

    nx, ny, nz = geo.nx, geo.ny, geo.nz
    local = np.where(valid, (ix * nz + iz) * ny + iy, 0)
    n_cells = nx * ny * nz

    s_event, s_cell, s_value = _aggregate_cells(
        event_index, local, np.where(valid, n_scint, 0.0) * calibration.s_gev_per_count, n_cells
    )
    c_event, _, c_value = _aggregate_cells(
        event_index, local, np.where(valid, n_cherenkov, 0.0) * calibration.c_gev_per_count, n_cells
    )
    s_x = s_cell // (nz * ny)
    s_z = (s_cell // ny) % nz
    s_y = s_cell % ny

    s_total = np.bincount(s_event, s_value, n_events)
    safe_total = np.maximum(s_total, 1.0e-12)
    weighted = lambda values: np.bincount(s_event, s_value * values, n_events) / safe_total
    x_mean, y_mean = weighted(s_x), weighted(s_y)
    r2 = (s_x - x_mean[s_event]) ** 2 + (s_y - y_mean[s_event]) ** 2
    dense = s_value > dense_threshold_gev

    profile = np.zeros((n_events, nz))
    np.add.at(profile, (s_event, s_z), s_value)
    cumulative = np.cumsum(profile, axis=1) / safe_total[:, None]
    z_start = np.where(s_total > 0, np.argmax(cumulative > start_fraction, axis=1), 0)

    return {
        "S_GeV": s_total,
        "C_GeV": np.bincount(c_event, c_value, n_events),
        "n_active_S": np.bincount(s_event, minlength=n_events).astype(np.float64),
        "n_active_C": np.bincount(c_event, minlength=n_events).astype(np.float64),
        "z_cog_layer": weighted(s_z),
        "z_start_layer": z_start.astype(np.float64),
        "late_fraction_S": weighted((s_z >= nz - late_layers).astype(np.float64)),
        "r_rms_cells": np.sqrt(weighted(r2)),
        "dense_fraction_S": weighted(dense.astype(np.float64)),
        "n_dense_S": np.bincount(s_event[dense], minlength=n_events).astype(np.float64),
    }


def _read_entry_range(task):
    path, start, stop, geo, calibration, options = task
    parts = []
    with uproot.open(path) as source:
        for arrays in source[TREE].iterate(
            BRANCHES, entry_start=start, entry_stop=stop, step_size=2000, library="ak"
        ):
            n = len(arrays)
            counts = ak.num(arrays["vecCellID"]).to_numpy()
            features = event_features(
                np.repeat(np.arange(n), counts),
                ak.flatten(arrays["vecCellID"]).to_numpy(),
                ak.flatten(arrays["vecFiberType"]).to_numpy(),
                ak.flatten(arrays["vecNScint"]).to_numpy(),
                ak.flatten(arrays["vecNChren"]).to_numpy(),
                n, geo, calibration, **options,
            )
            parts.append(pd.DataFrame({
                "source_file": Path(path).name,
                "entry_index": np.arange(start, start + n, dtype=np.int64),
                "event_id": arrays["eventID"].to_numpy().astype(np.int64),
                "true_energy_GeV": arrays["MCtruth_energy"].to_numpy().astype(np.float64) / 1000.0,
                **features,
            }))
            start += n
    return pd.concat(parts, ignore_index=True)


def load_features(paths, geo, calibration, options, args, cache_dir):
    """Return per-event features, reusing a cache built with identical settings."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature = json.dumps({
        "cell_id_decoding": CELL_ID_DECODING,
        "options": options, "geometry": asdict(geo), "calibration": asdict(calibration),
        "max_events_per_file": args.max_events_per_file,
    }, sort_keys=True)
    frames, tasks, pending = {}, [], []
    for path in paths:
        cache = cache_dir / (Path(path).stem + ".csv.gz")
        meta = cache.with_suffix(".json")
        if (not args.recompute_features and cache.exists() and meta.exists()
                and meta.read_text() == signature):
            frames[path] = pd.read_csv(cache)
            continue
        with uproot.open(path) as source:
            n_entries = source[TREE].num_entries
        if args.max_events_per_file > 0:
            n_entries = min(n_entries, args.max_events_per_file)
        for start in range(0, n_entries, args.events_per_task):
            tasks.append((path, start, min(start + args.events_per_task, n_entries),
                          geo, calibration, options))
        pending.append(path)
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(_read_entry_range, tasks))
        for path in pending:
            frame = pd.concat(
                [r for t, r in zip(tasks, results) if t[0] == path], ignore_index=True
            )
            cache = cache_dir / (Path(path).stem + ".csv.gz")
            frame.to_csv(cache, index=False)
            cache.with_suffix(".json").write_text(signature)
            frames[path] = frame
            print(f"[features] {Path(path).name}: {len(frame)} events", flush=True)
    frame = pd.concat([frames[path] for path in paths], ignore_index=True)
    uid = [event_uid(*row) for row in zip(frame.source_file, frame.entry_index, frame.event_id)]
    if len(set(uid)) != len(uid):
        raise RuntimeError("Duplicate composite event identities found")
    frame.insert(0, "event_uid", uid)
    return frame


def discover_fixed_energy_files(input_dir):
    files = sorted(
        str(path) for path in Path(input_dir).glob("*_pi-_*.root")
        if re.search(r"_pi-_([0-9]+)GeV\.root$", path.name)
    )
    if not files:
        raise RuntimeError(f"No fixed-energy pion files found in {input_dir}")
    return files


def split_fixed_energy(frame, fractions, seed):
    """Deterministically split each source file by event into Train/Valid/Test."""
    train_fraction, valid_fraction = fractions
    if train_fraction <= 0 or valid_fraction <= 0 or train_fraction + valid_fraction >= 1:
        raise ValueError("Split fractions must be positive and leave a non-empty Test part")
    role = np.empty(len(frame), dtype=object)
    for source, index in frame.groupby("source_file").indices.items():
        rng = np.random.default_rng([seed, zlib.crc32(source.encode())])
        order = index[rng.permutation(index.size)]
        n_train = int(round(train_fraction * index.size))
        n_valid = int(round(valid_fraction * index.size))
        role[order[:n_train]] = "train"
        role[order[n_train:n_train + n_valid]] = "valid"
        role[order[n_train + n_valid:]] = "test"
    parts = [frame[role == name].reset_index(drop=True) for name in ("train", "valid", "test")]
    return tuple(parts)


def relative_mse(prediction, truth, denom_min):
    return float(np.mean(0.5 * ((prediction - truth) / np.maximum(truth, denom_min)) ** 2))


MAX_BINS = 255


def unweighted_bin_thresholds(column, max_bins=MAX_BINS):
    """Bin edges from unweighted Train quantiles (sklearn 1.9 unweighted rule).

    Recent sklearn computes HistGradientBoosting bin edges from weighted
    quantiles; with 1/E^2 loss weights almost all edges fall at low energy and
    high-energy events share one bin.  Pre-binning with these edges keeps the
    weights in the loss only.
    """
    column = column[np.isfinite(column)]
    distinct = np.unique(column)
    if distinct.size <= max_bins:
        return 0.5 * (distinct[:-1] + distinct[1:])
    percentiles = np.linspace(0, 100, num=max_bins + 1)[1:-1]
    return np.unique(np.percentile(column, percentiles, method="averaged_inverted_cdf"))


class PreBinnedRegressor:
    """Apply fixed unweighted bin edges, then a HistGradientBoostingRegressor."""

    def __init__(self, thresholds, model):
        self.thresholds = thresholds
        self.model = model

    def transform(self, x):
        return np.column_stack([
            np.searchsorted(edges, x[:, j], side="left")
            for j, edges in enumerate(self.thresholds)
        ]).astype(np.float64)

    def predict(self, x):
        return self.model.predict(self.transform(x))

    def staged_predict(self, x):
        return self.model.staged_predict(self.transform(x))


def fit_feature_set(train, valid, features, args):
    """Fit on Train, choose the iteration count on Valid, refit to that count."""
    weight = 1.0 / np.maximum(train.true_energy_GeV.to_numpy(), args.denom_min) ** 2
    params = dict(
        loss="squared_error", learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes, min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization, early_stopping=False,
        random_state=args.seed,
    )
    x_train, y_train = train[list(features)].to_numpy(), train.true_energy_GeV.to_numpy()
    x_valid, y_valid = valid[list(features)].to_numpy(), valid.true_energy_GeV.to_numpy()
    thresholds = [unweighted_bin_thresholds(x_train[:, j]) for j in range(x_train.shape[1])]
    binner = PreBinnedRegressor(thresholds, None)
    x_train_binned = binner.transform(x_train)
    model = PreBinnedRegressor(
        thresholds, HistGradientBoostingRegressor(max_iter=args.max_iter, **params)
    )
    model.model.fit(x_train_binned, y_train, sample_weight=weight)
    valid_loss = [relative_mse(p, y_valid, args.denom_min) for p in model.staged_predict(x_valid)]
    best_iter = int(np.argmin(valid_loss)) + 1
    model = PreBinnedRegressor(
        thresholds, HistGradientBoostingRegressor(max_iter=best_iter, **params)
    )
    model.model.fit(x_train_binned, y_train, sample_weight=weight)
    train_loss = relative_mse(model.predict(x_train), y_train, args.denom_min)
    return model, {
        "best_iteration": best_iter,
        "train_loss": train_loss,
        "valid_loss": valid_loss[best_iter - 1],
        "valid_loss_last_iteration": valid_loss[-1],
    }


def metric_rows(method, features, prediction, test, train_range):
    rows = []
    for energy in sorted(test.true_energy_GeV.unique()):
        selected = (test.true_energy_GeV == energy).to_numpy()
        values = prediction[selected]
        s_values = test.S_GeV.to_numpy()[selected]
        rows.append({
            "method": method,
            "uses_c": bool(C_FEATURES & set(features)),
            "true_energy_GeV": float(energy),
            **energy_metrics(values, float(energy)),
            "corr_pred_S_fixed_E": float(np.corrcoef(values, s_values)[0, 1]),
            "distance_to_train_edge_GeV": float(min(energy - train_range[0], train_range[1] - energy)),
        })
    return rows


def main():
    args = parse_args()
    for name, features in FEATURE_SETS.items():
        if name.startswith("s_") and C_FEATURES & set(features):
            raise RuntimeError(f"S-only feature set {name} contains C-derived inputs")
    profile = load_baseline_profile(args.input_dir, args.reference_dir)
    options = {
        "dense_threshold_gev": args.dense_threshold_gev,
        "late_layers": args.late_layers,
        "start_fraction": args.start_fraction,
    }
    cache_dir = args.output_dir / "features"
    if args.smoke_split_fixed_energy:
        fixed_files = discover_fixed_energy_files(args.input_dir)
        train_files = validation_files = test_files = fixed_files
        frame = load_features(fixed_files, profile.geometry, profile.calibration, options, args, cache_dir)
        train, valid, test = split_fixed_energy(frame, args.split_fractions, args.split_seed)
        data_role_policy = (
            "SMOKE TEST ONLY: each fixed-energy pion file split by event "
            f"(train/valid fractions {tuple(args.split_fractions)}, seed {args.split_seed}); "
            "discrete training energies, not a physics result"
        )
    else:
        train_files, validation_files, test_files = discover_file_roles(args.input_dir)
        train = load_features(train_files, profile.geometry, profile.calibration, options, args, cache_dir)
        valid = load_features(validation_files, profile.geometry, profile.calibration, options, args, cache_dir)
        test = load_features(test_files, profile.geometry, profile.calibration, options, args, cache_dir)
        data_role_policy = (
            "*_1-60GeV_Train.root=fit; *_1-60GeV_Valid.root=iteration selection; "
            "fixed-energy pion files=test only"
        )
    train_range = (float(train.true_energy_GeV.min()), float(train.true_energy_GeV.max()))

    rows = metric_rows("S_raw", ("S_GeV",), test.S_GeV.to_numpy(), test, train_range)
    summary = []
    for name in args.feature_sets:
        features = FEATURE_SETS[name]
        model, fit_info = fit_feature_set(train, valid, features, args)
        prediction = model.predict(test[list(features)].to_numpy())
        out_dir = args.output_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        test[["event_uid", *IDENTITY_COLUMNS, "S_GeV", "C_GeV"]].assign(
            pred_energy_GeV=prediction
        ).to_csv(out_dir / "test_predictions.csv", index=False)
        rows.extend(metric_rows(name, features, prediction, test, train_range))
        summary.append({"feature_set": name, "features": " ".join(features),
                        "uses_c": bool(C_FEATURES & set(features)), **fit_info})
        print(f"[bdt] {name}: best_iter={fit_info['best_iteration']} "
              f"valid_loss={fit_info['valid_loss']:.6f}", flush=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output_dir / "bdt_test_metrics.csv", index=False)
    pd.DataFrame(summary).to_csv(args.output_dir / "bdt_training_summary.csv", index=False)
    wide = metrics.pivot(index="true_energy_GeV", columns="method",
                         values="resolution_sigma68_over_mpv")
    wide[["S_raw", *args.feature_sets]].mul(100.0).add_suffix("_res_percent").to_csv(
        args.output_dir / "bdt_resolution_summary.csv"
    )
    config = {
        "baseline": profile.name,
        "geometry": asdict(profile.geometry),
        "cell_id_decoding": CELL_ID_DECODING,
        "calibration": asdict(profile.calibration),
        "calibration_table": profile.calibration_table,
        "hovere_fit_table": profile.hovere_fit_table,
        "data_manifest": {"train": train_files, "validation": validation_files, "test": test_files},
        "train_energy_range_GeV": train_range,
        "n_events": {"train": len(train), "validation": len(valid), "test": len(test)},
        "feature_sets": {name: FEATURE_SETS[name] for name in args.feature_sets},
        "feature_options": options,
        "bdt": {
            "model": "sklearn HistGradientBoostingRegressor",
            "binning": f"pre-binned with unweighted Train quantiles, max_bins={MAX_BINS}; "
                       "1/E^2 weights enter the loss only",
            "max_iter": args.max_iter, "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes, "min_samples_leaf": args.min_samples_leaf,
            "l2_regularization": args.l2_regularization, "seed": args.seed,
        },
        "loss": f"0.5*((E_pred-E_true)/max(E_true,{args.denom_min} GeV))^2 via sample weights",
        "iteration_selection": "minimum relative-MSE on Valid; refit on Train to that count",
        "data_role_policy": data_role_policy,
        "truth_policy": "MCtruth_energy is target only; no truth or filename energy enters features",
        "s_only_policy": "feature sets named s_* contain no C-derived input",
        "resolution_definition": "sigma68/MPV (calo.dr_metrics.energy_metrics)",
        "prior_edge_note": (
            "Regression predictions are compressed toward the Train spectrum interior; "
            "test points near train_energy_range_GeV edges are not comparable with "
            "calibration-based methods"
        ),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2))
    print(wide.mul(100.0).round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
