import importlib.util
import unittest
from pathlib import Path

import numpy as np

from calo.dual_readout import DualReadoutCalibration, DualReadoutGeometry


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_bdt_baseline.py"
spec = importlib.util.spec_from_file_location("run_bdt_baseline", SCRIPT)
bdt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bdt)


def encoded_cell(ix, iy, iz, n_fiber=60):
    # Fiber segment cellID; the checkerboard fixes the type: (ix+iy) even = S.
    return (iz * n_fiber + iy) * n_fiber + ix, (ix + iy) % 2


class BDTFeatureTest(unittest.TestCase):
    def setUp(self):
        # 60 fibers per side, 120 z segments; the full grid is used.
        self.geo = DualReadoutGeometry(nx=60, ny=60, nz=120)
        self.calibration = DualReadoutCalibration(
            s_gev_per_count=0.001, c_gev_per_count=0.002,
            h_s_intercept=0.7, h_s_slope_per_gev=0.0,
            h_c_intercept=0.35, h_c_slope_per_gev=0.0,
        )

    def features(self, hits, n_events=1):
        event, cell_and_type, n_s, n_c = zip(*hits)
        cell, fiber_type = zip(*cell_and_type)
        event, cell, fiber_type, n_s, n_c = (
            np.asarray(v) for v in (event, cell, fiber_type, n_s, n_c)
        )
        return bdt.event_features(
            event, cell, fiber_type, n_s, n_c, n_events, self.geo, self.calibration,
            dense_threshold_gev=0.015, late_layers=20, start_fraction=0.05,
        )

    def test_repeated_entries_are_summed_per_cell_before_counting(self):
        f = self.features([
            (0, encoded_cell(20, 20, 3), 10, 0),
            (0, encoded_cell(20, 20, 3), 10, 0),
        ])
        self.assertAlmostEqual(f["S_GeV"][0], 0.020)
        self.assertEqual(f["n_active_S"][0], 1)
        # Each entry is below the dense threshold, the summed cell is above it.
        self.assertEqual(f["n_dense_S"][0], 1)
        self.assertAlmostEqual(f["dense_fraction_S"][0], 1.0)

    def test_full_grid_keeps_edge_cells(self):
        f = self.features([
            (0, encoded_cell(20, 20, 3), 10, 0),
            (0, encoded_cell(0, 0, 3), 500, 0),
            (0, encoded_cell(20, 20, 119), 500, 0),
            (0, encoded_cell(59, 0, 3), 0, 500),
        ])
        self.assertAlmostEqual(f["S_GeV"][0], 1.010)
        self.assertEqual(f["n_active_S"][0], 3)
        self.assertAlmostEqual(f["C_GeV"][0], 1.0)
        self.assertEqual(f["n_active_C"][0], 1)

    def test_channels_and_longitudinal_summaries(self):
        f = self.features([
            (0, encoded_cell(20, 20, 2), 10, 0),
            (0, encoded_cell(22, 20, 110), 30, 0),
            # C fiber: its vecNScint must not enter S, its vecNChren is C.
            (0, encoded_cell(21, 20, 2), 99, 7),
            # S fiber: its vecNChren must not enter C.
            (0, encoded_cell(24, 20, 2), 0, 50),
            (1, encoded_cell(30, 30, 0), 5, 0),
        ], n_events=2)
        np.testing.assert_allclose(f["S_GeV"], [0.040, 0.005])
        np.testing.assert_allclose(f["C_GeV"], [0.014, 0.0])
        np.testing.assert_allclose(f["n_active_C"], [1, 0])
        self.assertAlmostEqual(f["z_cog_layer"][0], (2 * 10 + 110 * 30) / 40)
        self.assertEqual(f["z_start_layer"][0], 2)
        self.assertAlmostEqual(f["late_fraction_S"][0], 0.75)
        self.assertGreater(f["r_rms_cells"][0], 0.0)
        self.assertEqual(f["r_rms_cells"][1], 0.0)

    def test_fiber_cell_id_round_trip(self):
        cells = [(0, 0, 0), (59, 0, 0), (0, 59, 0), (17, 42, 119), (59, 59, 119)]
        ids, types = zip(*(encoded_cell(*c) for c in cells))
        ix, iy, iz, valid = bdt.decode_fiber_cell_ids(np.array(ids), np.array(types), self.geo)
        self.assertTrue(np.all(valid))
        np.testing.assert_array_equal(np.stack([ix, iy, iz], axis=1), np.array(cells))

    def test_fiber_type_parity_mismatch_is_rejected(self):
        cell, fiber_type = encoded_cell(20, 20, 3)
        with self.assertRaises(RuntimeError):
            bdt.event_features(
                np.array([0]), np.array([cell]), np.array([1 - fiber_type]),
                np.array([10]), np.array([10]), 1, self.geo, self.calibration,
            )

    def test_smoke_split_is_disjoint_complete_and_deterministic(self):
        import pandas as pd
        frame = pd.DataFrame({
            "source_file": ["a.root"] * 100 + ["b.root"] * 50,
            "entry_index": list(range(100)) + list(range(50)),
        })
        parts = bdt.split_fixed_energy(frame, (0.6, 0.2), seed=7)
        keys = [set(zip(p.source_file, p.entry_index)) for p in parts]
        self.assertEqual([len(k) for k in keys], [90, 30, 30])
        self.assertFalse(keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
        self.assertEqual(set().union(*keys), set(zip(frame.source_file, frame.entry_index)))
        again = bdt.split_fixed_energy(frame, (0.6, 0.2), seed=7)
        self.assertEqual(keys, [set(zip(p.source_file, p.entry_index)) for p in again])

    def test_prebinning_keeps_high_energies_separable_under_1_over_e2_weights(self):
        from sklearn.ensemble import HistGradientBoostingRegressor
        rng = np.random.default_rng(3)
        energy = np.repeat([1.0, 5.0, 10.0, 20.0, 30.0, 50.0], 600)
        s = 0.86 * energy * (1 + 0.2 / np.sqrt(energy) * rng.standard_normal(energy.size))
        x = s[:, None]
        model = bdt.PreBinnedRegressor(
            [bdt.unweighted_bin_thresholds(s)],
            HistGradientBoostingRegressor(max_iter=200, early_stopping=False, random_state=0),
        )
        model.model.fit(model.transform(x), energy, sample_weight=1 / energy**2)
        prediction = model.predict(x)
        for e in (20.0, 30.0, 50.0):
            self.assertLess(abs(prediction[energy == e].mean() / e - 1), 0.05, e)

    def test_s_only_feature_sets_exclude_cherenkov_inputs(self):
        for name, features in bdt.FEATURE_SETS.items():
            self.assertTrue(set(features) <= set(bdt.FEATURE_COLUMNS))
            if name.startswith("s_"):
                self.assertFalse(bdt.C_FEATURES & set(features), name)


if __name__ == "__main__":
    unittest.main()
