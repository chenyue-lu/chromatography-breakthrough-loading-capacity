"""Focused unit and generated-output checks; run the complete study before discovery."""

from dataclasses import replace
import math
from pathlib import Path
import random
import unittest

random.seed(20261025)
import numpy as np

np.random.seed(20261025)
import pandas as pd

import chromatography_study as study


def example_result(ratios=(0.0, 0.2, 0.95)):
    """Small synthetic fixture; these values are never used as study outputs."""
    ratios = np.asarray(ratios, dtype=float)
    count = len(ratios)
    return study.SimulationResult(
        flow_ml_min=1.0, flow_m3_s=1e-6 / 60, samples_per_residence_time=200,
        axial_cells=2, rtol=1e-7, atol=1e-9, load_extensions=0,
        time_s=np.arange(count, dtype=float), inlet_mol_m3=np.full(count, 0.02),
        outlet_mol_m3=ratios * 0.02, bulk_liquid_mol_m3=np.zeros((count, 2)),
        pore_liquid_mol_m3=np.zeros((count, 2)), adsorbed_mol_m3_solid=np.zeros((count, 2)),
        axial_positions_m=np.array([0.025, 0.075]), cell_bed_volume_m3=study.BED_VOLUME_M3 / 2,
    )


class InterpolationTests(unittest.TestCase):
    def test_first_upward_crossing_uses_bracketing_interpolation(self):
        crossing = study.first_crossing([0, 2, 4, 6], [0, 0.2, 0.05, 0.3])
        self.assertEqual((crossing.left, crossing.right), (0, 1))
        self.assertEqual(crossing.time_s, 1.0)
        self.assertEqual(crossing.fraction, 0.5)
        self.assertNotIn(crossing.time_s, [0, 2, 4, 6])

    def test_irregular_spacing_and_exact_right_endpoint(self):
        crossing = study.first_crossing([0, 7, 12], [0, 0.06, 0.20])
        self.assertAlmostEqual(crossing.time_s, 7 + 5 * 0.04 / 0.14)
        exact = study.first_crossing([0, 7, 12], [0, 0.1, 0.3])
        self.assertEqual(exact.time_s, 7)
        self.assertEqual(exact.fraction, 1)

    def test_failure_without_crossing_or_lower_bracket(self):
        for ratios in ([0, 0.02, 0.09], [0.1, 0.2, 0.5], [0.2, 0.3, 0.9]):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                study.first_crossing([0, 1, 2], ratios)

    def test_invalid_series_rejected(self):
        for time, ratios in (([0, 0], [0, 1]), ([0, 1], [0, float("nan")]), ([0], [0]), ([0, 1], [0])):
            with self.subTest(time=time, ratios=ratios), self.assertRaises(ValueError):
                study.first_crossing(time, ratios)

    def test_partial_interval_mass_integral_is_not_interpolated_cumulative_mass(self):
        crossing = study.first_crossing([0, 2], [0, 0.2])
        mass = study.cumulative_mass_at_crossing(np.array([0, 2]), np.array([0, 2]), np.array([0, 2]), crossing)
        self.assertAlmostEqual(mass, 0.5)  # integral of rate=t from 0 to 1


class VolumeAndMassTests(unittest.TestCase):
    def test_geometry_hold_up_residence_loaded_volume_and_dbc(self):
        metrics = study.loading_metrics(1.0, 600.0)
        bed_ml = math.pi * 0.01**2 * 0.1 / 4 * 1e6
        hold_up_ml = bed_ml * (0.37 + 0.63 * 0.33)
        self.assertAlmostEqual(metrics["bed_volume_ml"], bed_ml)
        self.assertAlmostEqual(metrics["hold_up_volume_ml"], hold_up_ml)
        self.assertAlmostEqual(metrics["nominal_residence_time_s"], bed_ml * 60)
        self.assertAlmostEqual(metrics["loaded_feed_volume_ml"], 10)
        self.assertAlmostEqual(metrics["feed_concentration_mg_ml"], 1)
        self.assertAlmostEqual(metrics["dbc10_mg_ml_bed"], (10 - hold_up_ml) / bed_ml)
        self.assertAlmostEqual(study.HOLD_UP_VOLUME_M3 * 1e6, hold_up_ml)
        self.assertLess(hold_up_ml, bed_ml)

    def test_invalid_flow_or_time_rejected(self):
        for flow, time in ((0, 10), (1, 0), (-1, 10), (1, float("nan"))):
            with self.subTest(flow=flow, time=time), self.assertRaises(ValueError):
                study.loading_metrics(flow, time)

    def test_distinct_liquid_and_solid_volume_weights(self):
        result = example_result()
        result = replace(result, bulk_liquid_mol_m3=np.ones((3, 2)),
                         pore_liquid_mol_m3=np.full((3, 2), 2.0),
                         adsorbed_mol_m3_solid=np.full((3, 2), 3.0))
        masses = study.mass_trajectories(result)
        factor = study.BED_VOLUME_M3 * 50 * 1e6
        self.assertAlmostEqual(masses["liquid_mass_mg"][-1], factor * (0.37 + 0.63 * 0.33 * 2))
        self.assertAlmostEqual(masses["adsorbed_mass_mg"][-1], factor * 0.63 * 0.67 * 3)
        self.assertAlmostEqual(masses["cumulative_inlet_mass_mg"][-1], (1e-6 / 60) * 50e6 * 0.02 * 2)

    def test_normalization_and_acceptance_use_absolute_residual(self):
        self.assertEqual(study.normalized_mass_residual(100, 40, 10, 50), 0)
        self.assertAlmostEqual(study.normalized_mass_residual(100, 40, 10, 51), -0.01)
        study.check_mass_acceptance(-0.01, 0.01)
        for residuals in ((0.010001, 0), (0, -0.010001), (float("nan"), 0), (0, float("inf"))):
            with self.subTest(residuals=residuals), self.assertRaises(ValueError):
                study.check_mass_acceptance(*residuals)
        with self.assertRaises(ValueError):
            study.normalized_mass_residual(0, 0, 0, 0)


class RefinementTests(unittest.TestCase):
    def metrics(self):
        fine = np.array([[100, 1], [200, 2], [300, 3]], dtype=float)
        return np.stack([fine * 1.005, fine * 1.001, fine])

    def test_coarsest_level_within_one_percent_is_selected(self):
        selected, converged, disclosure = study.select_numerical_level(self.metrics())
        self.assertEqual(selected, 0)
        self.assertTrue(converged)
        self.assertEqual(disclosure, "none")

    def test_middle_level_selected_when_coarse_misses_either_metric(self):
        metrics = self.metrics()
        metrics[0, 2, 1] = metrics[2, 2, 1] * 1.02
        self.assertEqual(study.select_numerical_level(metrics)[:2], (1, True))

    def test_finest_gate_checks_both_metrics_at_every_flow(self):
        for flow_index in range(3):
            for metric_index in range(2):
                metrics = self.metrics()
                metrics[0] = metrics[2]  # coarse closeness cannot bypass the finest-pair gate
                metrics[1, flow_index, metric_index] *= 1.02
                with self.subTest(flow=flow_index, metric=metric_index):
                    selected, converged, disclosure = study.select_numerical_level(metrics)
                    self.assertEqual(selected, 2)
                    self.assertFalse(converged)
                    self.assertIn("Unresolved numerical sensitivity", disclosure)
                    self.assertIn("finest prescribed level", disclosure)

    def test_one_percent_boundary_is_inclusive(self):
        metrics = np.full((3, 3, 2), 100.0)
        metrics[:2] = 101
        self.assertEqual(study.select_numerical_level(metrics)[:2], (0, True))

    def test_missing_disclosure_and_invalid_metrics_rejected(self):
        metrics = self.metrics()
        metrics[1] *= 1.1
        decision = study.select_numerical_level(metrics)
        settings = tuple(study.NumericalSettings(samples_per_residence_time=n) for n in study.OUTPUT_SAMPLING_LEVELS)
        report = study.RefinementResult("sampling", settings, metrics, *decision)
        study.validate_refinement(report)
        with self.assertRaises(ValueError):
            study.validate_refinement(replace(report, disclosure="none"))
        with self.assertRaises(ValueError):
            study.validate_refinement(replace(report, selected_index=0))
        for invalid in (np.zeros((3, 3, 2)), np.ones((3, 2, 2)), np.full((3, 3, 2), np.nan)):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                study.select_numerical_level(invalid)


class RawCurveTests(unittest.TestCase):
    def test_valid_curve_and_defined_numerical_tolerance(self):
        study.validate_raw_curve(example_result())
        study.validate_raw_curve(example_result([0, 0.5, 0.5 - 0.5e-6, 0.95]))
        self.assertAlmostEqual(study.raw_ratio_tolerance(1e-8), 2e-6)
        loose = replace(example_result([0, 0.5, 0.5 - 1.5e-6, 0.95]), atol=1e-8)
        study.validate_raw_curve(loose)
        with self.assertRaises(ValueError):
            study.validate_raw_curve(replace(loose, atol=1e-9))

    def test_nonfinite_nonmonotonic_incomplete_and_no_crossing_rejected(self):
        for ratios in ([0, np.nan, 0.95], [0, 0.3, 0.2, 0.95], [0, 0.2, 0.94], [0, 0.01, 0.09]):
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                study.validate_raw_curve(example_result(ratios))

    def test_nonfinite_internal_state_rejected(self):
        result = example_result()
        state = result.pore_liquid_mol_m3.copy()
        state[1, 1] = np.inf
        with self.assertRaises(ValueError):
            study.validate_raw_curve(replace(result, pore_liquid_mol_m3=state))


class GeneratedOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(__file__).resolve().parent / "results"
        cls.parameters = pd.read_csv(cls.directory / "parameters.csv")
        cls.summary = pd.read_csv(cls.directory / "summary.csv")
        cls.curves = pd.read_csv(cls.directory / "breakthrough_curves.csv")

    def test_exact_output_set_required_columns_and_flow_coverage(self):
        self.assertEqual({path.name for path in self.directory.iterdir()}, set(study.RESULT_FILENAMES))
        self.assertTrue(set(study.SUMMARY_COLUMNS) | set(study.ACCEPTANCE_COLUMNS) <= set(self.summary))
        self.assertTrue(set(study.CURVE_COLUMNS) <= set(self.curves))
        self.assertEqual(set(self.summary.flow_ml_min), {0.5, 1, 2})
        self.assertEqual(set(self.curves.flow_ml_min), {0.5, 1, 2})
        study.validate_output_tables(self.parameters, self.summary, self.curves)

    def test_all_raw_curves_meet_completion_and_monotonicity(self):
        for flow, group in self.curves.groupby("flow_ml_min"):
            with self.subTest(flow=flow):
                ratios = group.outlet_inlet_ratio.to_numpy()
                self.assertTrue(np.isfinite(group.select_dtypes(include="number")).all().all())
                tolerance = self.summary.set_index("flow_ml_min").loc[flow, "raw_ratio_numerical_tolerance"]
                self.assertLessEqual(float(np.max(np.maximum.accumulate(ratios) - ratios)), tolerance)
                self.assertGreaterEqual(ratios[-1], 0.95)
                crossing = study.first_crossing(group.time_s.to_numpy(), ratios)
                self.assertLess(ratios[crossing.left], 0.1)
                self.assertGreaterEqual(ratios[crossing.right], 0.1)

    def test_mass_balance_and_convergence_or_finest_disclosure(self):
        for _, row in self.summary.iterrows():
            study.check_mass_acceptance(row.mass_balance_residual_t10_fraction, row.mass_balance_residual_end_fraction)
        for factor in study.REFINEMENT_FACTORS:
            metrics = np.stack([self.summary[[f"{factor}_{level}_t10_s", f"{factor}_{level}_dbc10_mg_ml_bed"]].to_numpy()
                                for level in ("coarse", "middle", "fine")])
            selected, converged, disclosure = study.select_numerical_level(metrics)
            self.assertTrue(self.summary[f"{factor}_selected_level_index"].eq(selected).all())
            self.assertTrue(self.summary[f"{factor}_converged"].eq(converged).all())
            self.assertTrue(self.summary[f"{factor}_disclosure"].eq(disclosure).all())
            if not converged:
                self.assertEqual(selected, 2)
                self.assertIn("Unresolved", disclosure)

    def test_manual_metric_override_is_rejected(self):
        summary = self.summary.copy()
        summary.loc[0, "t10_s"] += 1
        with self.assertRaises(ValueError):
            study.validate_output_tables(self.parameters, summary, self.curves)

    def test_missing_columns_and_flow_are_rejected(self):
        with self.assertRaises(ValueError):
            study.validate_output_tables(self.parameters, self.summary.drop(columns="hold_up_volume_ml"), self.curves)
        with self.assertRaises(ValueError):
            study.validate_output_tables(self.parameters, self.summary.iloc[:2], self.curves)

    def test_inconsistent_settings_and_false_convergence_are_rejected(self):
        summary = self.summary.copy()
        summary.loc[0, "selected_axial_cells"] *= 2
        with self.assertRaises(ValueError):
            study.validate_output_tables(self.parameters, summary, self.curves)
        summary = self.summary.copy()
        summary["sampling_converged"] = ~summary["sampling_converged"]
        with self.assertRaises(ValueError):
            study.validate_output_tables(self.parameters, summary, self.curves)

    def test_failing_mass_acceptance_flag_is_rejected(self):
        summary = self.summary.copy()
        summary.loc[0, "mass_balance_passed"] = False
        with self.assertRaises(ValueError):
            study.validate_output_tables(self.parameters, summary, self.curves)


if __name__ == "__main__":
    unittest.main()
