"""Seed pairing and data-integrity tests for SB-CFO plotting."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from util.sb_cfo_figure_data import FINAL, SEEDS, interval, load_study, load_studies

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/sb_cfo_cosine_lr0p001"


class ColourTests(unittest.TestCase):
    def test_distinct_series_and_redundant_encodings(self):
        from plot_sb_cfo_figures import SERIES_COLORS, trajectory_style
        self.assertEqual(len(set(SERIES_COLORS.values())), 4)
        for method, condition in SERIES_COLORS:
            style = trajectory_style(method, condition)
            self.assertEqual(style['color'], SERIES_COLORS[method, condition])
            self.assertEqual(style['linestyle'], '--' if condition == 'iid' else '-')
            self.assertEqual(style['marker'], 'o' if method == 'fedavg' else 's')

    def test_categorical_colours_contrast_with_white(self):
        from matplotlib.colors import to_rgb
        from util.paper_figure_style import ALPHA_COLORS, IID_COLOR, SETTING_COLORS
        colours = [*ALPHA_COLORS.values(), IID_COLOR]
        self.assertEqual(len(set(colours)), 5)
        self.assertEqual(set(SETTING_COLORS), set(ALPHA_COLORS.values()))
        for colour in colours:
            rgb = np.array(to_rgb(colour))
            linear = np.where(rgb <= 0.04045, rgb/12.92, ((rgb+0.055)/1.055)**2.4)
            luminance = float(linear @ np.array([0.2126, 0.7152, 0.0722]))
            self.assertGreaterEqual(1.05/(luminance+0.05), 3.0, colour)


class IntervalTests(unittest.TestCase):
    def test_student_t(self):
        mean, low, high = interval([1, 2, 3])
        self.assertAlmostEqual(mean, 2)
        self.assertAlmostEqual(high-mean, 2.4841377117)
        self.assertAlmostEqual(mean-low, high-mean)

    def test_geometric(self):
        mean, low, high = interval([0.01, 0.1, 1], geometric=True)
        self.assertAlmostEqual(mean, 0.1)
        self.assertGreater(low, 0)
        self.assertAlmostEqual(mean/low, high/mean)

    def test_invalid(self):
        for values in ([], [1], [1, float("nan")], [float("inf"), 1]):
            with self.assertRaises(ValueError):
                interval(values)
        for values in ([0, 1], [-1, 1]):
            with self.assertRaises(ValueError):
                interval(values, geometric=True)


@unittest.skipUnless(all((RESULTS / folder / "metadata.json").exists()
                        for folder in ("darcy/adam", "burgers/nu0p005/adam")),
                     "Local completed studies unavailable")
class CompletedStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.studies = load_studies(RESULTS)

    def test_complete_and_paired(self):
        for study in self.studies:
            self.assertEqual(len(study.rows), 200)
            for method in ("fedavg", "sb_cfo"):
                manual = [100*(float(study.index[(method, "alpha0.01", s, FINAL)]["test_error"])
                               - float(study.index[(method, "iid", s, FINAL)]["test_error"])) for s in SEEDS]
                np.testing.assert_allclose(study.gap(method), manual)

    def test_final_effects_match_saved_summary(self):
        # Local studies can be rerun; compare independent artifacts instead of
        # hard-coding effects from a previous batch-size configuration.
        for study in self.studies:
            with (study.directory / 'summary.csv').open(newline='', encoding='utf-8') as stream:
                rows = {r['method']: r for r in csv.DictReader(stream)
                        if r['condition'] == 'alpha0.01'}
            for method in ('fedavg', 'sb_cfo'):
                mean, low, high = interval(study.gap(method))
                for statistic, value in (('mean', mean), ('ci95_low', low), ('ci95_high', high)):
                    self.assertAlmostEqual(value, float(rows[method]['excess_vs_own_iid_pp_' + statistic]))

    def test_gradient_and_cost(self):
        for study in self.studies:
            self.assertTrue(np.all((study.gradient_remaining() > 0) & (study.gradient_remaining() < 1)))
            self.assertEqual(study.cost_ratios().shape, (3, 5))
            self.assertTrue(np.all(study.cost_ratios() > 1))

    def test_reject_duplicate_missing_incomplete(self):
        original = self.studies[0]
        for mode in ("duplicate", "missing", "incomplete"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                rows = copy.deepcopy(original.rows)
                meta = copy.deepcopy(original.metadata)
                if mode == "duplicate":
                    rows.append(rows[0])
                elif mode == "missing":
                    rows.pop()
                else:
                    meta["complete"] = False
                with (folder / "rounds.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
                (folder / "summary.csv").write_bytes((original.directory / "summary.csv").read_bytes())
                with self.assertRaises(ValueError):
                    load_study(folder, "burgers")


if __name__ == "__main__":
    unittest.main()
