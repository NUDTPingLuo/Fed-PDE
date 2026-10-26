"""Regression tests for read-only manuscript data and paired statistics."""
import copy
import unittest
from pathlib import Path

import numpy as np

from util.paper_figure_data import ALPHAS, FINAL_ROUND, Experiment, bin_counts, mean_ci95, unique_index

ROOT = Path(__file__).resolve().parents[1]
FORMAL_FILE = ROOT / "results/formal/burgers/burgers_formal_k10_s5_r1000_alpha_round_seed_raw.csv"
PUBLIC_FILE = ROOT / "results/public/darcy/darcy_beta0p01_k10_s5_r1000_alpha_round_seed_raw.csv"


class StatisticsTests(unittest.TestCase):
    def test_student_t_interval(self):
        mean, half = mean_ci95([1, 2, 3])
        self.assertEqual(mean, 2)
        self.assertAlmostEqual(half, 2.4841377117, places=8)

    def test_nonfinite_observations_are_not_silently_dropped(self):
        for values in ([1, float("nan")], [float("inf")], []):
            with self.assertRaises(ValueError):
                mean_ci95(values)

    def test_duplicate_keys_rejected(self):
        with self.assertRaises(ValueError):
            unique_index([{"seed": "0"}, {"seed": "0"}], ("seed",))

    def test_bin_matrix_round_trip(self):
        result = bin_counts("1;2;0|0;1;3")
        np.testing.assert_array_equal(result, [[1, 2, 0], [0, 1, 3]])
        with self.assertRaises(ValueError):
            bin_counts("1;2|3")


@unittest.skipUnless(FORMAL_FILE.exists() and PUBLIC_FILE.exists(), "Completed real-result fixtures are optional")
class CompletedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.formal = Experiment.load(FORMAL_FILE, "formal")
        cls.public = Experiment.load(PUBLIC_FILE, "public")

    def test_both_metadata_layouts(self):
        self.assertEqual(self.formal.config["n_clients"], 10)
        self.assertEqual(self.public.config["federation"]["n_clients"], 10)
        self.assertEqual(len(self.formal.records), 200)
        self.assertEqual(len(self.public.references), 50)

    def test_same_seed_subtraction_precedes_ci(self):
        e = self.formal
        alpha = 0.01
        differences = e.values("federated_error", alpha) - e.values("federated_error", None)
        np.testing.assert_allclose(differences, e.values("drift", alpha), atol=1e-12)
        mean, half = e.stats("drift", alpha)
        expected_mean, expected_half = mean_ci95(differences)
        self.assertAlmostEqual(mean, expected_mean, places=12)
        self.assertAlmostEqual(half, expected_half, places=12)

    def test_late_curve_ci_is_paired_at_every_checkpoint(self):
        for e in (self.formal, self.public):
            for alpha in ALPHAS:
                rounds, means, halves = e.curve("drift", alpha, start=200)
                np.testing.assert_array_equal(rounds, [200, 500, 1000])
                for r, mean, half in zip(rounds, means, halves):
                    paired = (e.values("federated_error", alpha, int(r))
                              - e.values("federated_error", None, int(r)))
                    expected_mean, expected_half = mean_ci95(paired)
                    self.assertAlmostEqual(mean, expected_mean, places=12)
                    self.assertAlmostEqual(half, expected_half, places=12)

    def test_round_zero_preserved_but_not_on_log_axis(self):
        rounds, _, _ = self.formal.curve("drift", 0.01)
        self.assertIn(0, self.formal.rounds)
        self.assertNotIn(0, rounds)
        self.assertIn(FINAL_ROUND, rounds)

    def test_no_pseudoreplication_of_iid(self):
        mean, half = self.public.stats("federated_error", None)
        expected = mean_ci95(self.public.values("federated_error", None))
        self.assertEqual((mean, half), expected)
        self.assertEqual(len(self.public.values("federated_error", None)), 5)

    def test_missing_or_incorrect_records_fail(self):
        e = copy.deepcopy(self.formal)
        e.records.pop((ALPHAS[0], e.seeds[0], FINAL_ROUND))
        with self.assertRaises(ValueError):
            e.validate()
        e = copy.deepcopy(self.formal)
        e.records[(0.01, e.seeds[0], FINAL_ROUND)]["drift"] = "9"
        with self.assertRaises(ValueError):
            e.validate()


@unittest.skipUnless(FORMAL_FILE.exists(), "Completed real-result fixture is optional")
class FigureLayoutTests(unittest.TestCase):
    def setUp(self):
        import plot_paper_figures as plotting
        from util.paper_figure_style import configure_matplotlib
        configure_matplotlib()
        self.plotting = plotting

    def tearDown(self):
        self.plotting.plt.close("all")

    def test_error_curve_has_full_left_and_paired_right_without_footer(self):
        class Capture:
            def save(self, figure, *args, **kwargs):
                self.figure = figure

        e = Experiment.load(FORMAL_FILE, "formal")
        capture = Capture()
        self.plotting.plot_error_curves([e], capture, "test")
        left, right = capture.figure.axes
        self.assertEqual(left.get_xscale(), "log")
        self.assertEqual(left.get_yscale(), "linear")
        np.testing.assert_array_equal(left.get_xticks(), [1, 10, 100, 1000])
        self.assertEqual(left.get_ylabel(), r"Relative $L_2$ error")
        self.assertEqual(left.get_xlabel(), "Communication round (log scale)")
        self.assertEqual(len(left.lines), 5)
        self.assertEqual(right.get_xscale(), "linear")
        self.assertEqual(right.get_ylabel(), "Excess error (pp)")
        self.assertEqual(len(right.lines), 5)  # Four alpha curves and a zero line.
        for line, alpha in zip(right.lines[:4], ALPHAS):
            rounds, mean, _ = e.curve("drift", alpha, start=200)
            np.testing.assert_array_equal(line.get_xdata(), rounds)
            np.testing.assert_allclose(line.get_ydata(), 100 * mean)
        np.testing.assert_array_equal(right.lines[-1].get_ydata(), [0, 0])
        self.assertEqual(len(right.collections), 4)  # CI retained for every alpha.
        self.assertEqual(capture.figure.texts, [])  # No embedded caption/footer.

    def test_linear_round_axis_has_equal_physical_spacing(self):
        fig, axis = self.plotting.plt.subplots()
        axis.set_xscale("log")
        self.plotting.style_round_axis(axis, logarithmic=False)
        points = axis.transData.transform([[0, 0], [200, 0], [400, 0]])
        self.assertAlmostEqual(points[1, 0]-points[0, 0], points[2, 0]-points[1, 0])

    def test_log_round_axis_has_equal_decade_spacing(self):
        fig, axis = self.plotting.plt.subplots()
        self.plotting.style_round_axis(axis, logarithmic=True)
        points = axis.transData.transform([[1, 0], [10, 0], [100, 0], [1000, 0]])
        np.testing.assert_allclose(np.diff(points[:, 0]), np.diff(points[:, 0])[0])
        self.plotting.style_round_axis(axis, late=True)
        self.assertEqual(axis.get_xscale(), "linear")

    def test_multitask_curves_share_bottom_axis_titles(self):
        class Capture:
            def save(self, figure, *args, **kwargs):
                self.figure = figure

        e = Experiment.load(FORMAL_FILE, "formal")
        capture = Capture()
        self.plotting.plot_error_curves([e, e], capture, "test")
        axes = capture.figure.axes
        self.assertEqual([axis.get_xlabel() for axis in axes[:2]], ["", ""])
        self.assertEqual(axes[2].get_xlabel(), "Communication round (log scale)")
        self.assertEqual(axes[3].get_xlabel(), "Communication round")

    def test_physical_regime_legends_belong_to_their_own_axes(self):
        from util.paper_figure_data import viscosity_experiments

        class Capture:
            def save(self, figure, *args, **kwargs):
                self.figure = figure

        capture = Capture()
        self.plotting.plot_physical_regime(viscosity_experiments(ROOT / "results"), capture)
        left = next(ax for ax in capture.figure.axes if ax.get_gid() == 'b')
        right = next(ax for ax in capture.figure.axes if ax.get_gid() == 'c')
        fields = [ax.images[0] for ax in capture.figure.axes if str(ax.get_gid()).startswith('a-')]
        self.assertEqual(len(fields), 4)
        self.assertTrue(all(image.norm is fields[0].norm for image in fields))
        for image in fields[1:]:
            np.testing.assert_array_equal(image.get_array()[:, 0], fields[0].get_array()[:, 0])
        self.assertEqual(capture.figure.legends, [])
        self.assertEqual(len(left.get_legend().get_texts()), 4)
        self.assertTrue(all("ν =" in t.get_text() for t in left.get_legend().get_texts()))
        self.assertEqual([t.get_text() for t in right.get_legend().get_texts()], ["IID", "α = 0.01"])

    def test_panel_letters_clear_long_vertical_labels(self):
        from util.paper_figure_style import grid, panel, require_panel_label_clearance
        fig, axes = grid(3, 3, height_mm=155)
        for index, axis in enumerate(axes.flat):
            panel(axis, index, "Task" if index % 3 == 0 else "")
            axis.set_ylabel("Normalized initial gradient heterogeneity")
        require_panel_label_clearance(fig, list(axes.flat))

    def test_mixed_width_layout_uses_explicit_comparable_groups(self):
        import tempfile
        from util.paper_figure_style import require_basic_alignment
        fig = self.plotting.plt.figure()
        small = fig.add_axes([0.1, 0.65, 0.2, 0.2]); small.set_gid('small')
        wide = fig.add_axes([0.1, 0.1, 0.5, 0.3]); wide.set_gid('wide')
        peer = fig.add_axes([0.4, 0.65, 0.2, 0.2]); peer.set_gid('peer')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'layout.json'
            require_basic_alignment(fig, [small, wide, peer], path,
                                    alignment_options={'row_groups': [['small', 'peer']]})
            with self.assertRaisesRegex(ValueError, 'Unequal column widths'):
                require_basic_alignment(fig, [small, wide, peer], path,
                                        alignment_options={'column_groups': [['small', 'wide']]})

    def test_clearance_gate_rejects_label_overlap(self):
        from util.paper_figure_style import grid, require_panel_label_clearance
        fig, axes = grid(1, 1)
        axis = axes[0, 0]
        axis.set_ylabel("Gradient heterogeneity")
        fig.canvas.draw()
        box = axis.yaxis.label.get_window_extent(fig.canvas.get_renderer())
        width, height = fig.canvas.get_width_height()
        axis.text((box.x0 + box.x1) / (2 * width), (box.y0 + box.y1) / (2 * height),
                  "a", transform=fig.transFigure, ha="center", va="center",
                  fontweight="bold", gid="panel-label")
        with self.assertRaisesRegex(ValueError, "overlaps vertical-axis label"):
            require_panel_label_clearance(fig, [axis])


if __name__ == "__main__":
    unittest.main()
