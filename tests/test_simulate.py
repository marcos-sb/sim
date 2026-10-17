import csv
import io
import math
import os
import random
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

import matplotlib.pyplot  # noqa: F401  (load-bearing: @patch target needs it)
import numpy as np

# Import the modules to test
import model
import simulate


class TestSimulateFunctions(unittest.TestCase):
    def setUp(self):
        # Use a fixed seed for reproducibility
        self.seed = 42
        np.random.seed(self.seed)
        random.seed(self.seed)

    def _write_min_cfg(self, path="test_min.toml", generations=2, num_traits=1):
        """A minimal valid config for main()-level tests."""
        lists = ", ".join(["0.1"] * num_traits)
        with open(path, "w") as f:
            f.write(
                f"[run]\ngenerations = {generations}\n"
                "[traits]\n"
                f'init_means = [{", ".join(["1.0"] * num_traits)}]\n'
                f"init_stds = [{lists}]\n"
                f"mutation_stds = [{lists}]\n"
            )
        return path

    def test_simulate_single_trait_length(self):
        """Test that simulate returns correct number of generations."""
        pop_size = 20
        generations = 10
        avg = simulate.simulate(
            pop_size=pop_size,
            generations=generations,
            mutation_std=0.0,
            initial_mean=1.0,
            initial_std=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
        )
        self.assertEqual(len(avg), generations)
        # With zero mutation and zero initial std, all values should be exactly initial_mean
        self.assertTrue(all(abs(a - 1.0) < 1e-12 for a in avg))

    def test_simulate_parity_with_multi_engine(self):
        """simulate() with the same seed reproduces the multi-engine
        single-trait run bitwise (RNG call order is preserved)."""
        kwargs = {
            "pop_size": 9,
            "generations": 6,
            "mutation_std": 0.05,
            "initial_mean": 1.0,
            "initial_std": 0.1,
            "position_std": 0.05,
            "position_min": 0.0,
            "position_max": 10.0,
        }
        random.seed(11)
        legacy = model.simulate(**kwargs)
        random.seed(11)
        multi = model.simulate_multi(
            pop_size=9,
            generations=6,
            num_traits=1,
            init_means=[1.0],
            init_stds=[0.1],
            mutation_stds=[0.05],
            position_std=0.05,
            position_min=0.0,
            position_max=10.0,
        )
        self.assertEqual(legacy, multi[0])

    def test_simulate_multi_trait_length(self):
        """Test that simulate_multi returns correct number of generations per trait."""
        pop_size = 15
        generations = 7
        num_traits = 3
        avg_per_trait = simulate.simulate_multi(
            pop_size=pop_size,
            generations=generations,
            num_traits=num_traits,
            init_means=[0.0, 1.0, 2.0],
            init_stds=[0.0, 0.0, 0.0],
            mutation_stds=[0.0, 0.0, 0.0],
            optimum=[0.0, 1.0, 2.0],
            selection_strength=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
        )
        self.assertEqual(len(avg_per_trait), num_traits)
        for trait_avg in avg_per_trait:
            self.assertEqual(len(trait_avg), generations)
        # Check that averages match initial means (since no mutation/variation)
        expected = [0.0, 1.0, 2.0]
        for i, trait_avg in enumerate(avg_per_trait):
            self.assertTrue(all(abs(a - expected[i]) < 1e-12 for a in trait_avg))

    def test_nearest_neighbours_grid_matches_linear_scan(self):
        """The grid NN agrees with the linear scan on achieved distance,
        including stacked and clustered configurations."""
        cases = []
        # random spread
        for _ in range(20):
            n = random.choice([2, 5, 17, 50])
            cases.append(
                [{"pos": [random.uniform(0.0, 10.0), random.uniform(0.0, 10.0)]} for _ in range(n)]
            )
        # everything stacked on one point
        cases.append([{"pos": [3.0, 3.0]} for _ in range(6)])
        # two far-apart clusters (ring expansion across empty space)
        cases.append(
            [{"pos": [random.uniform(0.0, 0.2), random.uniform(0.0, 0.2)]} for _ in range(8)]
            + [
                {"pos": [random.uniform(100.0, 100.2), random.uniform(100.0, 100.2)]}
                for _ in range(7)
            ]
        )
        for population in cases:
            via_grid = model._nearest_neighbours(population)
            for i, creature in enumerate(population):
                scan = model._nearest_neighbour(creature, population)
                if scan is None:
                    self.assertIsNone(via_grid[i])
                else:
                    self.assertAlmostEqual(
                        math.dist(creature["pos"], via_grid[i]["pos"]),
                        math.dist(creature["pos"], scan["pos"]),
                        places=9,
                    )

    def test_nearest_neighbour_plus_one(self):
        """Reproduction partner is never the creature itself."""
        pop = [{"traits": [1.0], "pos": [0.0, 0.0]}]
        self.assertEqual(model._nearest_neighbours(pop), [None])

    def test_positions_stay_within_bounds(self):
        """Positions stay within [position_min, position_max] even with large position_std."""
        positions = []

        def cb(gen, population):
            positions.extend((c["pos"][0], c["pos"][1]) for c in population)

        simulate.simulate(
            pop_size=20,
            generations=50,
            mutation_std=0.0,
            initial_mean=1.0,
            initial_std=0.0,
            position_std=5.0,
            position_min=0.0,
            position_max=10.0,
            plot_callback=None,
            anim_callback=cb,
        )
        self.assertTrue(positions)
        for x, y in positions:
            self.assertGreaterEqual(x, 0.0)
            self.assertLessEqual(x, 10.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y, 10.0)

    def test_offspring_traits_come_from_parents(self):
        """With zero mutation, every trait value must trace back to an initial value (passed on, not blended)."""
        seen = []

        def cb(gen, population):
            seen.append([c["trait"] for c in population])

        simulate.simulate(
            pop_size=6,
            generations=10,
            mutation_std=0.0,
            initial_mean=0.0,
            initial_std=5.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            plot_callback=None,
            anim_callback=cb,
        )
        initial = set(seen[0])
        self.assertGreater(len(initial), 1)
        for gen_traits in seen[1:]:
            for t in gen_traits:
                self.assertIn(t, initial)

    def test_multi_offspring_traits_come_from_parents(self):
        """With zero mutation, each trait value must trace back to an initial value of that trait."""
        seen = []

        def cb(gen, population):
            seen.append([list(c["traits"]) for c in population])

        simulate.simulate_multi(
            pop_size=6,
            generations=10,
            num_traits=3,
            init_means=[0.0, 1.0, 2.0],
            init_stds=[5.0, 5.0, 5.0],
            mutation_stds=[0.0, 0.0, 0.0],
            optimum=[0.0, 1.0, 2.0],
            selection_strength=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            plot_callback=None,
            anim_callback=cb,
        )
        initial_vals = [{v[i] for v in seen[0]} for i in range(3)]
        self.assertTrue(all(len(s) > 1 for s in initial_vals))
        for gen in seen[1:]:
            for traits in gen:
                for i, t in enumerate(traits):
                    self.assertIn(t, initial_vals[i])

    def test_offspring_position_is_midpoint_of_parents(self):
        """With pop_size=2 and zero position mutation, offspring sit at the parents' midpoint (distance-linked interaction)."""
        calls = []

        def cb(gen, population):
            calls.append([c["pos"] for c in population])

        simulate.simulate(
            pop_size=2,
            generations=2,
            mutation_std=0.0,
            initial_mean=1.0,
            initial_std=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            plot_callback=None,
            anim_callback=cb,
        )
        p0, p1 = calls[0]
        mid = [(p0[i] + p1[i]) / 2.0 for i in range(2)]
        for pos in calls[1]:
            self.assertEqual(pos, mid)

    def test_main_backward_compatibility_creates_csv(self):
        """Test that main() with a single-trait config creates a CSV with correct format."""
        test_file = "test_back_compat.csv"
        test_cfg = "test_back_compat.toml"
        if os.path.exists(test_file):
            os.remove(test_file)
        with open(test_cfg, "w") as f:
            f.write("[run]\ngenerations = 5\n[traits]\ninit_means = [1.0]\n")
        # Backup sys.argv
        old_argv = sys.argv.copy()
        try:
            sys.argv = [
                "simulate.py",
                "--config",
                test_cfg,
                "--pop-size",
                "8",
                "--output",
                test_file,
            ]
            simulate.main()
            # Check file exists
            self.assertTrue(os.path.exists(test_file))
            # Read CSV
            with open(test_file, "r") as f:
                reader = list(csv.reader(f))
            # Header: unified format now includes speed/size columns
            self.assertEqual(reader[0], ["generation", "trait0_avg", "speed_avg", "size_avg"])
            # Number of rows = generations + 1 (header)
            self.assertEqual(len(reader), 6)  # 5 generations + header
            # All rows after header should have 4 columns
            for row in reader[1:]:
                self.assertEqual(len(row), 4)
                # Ensure they are numeric
                float(row[0])  # generation
                float(row[1])  # trait average
        finally:
            sys.argv = old_argv
            for path in (test_file, test_cfg):
                if os.path.exists(path):
                    os.remove(path)

    def test_single_creature_does_not_crash(self):
        """A lone creature (no partner) reproduces by cloning in both modes."""
        avg = simulate.simulate(
            pop_size=1,
            generations=3,
            mutation_std=0.05,
            initial_mean=1.0,
            initial_std=0.1,
            position_std=0.05,
            position_min=0.0,
            position_max=1.0,
        )
        self.assertEqual(len(avg), 3)
        result = simulate.simulate_multi(
            pop_size=1,
            generations=3,
            num_traits=2,
            init_means=[0.0, 1.0],
            init_stds=[0.1, 0.1],
            mutation_stds=[0.05, 0.05],
            optimum=[0.0, 1.0],
            selection_strength=0.0,
            position_std=0.05,
            position_min=0.0,
            position_max=1.0,
        )
        self.assertEqual([len(t) for t in result], [3, 3])

    def test_selection_produces_independent_creatures(self):
        """Selection copies offspring: no two population slots share one dict."""
        frames = []

        def cb(gen, population):
            frames.append([id(c) for c in population])

        simulate.simulate_multi(
            pop_size=10,
            generations=4,
            num_traits=2,
            init_means=[0.0, 0.0],
            init_stds=[1.0, 1.0],
            mutation_stds=[0.1, 0.1],
            optimum=[5.0, 5.0],
            selection_strength=2.0,
            position_std=0.05,
            position_min=0.0,
            position_max=10.0,
            anim_callback=cb,
        )
        self.assertEqual(len(frames), 4)
        for ids in frames:
            self.assertEqual(len(set(ids)), 10)

    def test_optimum_required_with_selection(self):
        """Selection without an optimum is rejected with a clear error."""
        with self.assertRaises(ValueError):
            simulate.simulate_multi(
                pop_size=5,
                generations=3,
                num_traits=2,
                init_means=[0.0, 1.0],
                init_stds=[0.1, 0.1],
                mutation_stds=[0.05, 0.05],
                optimum=None,
                selection_strength=1.0,
            )

    def test_pop_size_zero_rejected(self):
        """pop_size < 1 is rejected instead of crashing on an empty population."""
        with self.assertRaises(ValueError):
            simulate.simulate(pop_size=0, generations=3)
        with self.assertRaises(ValueError):
            simulate.simulate_multi(
                pop_size=0,
                generations=3,
                num_traits=2,
                init_means=[0.0, 1.0],
                init_stds=[0.1, 0.1],
                mutation_stds=[0.05, 0.05],
                optimum=[0.0, 1.0],
            )

    def test_seed_reproducible(self):
        """Two runs with the same --seed produce byte-identical CSVs."""
        files = ["test_seed_a.csv", "test_seed_b.csv"]
        for f in files:
            if os.path.exists(f):
                os.remove(f)
        test_cfg = "test_seed.toml"
        with open(test_cfg, "w") as f:
            f.write("[run]\ngenerations = 3\n[traits]\ninit_means = [1.0]\n")
        old_argv = sys.argv.copy()
        try:
            for f in files:
                sys.argv = [
                    "simulate.py",
                    "--config",
                    test_cfg,
                    "--seed",
                    "7",
                    "--pop-size",
                    "6",
                    "--output",
                    f,
                ]
                simulate.main()
            with open(files[0], "rb") as fa, open(files[1], "rb") as fb:
                self.assertEqual(fa.read(), fb.read())
        finally:
            sys.argv = old_argv
            for f in files:
                if os.path.exists(f):
                    os.remove(f)
            if os.path.exists(test_cfg):
                os.remove(test_cfg)

    def test_speed_size_cli_controls(self):
        """[body].speed/size mean with zero std sets the initial averages."""
        test_file = "test_body_cli.csv"
        test_cfg = "test_body_cli.toml"
        if os.path.exists(test_file):
            os.remove(test_file)
        with open(test_cfg, "w") as f:
            f.write("""[run]
generations = 2
[body]
speed_mean = 2.0
speed_std = 0.0
size_mean = 0.5
size_std = 0.0
[traits]
init_means = [0.0, 1.0]
init_stds = [0.1, 0.1]
mutation_stds = [0.05, 0.05]
optimum = [0.0, 1.0]
""")
        old_argv = sys.argv.copy()
        try:
            sys.argv = [
                "simulate.py",
                "--config",
                test_cfg,
                "--pop-size",
                "10",
                "--output",
                test_file,
            ]
            simulate.main()
            with open(test_file, "r") as f:
                reader = list(csv.reader(f))
            self.assertEqual(
                reader[0], ["generation", "trait0_avg", "trait1_avg", "speed_avg", "size_avg"]
            )
            # generation 0 is recorded before any mutation: exact init values
            self.assertAlmostEqual(float(reader[1][3]), 2.0, places=12)
            self.assertAlmostEqual(float(reader[1][4]), 0.5, places=12)
        finally:
            sys.argv = old_argv
            for path in (test_file, test_cfg):
                if os.path.exists(path):
                    os.remove(path)

    def test_main_multi_trait_creates_csv(self):
        """Test that main() with a multi-trait config creates correct CSV."""
        test_file = "test_multi.csv"
        test_cfg = "test_multi.toml"
        if os.path.exists(test_file):
            os.remove(test_file)
        with open(test_cfg, "w") as f:
            f.write("""[run]
generations = 4
[traits]
init_means = [0.0, 1.0]
init_stds = [0.1, 0.1]
mutation_stds = [0.05, 0.05]
optimum = [0.0, 1.0]
selection_strength = 0.0
""")
        old_argv = sys.argv.copy()
        try:
            sys.argv = [
                "simulate.py",
                "--config",
                test_cfg,
                "--pop-size",
                "6",
                "--output",
                test_file,
            ]
            simulate.main()
            self.assertTrue(os.path.exists(test_file))
            with open(test_file, "r") as f:
                reader = list(csv.reader(f))
            self.assertEqual(
                reader[0], ["generation", "trait0_avg", "trait1_avg", "speed_avg", "size_avg"]
            )
            self.assertEqual(len(reader), 5)  # 4 generations + header
            for row in reader[1:]:
                self.assertEqual(len(row), 5)
                for value in row:
                    float(value)
        finally:
            sys.argv = old_argv
            for path in (test_file, test_cfg):
                if os.path.exists(path):
                    os.remove(path)

    @patch("matplotlib.pyplot")
    def test_live_plot_callback_called(self, mock_plt):
        """Test that live_plot callback triggers matplotlib calls."""
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_ax.plot.return_value = [MagicMock()]  # unpackable for `line, = ax.plot(...)`
        mock_plt.pause = MagicMock()
        mock_plt.ion = MagicMock()
        mock_plt.ioff = MagicMock()
        mock_plt.show = MagicMock()
        test_file = "test_lp.csv"
        test_cfg = self._write_min_cfg("test_lp_cfg.toml", generations=2)
        if os.path.exists(test_file):
            os.remove(test_file)
        old_argv = sys.argv.copy()
        try:
            sys.argv = [
                "simulate.py",
                "--config",
                test_cfg,
                "--live-plot",
                "--pop-size",
                "5",
                "--output",
                test_file,
            ]
            # window "closes" after 2 generations
            mock_plt.get_fignums.side_effect = [[1], [1], []]
            # We need to prevent actual SystemExit from shutting down; but main will not exit.
            simulate.main()
            # Ensure plt.ion was called
            mock_plt.ion.assert_called()
            # one pause per generation; the run ends when the window closes
            self.assertEqual(mock_plt.pause.call_count, 2)
            # Ensure plt.ioff and plt.show called at end
            mock_plt.ioff.assert_called()
            mock_plt.show.assert_called()
        finally:
            sys.argv = old_argv
            for path in (test_file, test_cfg):
                if os.path.exists(path):
                    os.remove(path)

    @patch("matplotlib.pyplot")
    @patch("simulate.np")
    def test_animate_callback_called_single_trait(self, mock_np, mock_plt):
        """Test that animate callback works for single-trait mode."""
        # Configure numpy mock
        mock_np.array.return_value = np.array([0.1, 0.2])
        mock_np.min.return_value = 0.1
        mock_np.max.return_value = 0.2
        mock_np.zeros_like.return_value = np.array([0.0, 0.0])
        # Mock viridis colormap
        mock_viridis = MagicMock()
        mock_viridis.return_value = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        mock_cm = MagicMock()
        mock_cm.viridis = mock_viridis
        mock_plt.cm = mock_cm
        # Mock matplotlib components
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.pause = MagicMock()
        mock_plt.ion = MagicMock()
        mock_plt.ioff = MagicMock()
        mock_plt.show = MagicMock()
        # Mock scatter returns an object with set_offsets and set_color
        mock_scat = MagicMock()
        mock_ax.scatter.return_value = mock_scat
        test_file = "test_an.csv"
        test_cfg = self._write_min_cfg("test_an_cfg.toml", generations=2)
        if os.path.exists(test_file):
            os.remove(test_file)
        old_argv = sys.argv.copy()
        try:
            sys.argv = [
                "simulate.py",
                "--config",
                test_cfg,
                "--animate",
                "--pop-size",
                "3",
                "--output",
                test_file,
            ]
            # window "closes" after 2 generations
            mock_plt.get_fignums.side_effect = [[1], [1], []]
            simulate.main()
            # Check that scatter was created and updated
            self.assertTrue(mock_ax.scatter.called)
            # set_offsets and set_color should be called each generation after first
            # First generation creates scatter, subsequent calls update.
            # With 2 generations, we expect 1 scatter call, then 1 update for set_offsets and set_color each.
            self.assertEqual(mock_scat.set_offsets.call_count, 1)
            self.assertEqual(mock_scat.set_color.call_count, 1)
            mock_plt.ion.assert_called()
            self.assertGreaterEqual(mock_plt.pause.call_count, 2)
            mock_plt.ioff.assert_called()
            mock_plt.show.assert_called()
        finally:
            sys.argv = old_argv
            for path in (test_file, test_cfg):
                if os.path.exists(path):
                    os.remove(path)

    @patch("matplotlib.pyplot")
    @patch("simulate.np")
    def test_animate_fixes_axis_to_position_range(self, mock_np, mock_plt):
        """--animate pins the axis to [position_min, position_max] once; stray creatures cannot stretch it."""
        mock_np.array.return_value = np.array([0.1, 0.2])
        mock_np.min.return_value = 0.1
        mock_np.max.return_value = 0.2
        mock_np.zeros_like.return_value = np.array([0.0, 0.0])
        mock_viridis = MagicMock()
        mock_viridis.return_value = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
        mock_cm = MagicMock()
        mock_cm.viridis = mock_viridis
        mock_plt.cm = mock_cm
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.pause = MagicMock()
        mock_plt.ion = MagicMock()
        mock_plt.ioff = MagicMock()
        mock_plt.show = MagicMock()
        mock_scat = MagicMock()
        mock_ax.scatter.return_value = mock_scat
        test_file = "test_axis.csv"
        test_cfg = "test_axis_cfg.toml"
        with open(test_cfg, "w") as f:
            f.write(
                "[run]\ngenerations = 3\n"
                "[space]\nposition_min = 0.0\nposition_max = 10.0\n"
                "[traits]\ninit_means = [1.0]\ninit_stds = [0.1]\n"
                "mutation_stds = [0.1]\n"
            )
        if os.path.exists(test_file):
            os.remove(test_file)
        old_argv = sys.argv.copy()
        try:
            sys.argv = [
                "simulate.py",
                "--config",
                test_cfg,
                "--animate",
                "--pop-size",
                "5",
                "--output",
                test_file,
            ]
            # window "closes" after 1 generation
            mock_plt.get_fignums.side_effect = [[1], []]
            simulate.main()
            # axis fixed once to the position range with 5% padding
            mock_ax.set_xlim.assert_called_once_with(-0.5, 10.5)
            mock_ax.set_ylim.assert_called_once_with(-0.5, 10.5)
        finally:
            sys.argv = old_argv
            for path in (test_file, test_cfg):
                if os.path.exists(path):
                    os.remove(path)

    def test_widen_scale_only_grows(self):
        """The fixed color scale only widens, never shrinks."""
        self.assertEqual(model._widen_scale(None, None, [2.0, 8.0]), (2.0, 8.0))
        self.assertEqual(model._widen_scale(2.0, 8.0, [0.0, 5.0]), (0.0, 8.0))
        self.assertEqual(model._widen_scale(0.0, 8.0, [3.0, 4.0]), (0.0, 8.0))

    def test_color_norm_fixed_scale(self):
        """Normalization against a fixed scale maps identical values to identical colors."""
        self.assertEqual(model._color_norm([1.0, 2.0, 3.0], 1.0, 3.0), [0.0, 0.5, 1.0])
        self.assertEqual(model._color_norm([5.0, 5.0], 5.0, 5.0), [0.0, 0.0])

    def test_move_toward_similar_traits(self):
        """Similar-trait creatures attract each other; dissimilar ones are left alone."""
        population = [
            {"trait": 1.0, "pos": [0.0, 0.0]},
            {"trait": 1.0, "pos": [2.0, 0.0]},
            {"trait": 9.0, "pos": [9.0, 9.0]},
        ]
        model._move_toward_similar(
            population,
            similarity_scale=0.5,
            move_step=0.5,
            position_std=0.0,
            position_min=0.0,
            position_max=10.0,
        )
        # the identical pair meets at its centroid
        self.assertAlmostEqual(population[0]["pos"][0], 1.0)
        self.assertAlmostEqual(population[0]["pos"][1], 0.0)
        self.assertAlmostEqual(population[1]["pos"][0], 1.0)
        self.assertAlmostEqual(population[1]["pos"][1], 0.0)
        # the dissimilar creature does not move
        self.assertAlmostEqual(population[2]["pos"][0], 9.0)
        self.assertAlmostEqual(population[2]["pos"][1], 9.0)

    def test_move_toward_similar_multi_traits(self):
        """Multi-trait creatures cluster by trait-space distance."""
        population = [
            {"traits": [0.0, 0.0], "pos": [0.0, 0.0]},
            {"traits": [0.0, 0.0], "pos": [2.0, 0.0]},
            {"traits": [5.0, 5.0], "pos": [5.0, 5.0]},
        ]
        model._move_toward_similar(
            population,
            similarity_scale=1.0,
            move_step=0.5,
            position_std=0.0,
            position_min=0.0,
            position_max=10.0,
        )
        self.assertAlmostEqual(population[0]["pos"][0], 1.0)
        self.assertAlmostEqual(population[1]["pos"][0], 1.0)
        self.assertAlmostEqual(population[2]["pos"][0], 5.0)

    def test_movement_stays_in_bounds(self):
        """Movement toward a centroid outside the bounds is clamped."""
        population = [
            {"trait": 1.0, "pos": [0.0, 0.0]},
            {"trait": 1.0, "pos": [-5.0, 0.0]},
        ]
        model._move_toward_similar(
            population,
            similarity_scale=0.5,
            move_step=0.5,
            position_std=0.0,
            position_min=0.0,
            position_max=10.0,
        )
        self.assertEqual(population[0]["pos"][0], 0.0)
        self.assertEqual(population[1]["pos"][0], 0.0)

    def test_creatures_move_over_generations(self):
        """Creatures move each generation and stay within bounds."""
        positions = []

        def cb(gen, population):
            positions.append([tuple(c["pos"]) for c in population])

        simulate.simulate(
            pop_size=10,
            generations=10,
            mutation_std=0.0,
            initial_mean=1.0,
            initial_std=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=10.0,
            similarity_scale=0.5,
            move_step=0.5,
            plot_callback=None,
            anim_callback=cb,
        )
        self.assertEqual(len(positions), 10)
        # identical traits: everyone converges toward the common centroid
        self.assertNotEqual(positions[0], positions[-1])
        for frame in positions:
            for x, y in frame:
                self.assertGreaterEqual(x, 0.0)
                self.assertLessEqual(x, 10.0)
                self.assertGreaterEqual(y, 0.0)
                self.assertLessEqual(y, 10.0)

    def test_stop_when_ends_simulation_early(self):
        """A stop_when predicate ends the run even with generations remaining."""
        checks = []

        def stop():
            checks.append(None)
            return len(checks) >= 4  # stop after 3 generations

        avg = simulate.simulate(
            pop_size=5,
            generations=1000,
            mutation_std=0.0,
            initial_mean=1.0,
            initial_std=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            plot_callback=None,
            anim_callback=None,
            stop_when=stop,
        )
        self.assertEqual(len(avg), 3)

    def test_stop_when_ends_multi_simulation_early(self):
        """stop_when also ends multi-trait runs early."""
        checks = []

        def stop():
            checks.append(None)
            return len(checks) >= 2  # stop after 1 generation

        result = simulate.simulate_multi(
            pop_size=4,
            generations=100,
            num_traits=2,
            init_means=[0.0, 1.0],
            init_stds=[0.0, 0.0],
            mutation_stds=[0.0, 0.0],
            optimum=[0.0, 1.0],
            selection_strength=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            plot_callback=None,
            anim_callback=None,
            stop_when=stop,
        )
        self.assertEqual([len(t) for t in result], [1, 1])

    def test_generations_none_requires_stop_when(self):
        """generations=None without stop_when is rejected (it would run forever)."""
        with self.assertRaises(ValueError):
            simulate.simulate(pop_size=5, generations=None, initial_mean=1.0, initial_std=0.0)

    def test_missing_required_multi_trait_args(self):
        """Running without a [traits] section is rejected with a clear error."""
        test_cfg = "test_no_traits.toml"
        with open(test_cfg, "w") as f:
            f.write("[run]\ngenerations = 2\n")
        old_argv = sys.argv.copy()
        try:
            sys.argv = ["simulate.py", "--config", test_cfg]
            with self.assertRaises(SystemExit):
                simulate.main()
        finally:
            sys.argv = old_argv
            if os.path.exists(test_cfg):
                os.remove(test_cfg)

    def test_no_config_arg_rejected(self):
        """Running without --config prints instructions and exits."""
        old_argv = sys.argv.copy()
        try:
            sys.argv = ["simulate.py"]
            with self.assertRaises(SystemExit):
                simulate.main()
        finally:
            sys.argv = old_argv

    def test_config_template_printed(self):
        """--config-template prints a valid config and exits without running."""
        old_argv = sys.argv.copy()
        old_stdout = io.StringIO()
        try:
            sys.argv = ["simulate.py", "--config-template"]
            with redirect_stdout(old_stdout):
                simulate.main()
            text = old_stdout.getvalue()
            self.assertIn("[traits]", text)
            self.assertIn("init_means", text)
        finally:
            sys.argv = old_argv

    def test_population_stats(self):
        """_population_stats computes averages, stds, and mean spread."""
        population = [
            {"trait": 1.0, "pos": [0.0, 0.0]},
            {"trait": 2.0, "pos": [3.0, 0.0]},
            {"trait": 3.0, "pos": [0.0, 4.0]},
        ]
        stats = model._population_stats(population)
        self.assertEqual(stats["averages"], [2.0])
        self.assertAlmostEqual(stats["stds"][0], math.sqrt(2.0 / 3.0), places=7)
        # pair distances are 3, 4, and 5
        self.assertAlmostEqual(stats["spread"], 4.0, places=7)

    def test_population_stats_multi(self):
        """_population_stats handles multi-trait populations."""
        population = [
            {"traits": [1.0, 0.0], "pos": [0.0, 0.0]},
            {"traits": [2.0, 1.0], "pos": [3.0, 0.0]},
            {"traits": [3.0, 2.0], "pos": [0.0, 4.0]},
        ]
        stats = model._population_stats(population)
        self.assertEqual(stats["averages"], [2.0, 1.0])
        for std in stats["stds"]:
            self.assertAlmostEqual(std, math.sqrt(2.0 / 3.0), places=7)
        self.assertAlmostEqual(stats["spread"], 4.0, places=7)

    def test_format_stats(self):
        """_format_stats renders the stats text block."""
        text = model._format_stats(5, {"averages": [1.0], "stds": [0.5], "spread": 1.234})
        self.assertEqual(text, "gen 5\navg 1.0000\nstd 0.5000\nspread 1.23")

    def test_stats_callback_receives_stats(self):
        """stats_callback is called each generation with the stats dict."""
        calls = []

        def cb(gen, stats):
            calls.append((gen, stats))

        simulate.simulate(
            pop_size=5,
            generations=3,
            mutation_std=0.0,
            initial_mean=1.0,
            initial_std=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            plot_callback=None,
            anim_callback=None,
            stats_callback=cb,
        )
        self.assertEqual([gen for gen, _ in calls], [0, 1, 2])
        for _, stats in calls:
            self.assertEqual(stats["averages"], [1.0])
            self.assertEqual(stats["stds"], [0.0])
            self.assertGreaterEqual(stats["spread"], 0.0)

    def test_stats_callback_multi(self):
        """stats_callback also fires for multi-trait runs."""
        calls = []

        def cb(gen, stats):
            calls.append((gen, stats))

        simulate.simulate_multi(
            pop_size=5,
            generations=3,
            num_traits=2,
            init_means=[0.0, 2.0],
            init_stds=[0.0, 0.0],
            mutation_stds=[0.0, 0.0],
            optimum=[0.0, 2.0],
            selection_strength=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            plot_callback=None,
            anim_callback=None,
            stats_callback=cb,
        )
        self.assertEqual([gen for gen, _ in calls], [0, 1, 2])
        for _, stats in calls:
            self.assertEqual(stats["averages"], [0.0, 2.0])
            self.assertEqual(stats["stds"], [0.0, 0.0])

    def test_update_elements_boosts_nearby_creatures(self):
        """Elements boost the trait of creatures within their radius."""
        population = [
            {"trait": 1.0, "pos": [5.0, 5.0]},  # at the center
            {"trait": 1.0, "pos": [6.0, 5.0]},  # halfway to the edge
            {"trait": 1.0, "pos": [0.0, 0.0]},  # outside the radius
        ]
        elements = [{"pos": [5.0, 5.0], "trait": 0, "radius": 2.0, "boost": 0.1, "lifetime": 5}]
        model._update_elements(
            population,
            elements,
            num_traits=1,
            rate=0.0,
            radius=2.0,
            boost=0.1,
            lifetime=5,
            position_min=0.0,
            position_max=10.0,
            element_max=50,
        )
        self.assertAlmostEqual(population[0]["trait"], 1.1, places=7)
        self.assertAlmostEqual(population[1]["trait"], 1.05, places=7)
        self.assertAlmostEqual(population[2]["trait"], 1.0, places=7)
        self.assertEqual(len(elements), 1)
        self.assertEqual(elements[0]["lifetime"], 4)

    def test_update_elements_expire_and_spawn(self):
        """Elements expire after their lifetime and new ones spawn."""
        population = []
        elements = [{"pos": [5.0, 5.0], "trait": 0, "radius": 1.0, "boost": 0.1, "lifetime": 1}]
        model._update_elements(
            population,
            elements,
            num_traits=2,
            rate=2.0,
            radius=1.0,
            boost=0.1,
            lifetime=7,
            position_min=0.0,
            position_max=10.0,
            element_max=50,
        )
        # the lifetime-1 element expired; exactly 2 new ones spawned
        self.assertEqual(len(elements), 2)
        for element in elements:
            self.assertIn(element["trait"], (0, 1))
            self.assertEqual(element["radius"], 1.0)
            self.assertEqual(element["boost"], 0.1)
            self.assertEqual(element["lifetime"], 7)
            for coord in element["pos"]:
                self.assertGreaterEqual(coord, 0.0)
                self.assertLessEqual(coord, 10.0)

    def test_update_elements_max_cap(self):
        """The element cap is a hard limit: no spawns while full."""
        elements = [
            {"pos": [1.0, 1.0], "trait": 0, "radius": 1.0, "boost": 0.1, "lifetime": 10}
            for _ in range(2)
        ]
        model._update_elements(
            [],
            elements,
            num_traits=1,
            rate=5.0,
            radius=1.0,
            boost=0.1,
            lifetime=10,
            position_min=0.0,
            position_max=10.0,
            element_max=2,
        )
        # at capacity: nothing spawns and existing elements are kept
        self.assertEqual(len(elements), 2)
        self.assertEqual([e["lifetime"] for e in elements], [9, 9])

    def test_update_elements_fills_free_slots(self):
        """Spawning fills free slots up to the cap."""
        elements = [{"pos": [1.0, 1.0], "trait": 0, "radius": 1.0, "boost": 0.1, "lifetime": 10}]
        model._update_elements(
            [],
            elements,
            num_traits=1,
            rate=5.0,
            radius=1.0,
            boost=0.1,
            lifetime=10,
            position_min=0.0,
            position_max=10.0,
            element_max=3,
        )
        # only the two free slots are filled
        self.assertEqual(len(elements), 3)

    def test_simulate_elements_stats_callback(self):
        """simulate reports the active element count in stats."""
        calls = []

        def cb(gen, stats):
            calls.append((gen, stats))

        simulate.simulate(
            pop_size=5,
            generations=3,
            mutation_std=0.0,
            initial_mean=1.0,
            initial_std=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            elements=True,
            element_rate=10.0,
            element_radius=0.5,
            element_boost=0.05,
            element_lifetime=20,
            element_max=50,
            plot_callback=None,
            anim_callback=None,
            stats_callback=cb,
        )
        self.assertEqual([gen for gen, _ in calls], [0, 1, 2])
        # elements accumulate: 10 spawn per generation and none
        # expire within 3 generations (lifetime 20)
        self.assertEqual([stats["elements"] for _, stats in calls], [10, 20, 30])

    def test_simulate_multi_elements_stats_callback(self):
        """simulate_multi reports the active element count in stats."""
        calls = []

        def cb(gen, stats):
            calls.append((gen, stats))

        simulate.simulate_multi(
            pop_size=5,
            generations=3,
            num_traits=3,
            init_means=[0.0, 1.0, 2.0],
            init_stds=[0.0, 0.0, 0.0],
            mutation_stds=[0.0, 0.0, 0.0],
            optimum=[0.0, 1.0, 2.0],
            selection_strength=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            elements=True,
            element_rate=10.0,
            element_radius=0.5,
            element_boost=0.05,
            element_lifetime=20,
            element_max=50,
            plot_callback=None,
            anim_callback=None,
            stats_callback=cb,
        )
        self.assertEqual([gen for gen, _ in calls], [0, 1, 2])
        # elements accumulate: 10 spawn per generation and none
        # expire within 3 generations (lifetime 20)
        self.assertEqual([stats["elements"] for _, stats in calls], [10, 20, 30])
        for _, stats in calls:
            self.assertEqual(len(stats["averages"]), 3)

    def test_simulate_single_element_persists(self):
        """With a cap of 1, the element stays for its full
        lifetime and is replaced only when it expires."""
        frames = []

        def cb(gen, population, elements):
            frames.append([dict(e) for e in elements])

        simulate.simulate(
            pop_size=5,
            generations=7,
            mutation_std=0.0,
            initial_mean=1.0,
            initial_std=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            elements=True,
            element_rate=1.0,
            element_radius=0.5,
            element_boost=0.05,
            element_lifetime=5,
            element_max=1,
            plot_callback=None,
            anim_callback=cb,
        )
        self.assertEqual(len(frames), 7)
        for elements in frames:
            self.assertEqual(len(elements), 1)
        # the element ages 5,4,3,2,1, then a fresh one appears
        self.assertEqual([e[0]["lifetime"] for e in frames], [5, 4, 3, 2, 1, 5, 4])

    def test_format_stats_with_elements(self):
        """_format_stats appends the element count when present."""
        stats = {"averages": [1.0], "stds": [0.5], "spread": 1.0, "elements": 7}
        text = model._format_stats(2, stats)
        self.assertEqual(text, "gen 2\navg 1.0000\nstd 0.5000\nspread 1.00\n" "elements 7")


if __name__ == "__main__":
    unittest.main()
