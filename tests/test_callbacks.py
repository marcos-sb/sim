import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import matplotlib.pyplot  # noqa: F401  (load-bearing: @patch target needs it)

import simulate


class TestCallbacks(unittest.TestCase):
    def setUp(self):
        self.test_output = "test_output.csv"
        # minimal config for main()-level tests
        self.test_cfg = "test_callbacks.toml"
        with open(self.test_cfg, "w") as f:
            f.write(
                "[run]\ngenerations = 1\n"
                "[traits]\ninit_means = [1.0]\ninit_stds = [0.1]\n"
                "mutation_stds = [0.1]\n"
            )

    def tearDown(self):
        if os.path.exists(self.test_output):
            os.remove(self.test_output)
        if os.path.exists(self.test_cfg):
            os.remove(self.test_cfg)

    def test_simulate_invokes_plot_callback(self):
        """Test that simulate calls plot_callback each generation."""
        callbacks = []

        def cb(gen, averages):
            callbacks.append((gen, averages.copy()))

        # Run simulation with zero mutation to keep deterministic
        simulate.simulate(
            pop_size=5,
            generations=3,
            mutation_std=0.0,
            initial_mean=1.0,
            initial_std=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            plot_callback=cb,
            anim_callback=None,
        )
        # Should have been called for generations 0,1,2
        self.assertEqual(len(callbacks), 3)
        self.assertEqual([c[0] for c in callbacks], [0, 1, 2])
        # Each averages list should have one element (single trait)
        for gen, avg_list in callbacks:
            self.assertEqual(len(avg_list), 1)
            # Value should be close to 1.0 (no mutation)
            self.assertAlmostEqual(avg_list[0], 1.0, places=6)

    def test_simulate_multi_invokes_plot_callback(self):
        """Test that simulate_multi calls plot_callback each generation."""
        callbacks = []

        def cb(gen, averages):
            callbacks.append((gen, averages.copy()))

        simulate.simulate_multi(
            pop_size=4,
            generations=2,
            num_traits=2,
            init_means=[0.0, 2.0],
            init_stds=[0.0, 0.0],
            mutation_stds=[0.0, 0.0],
            optimum=[0.0, 2.0],
            selection_strength=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            plot_callback=cb,
            anim_callback=None,
        )
        self.assertEqual(len(callbacks), 2)
        self.assertEqual([c[0] for c in callbacks], [0, 1])
        for gen, avg_list in callbacks:
            self.assertEqual(len(avg_list), 2)
            self.assertAlmostEqual(avg_list[0], 0.0, places=6)
            self.assertAlmostEqual(avg_list[1], 2.0, places=6)

    def test_simulate_invokes_anim_callback(self):
        """Test that simulate calls anim_callback each generation with population."""
        calls = []

        def cb(gen, population):
            calls.append((gen, len(population), population[0]["trait"]))

        simulate.simulate(
            pop_size=3,
            generations=2,
            mutation_std=0.0,
            initial_mean=5.0,
            initial_std=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            plot_callback=None,
            anim_callback=cb,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual([c[0] for c in calls], [0, 1])
        for gen, size, trait0 in calls:
            self.assertEqual(size, 3)
            self.assertAlmostEqual(trait0, 5.0, places=6)

    def test_simulate_multi_invokes_anim_callback(self):
        """Test that simulate_multi calls anim_callback each generation."""
        calls = []

        def cb(gen, population):
            calls.append((gen, len(population), population[0]["traits"].copy()))

        simulate.simulate_multi(
            pop_size=2,
            generations=2,
            num_traits=2,
            init_means=[1.0, 3.0],
            init_stds=[0.0, 0.0],
            mutation_stds=[0.0, 0.0],
            optimum=[1.0, 3.0],
            selection_strength=0.0,
            position_std=0.0,
            position_min=0.0,
            position_max=1.0,
            plot_callback=None,
            anim_callback=cb,
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual([c[0] for c in calls], [0, 1])
        for gen, size, traits0 in calls:
            self.assertEqual(size, 2)
            self.assertAlmostEqual(traits0[0], 1.0, places=6)
            self.assertAlmostEqual(traits0[1], 3.0, places=6)

    @patch("matplotlib.pyplot")
    def test_main_live_plot_sets_callback(self, mock_plt):
        """Test that --live-plot results in a non-None plot_callback being passed."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.plot.return_value = [MagicMock()]  # unpackable for `line, = ax.plot(...)`
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        original_multi = simulate.simulate_multi
        captured = {}

        def mock_multi(*args, **kwargs):
            captured["plot_callback"] = kwargs.get("plot_callback")
            return [[1.0]]

        # Patch simulate.simulate_multi
        simulate.simulate_multi = mock_multi
        try:
            old_argv = sys.argv.copy()
            try:
                sys.argv = [
                    "simulate.py",
                    "--config",
                    self.test_cfg,
                    "--live-plot",
                    "--pop-size",
                    "2",
                    "--output",
                    self.test_output,
                ]
                # window "closes" after 1 generation
                mock_plt.get_fignums.side_effect = [[1], []]
                simulate.main()
            finally:
                sys.argv = old_argv
            self.assertIsNotNone(captured.get("plot_callback"))
            # Ensure it's callable
            self.assertTrue(callable(captured["plot_callback"]))
        finally:
            simulate.simulate_multi = original_multi

    @patch("matplotlib.pyplot")
    def test_main_animate_sets_callback(self, mock_plt):
        """Test that --animate results in a non-None anim_callback being passed."""
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        original_multi = simulate.simulate_multi
        captured = {}

        def mock_multi(*args, **kwargs):
            captured["anim_callback"] = kwargs.get("anim_callback")
            return [[1.0]]

        simulate.simulate_multi = mock_multi
        try:
            old_argv = sys.argv.copy()
            try:
                sys.argv = [
                    "simulate.py",
                    "--config",
                    self.test_cfg,
                    "--animate",
                    "--pop-size",
                    "2",
                    "--output",
                    self.test_output,
                ]
                # window "closes" after 1 generation
                mock_plt.get_fignums.side_effect = [[1], []]
                simulate.main()
            finally:
                sys.argv = old_argv
            self.assertIsNotNone(captured.get("anim_callback"))
            self.assertTrue(callable(captured["anim_callback"]))
        finally:
            simulate.simulate_multi = original_multi

    @patch("matplotlib.pyplot")
    def test_main_animate_sets_stats_text(self, mock_plt):
        """Test that --animate adds a stats text box updated per generation."""
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)
        original_multi = simulate.simulate_multi
        captured = {}

        def mock_multi(*args, **kwargs):
            captured["stats_callback"] = kwargs.get("stats_callback")
            return [[1.0]]

        simulate.simulate_multi = mock_multi
        try:
            old_argv = sys.argv.copy()
            try:
                sys.argv = [
                    "simulate.py",
                    "--config",
                    self.test_cfg,
                    "--animate",
                    "--pop-size",
                    "2",
                    "--output",
                    self.test_output,
                ]
                # window "closes" after 1 generation
                mock_plt.get_fignums.side_effect = [[1], []]
                simulate.main()
            finally:
                sys.argv = old_argv
            self.assertTrue(mock_ax.text.called)
            self.assertIsNotNone(captured.get("stats_callback"))
            self.assertTrue(callable(captured["stats_callback"]))
            # the real run already updated the text once; drive the
            # captured callback with known stats to check formatting
            mock_ax.text.return_value.set_text.reset_mock()
            captured["stats_callback"](
                0,
                {
                    "averages": [1.0],
                    "stds": [0.0],
                    "spread": 1.0,
                    "speed_avg": 1.0,
                    "speed_std": 0.0,
                    "size_avg": 1.0,
                    "size_std": 0.0,
                },
            )
            mock_ax.text.return_value.set_text.assert_called_once_with(
                "gen 0\navg 1.0000\nstd 0.0000\nspread 1.00"
                "\nspeed 1.0000±0.0000\nsize 1.0000±0.0000"
            )
        finally:
            simulate.simulate_multi = original_multi

    @patch("matplotlib.pyplot")
    def test_main_both_flags_set_both_callbacks(self, mock_plt):
        """Test that --live-plot and --animate together set both callbacks."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.plot.return_value = [MagicMock()]  # unpackable for `line, = ax.plot(...)`
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        original_multi = simulate.simulate_multi
        captured = {"plot": None, "anim": None}

        def mock_multi(*args, **kwargs):
            captured["plot"] = kwargs.get("plot_callback")
            captured["anim"] = kwargs.get("anim_callback")
            return [[1.0]]

        simulate.simulate_multi = mock_multi
        try:
            old_argv = sys.argv.copy()
            try:
                sys.argv = [
                    "simulate.py",
                    "--config",
                    self.test_cfg,
                    "--live-plot",
                    "--animate",
                    "--pop-size",
                    "2",
                    "--output",
                    self.test_output,
                ]
                # window "closes" after 1 generation
                mock_plt.get_fignums.side_effect = [[1], []]
                simulate.main()
            finally:
                sys.argv = old_argv
            self.assertIsNotNone(captured["plot"])
            self.assertIsNotNone(captured["anim"])
            self.assertTrue(callable(captured["plot"]))
            self.assertTrue(callable(captured["anim"]))
        finally:
            simulate.simulate_multi = original_multi

    @patch("matplotlib.pyplot")
    def test_main_without_flags_sets_no_callbacks(self, mock_plt):
        """Test that without flags, callbacks are None."""
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())
        original_multi = simulate.simulate_multi
        captured = {"plot": None, "anim": None, "stats": None, "elements": None}

        def mock_multi(*args, **kwargs):
            captured["plot"] = kwargs.get("plot_callback")
            captured["anim"] = kwargs.get("anim_callback")
            captured["stats"] = kwargs.get("stats_callback")
            captured["elements"] = kwargs.get("elements")
            return [[1.0]]

        simulate.simulate_multi = mock_multi
        try:
            old_argv = sys.argv.copy()
            try:
                sys.argv = [
                    "simulate.py",
                    "--config",
                    self.test_cfg,
                    "--pop-size",
                    "2",
                    "--output",
                    self.test_output,
                ]
                simulate.main()
            finally:
                sys.argv = old_argv
            self.assertIsNone(captured["plot"])
            self.assertIsNone(captured["anim"])
            # stats_callback is always provided now: it records the
            # speed/size history for the CSV even without any plot
        finally:
            simulate.simulate_multi = original_multi

    def test_main_elements_enabled(self):
        """--elements is forwarded to the simulation."""
        original_multi = simulate.simulate_multi
        captured = {}

        def mock_multi(*args, **kwargs):
            for k in (
                "elements",
                "element_rate",
                "element_radius",
                "element_boost",
                "element_lifetime",
                "element_max",
            ):
                captured[k] = kwargs.get(k)
            return [[1.0]]

        simulate.simulate_multi = mock_multi
        try:
            old_argv = sys.argv.copy()
            try:
                sys.argv = [
                    "simulate.py",
                    "--config",
                    self.test_cfg,
                    "--elements",
                    "--element-rate",
                    "2",
                    "--element-radius",
                    "1.5",
                    "--element-boost",
                    "-0.1",
                    "--element-lifetime",
                    "7",
                    "--element-max",
                    "9",
                    "--pop-size",
                    "2",
                    "--output",
                    self.test_output,
                ]
                simulate.main()
            finally:
                sys.argv = old_argv
            self.assertIs(captured["elements"], True)
            self.assertEqual(captured["element_rate"], 2.0)
            self.assertEqual(captured["element_radius"], 1.5)
            self.assertEqual(captured["element_boost"], -0.1)
            self.assertEqual(captured["element_lifetime"], 7)
            self.assertEqual(captured["element_max"], 9)
        finally:
            simulate.simulate_multi = original_multi

    def test_main_elements_defaults_from_config(self):
        """Without --elements, the config's [elements] section is honored."""
        cfg = "test_callbacks_elements.toml"
        with open(cfg, "w") as f:
            f.write(
                "[run]\ngenerations = 1\n"
                "[traits]\ninit_means = [1.0]\ninit_stds = [0.1]\n"
                "mutation_stds = [0.1]\n"
                "[elements]\nenabled = true\nrate = 3.0\nmax = 4\n"
            )
        original_multi = simulate.simulate_multi
        captured = {}

        def mock_multi(*args, **kwargs):
            for k in ("elements", "element_rate", "element_max"):
                captured[k] = kwargs.get(k)
            return [[1.0]]

        simulate.simulate_multi = mock_multi
        try:
            old_argv = sys.argv.copy()
            try:
                sys.argv = [
                    "simulate.py",
                    "--config",
                    cfg,
                    "--pop-size",
                    "2",
                    "--output",
                    self.test_output,
                ]
                simulate.main()
            finally:
                sys.argv = old_argv
            self.assertIs(captured["elements"], True)
            self.assertEqual(captured["element_rate"], 3.0)
            self.assertEqual(captured["element_max"], 4)
        finally:
            simulate.simulate_multi = original_multi
            if os.path.exists(cfg):
                os.remove(cfg)

    @patch("matplotlib.pyplot")
    def test_main_animate_renders_elements(self, mock_plt):
        """Test that --animate --elements draws markers and circles."""
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)
        original_multi = simulate.simulate_multi
        captured = {}

        def mock_multi(*args, **kwargs):
            # do not run the real simulation: keep the anim_callback
            # closure fresh (scat/elem_scat still None) so the
            # first-frame drawing path is exercised below
            captured["anim_callback"] = kwargs.get("anim_callback")
            return [[1.0]]

        simulate.simulate_multi = mock_multi
        try:
            old_argv = sys.argv.copy()
            try:
                sys.argv = [
                    "simulate.py",
                    "--config",
                    self.test_cfg,
                    "--animate",
                    "--elements",
                    "--element-rate",
                    "3",
                    "--element-boost",
                    "0.2",
                    "--pop-size",
                    "2",
                    "--output",
                    self.test_output,
                ]
                mock_plt.get_fignums.side_effect = [[1], []]
                simulate.main()
            finally:
                sys.argv = old_argv
            anim_callback = captured["anim_callback"]
            self.assertTrue(callable(anim_callback))
            # first frame: creature scatter + element scatter + legend
            mock_ax.scatter.reset_mock()
            mock_ax.legend.reset_mock()
            mock_ax.add_patch.reset_mock()
            population = [{"trait": 1.0, "pos": [1.0, 1.0]}, {"trait": 2.0, "pos": [2.0, 2.0]}]
            elements = [
                {"pos": [1.5, 1.5], "trait": 0, "radius": 1.0, "boost": 0.2, "lifetime": 5},
                {"pos": [3.0, 3.0], "trait": 0, "radius": 1.0, "boost": -0.1, "lifetime": 5},
            ]
            anim_callback(0, population, elements)
            self.assertEqual(mock_ax.scatter.call_count, 2)
            elem_call = mock_ax.scatter.call_args_list[1]
            self.assertEqual(elem_call.kwargs["marker"], "X")
            # marker size encodes the bonus: 150 + 600 * |boost|
            self.assertEqual(list(elem_call.kwargs["s"]), [270.0, 210.0])
            # creature marker area encodes size; these creatures use the
            # default size 1.0, which normalizes to 0 on its fresh scale
            creature_call = mock_ax.scatter.call_args_list[0]
            self.assertEqual(list(creature_call.kwargs["s"]), [15.0, 15.0])
            self.assertTrue(mock_ax.legend.called)
            self.assertEqual(mock_ax.add_patch.call_count, 2)
            # second frame: the element scatter is updated in place
            anim_callback(1, population, elements)
            self.assertTrue(mock_ax.scatter.return_value.set_sizes.called)
            self.assertEqual(mock_ax.scatter.call_count, 2)
            self.assertEqual(mock_ax.add_patch.call_count, 4)
        finally:
            simulate.simulate_multi = original_multi

    @patch("matplotlib.pyplot")
    def test_main_plot_block_caps_generations(self, mock_plt):
        """--plot-block with an interactive backend honors --generations
        instead of running until the (possibly invisible) window closes."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.plot.return_value = [MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.get_fignums.return_value = [1]  # window "always open"
        original_multi = simulate.simulate_multi
        calls = []
        simulated_generations = []

        def mock_multi(*args, **kwargs):
            calls.append(kwargs)
            # mimic the real loop: keep going while generations is None
            gen, limit = 0, kwargs.get("generations")
            while limit is None or gen < limit:
                gen += 1
            simulated_generations.append(gen)
            return [[1.0] * gen]

        simulate.simulate_multi = mock_multi
        old_argv = sys.argv.copy()
        try:
            sys.argv = [
                "simulate.py",
                "--config",
                self.test_cfg,
                "--live-plot",
                "--plot-block",
                "--generations",
                "3",
                "--pop-size",
                "2",
                "--output",
                self.test_output,
            ]
            simulate.main()
        finally:
            sys.argv = old_argv
            simulate.simulate_multi = original_multi
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["generations"], 3)
        self.assertIsNone(calls[0]["stop_when"])
        self.assertEqual(simulated_generations, [3])

    @patch("matplotlib.pyplot")
    def test_main_plot_block_off_runs_until_window_closes(self, mock_plt):
        """Without --plot-block the existing behavior is unchanged:
        generations=None + stop_when polls get_fignums."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_ax.plot.return_value = [MagicMock()]
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.get_fignums.side_effect = [[1], [1], [1], []]
        original_multi = simulate.simulate_multi
        captured = {}

        def mock_multi(*args, **kwargs):
            captured["generations"] = kwargs.get("generations")
            captured["stop_when"] = kwargs.get("stop_when")
            return [[1.0]]

        simulate.simulate_multi = mock_multi
        old_argv = sys.argv.copy()
        try:
            mock_plt.get_fignums.side_effect = [[1], [1], [1], []]
            sys.argv = [
                "simulate.py",
                "--config",
                self.test_cfg,
                "--live-plot",
                "--generations",
                "99",
                "--pop-size",
                "2",
                "--output",
                self.test_output,
            ]
            simulate.main()
        finally:
            sys.argv = old_argv
            simulate.simulate_multi = original_multi
        self.assertIsNone(captured["generations"])
        self.assertTrue(callable(captured["stop_when"]))


if __name__ == "__main__":
    unittest.main()
