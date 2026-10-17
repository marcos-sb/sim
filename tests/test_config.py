"""Tests for config.py (sim.toml validation) and the trait-property
behavior hooks in model.py."""

import os
import random
import sys
import tempfile
import unittest

import numpy as np

import model
from config import ConfigError, load_config, template


def _write_cfg(text):
    fd, path = tempfile.mkstemp(suffix=".toml")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


class TestConfigValidation(unittest.TestCase):
    def test_template_is_valid_and_loadable(self):
        path = _write_cfg(template())
        try:
            cfg = load_config(path)
            self.assertEqual(cfg["pop_size"], 100)
            self.assertEqual(len(cfg["traits"]["init_means"]), 3)
            self.assertEqual(cfg["traits"]["properties"], ["move_speed", "steer", "repel"])
        finally:
            os.remove(path)

    def test_trait_count_and_defaults_derive_from_init_means(self):
        path = _write_cfg("[traits]\ninit_means = [1.0]\n")
        try:
            cfg = load_config(path)
            self.assertEqual(len(cfg["traits"]["init_stds"]), 1)
            self.assertEqual(cfg["traits"]["mutation_stds"], [0.05])
            self.assertIsNone(cfg["traits"]["optimum"])
            self.assertEqual(cfg["traits"]["selection_strength"], 0.0)
            self.assertEqual(cfg["traits"]["properties"], ["none"])
        finally:
            os.remove(path)

    def test_mutation_std_shorthand(self):
        path = _write_cfg("[traits]\ninit_means = [0.0, 1.0]\n" "mutation_std = 0.3\n")
        try:
            cfg = load_config(path)
            self.assertEqual(cfg["traits"]["mutation_stds"], [0.3, 0.3])
        finally:
            os.remove(path)

    def test_mutation_std_and_stds_conflict(self):
        path = _write_cfg(
            "[traits]\ninit_means = [0.0]\n" "mutation_std = 0.1\nmutation_stds = [0.1]\n"
        )
        try:
            with self.assertRaises(ConfigError):
                load_config(path)
        finally:
            os.remove(path)

    def test_strength_without_optimum_rejected(self):
        path = _write_cfg("[traits]\ninit_means = [0.0, 1.0]\n" "selection_strength = 0.5\n")
        try:
            with self.assertRaises(ConfigError) as ctx:
                load_config(path)
            self.assertIn("optimum", str(ctx.exception))
        finally:
            os.remove(path)

    def test_strength_zero_allows_missing_optimum(self):
        path = _write_cfg("[traits]\ninit_means = [0.0]\n" "selection_strength = 0.0\n")
        try:
            cfg = load_config(path)
            self.assertIsNone(cfg["traits"]["optimum"])
        finally:
            os.remove(path)

    def test_optimum_length_must_match(self):
        path = _write_cfg("[traits]\ninit_means = [0.0, 1.0]\n" "optimum = [0.0]\n")
        try:
            with self.assertRaises(ConfigError):
                load_config(path)
        finally:
            os.remove(path)

    def test_unknown_property_rejected(self):
        path = _write_cfg("[traits]\ninit_means = [0.0]\n" 'properties = ["telepathy"]\n')
        try:
            with self.assertRaises(ConfigError) as ctx:
                load_config(path)
            self.assertIn("telepathy", str(ctx.exception))
        finally:
            os.remove(path)

    def test_properties_length_must_match_trait_count(self):
        path = _write_cfg("[traits]\ninit_means = [0.0, 1.0]\n" 'properties = ["steer"]\n')
        try:
            with self.assertRaises(ConfigError):
                load_config(path)
        finally:
            os.remove(path)

    def test_non_string_property_rejected(self):
        path = _write_cfg("[traits]\ninit_means = [0.0]\n" "properties = [3]\n")
        try:
            with self.assertRaises(ConfigError):
                load_config(path)
        finally:
            os.remove(path)

    def test_unknown_section_rejected(self):
        path = _write_cfg("[weird]\nx = 1\n")
        try:
            with self.assertRaises(ConfigError) as ctx:
                load_config(path)
            self.assertIn("weird", str(ctx.exception))
        finally:
            os.remove(path)

    def test_unknown_key_rejected(self):
        path = _write_cfg("[run]\npopsize = 5\n")
        try:
            with self.assertRaises(ConfigError) as ctx:
                load_config(path)
            self.assertIn("popsize", str(ctx.exception))
        finally:
            os.remove(path)

    def test_unknown_traits_key_rejected(self):
        path = _write_cfg("[traits]\ninit_means = [0.0]\nfoo = 1\n")
        try:
            with self.assertRaises(ConfigError):
                load_config(path)
        finally:
            os.remove(path)

    def test_invalid_types_rejected(self):
        path = _write_cfg("[run]\npop_size = 0.5\n")
        try:
            with self.assertRaises(ConfigError):
                load_config(path)
        finally:
            os.remove(path)

    def test_position_bounds_sanity(self):
        path = _write_cfg("[space]\nposition_min = 10.0\n")
        try:
            with self.assertRaises(ConfigError) as ctx:
                load_config(path)
            self.assertIn("position_max", str(ctx.exception))
        finally:
            os.remove(path)

    def test_missing_file_named(self):
        with self.assertRaises(ConfigError) as ctx:
            load_config("definitely_not_here.toml")
        self.assertIn("definitely_not_here", str(ctx.exception))

    def test_invalid_toml_named(self):
        path = _write_cfg("this is [not toml\n")
        try:
            with self.assertRaises(ConfigError) as ctx:
                load_config(path)
            self.assertIn("invalid TOML", str(ctx.exception))
        finally:
            os.remove(path)

    def test_lists_length_must_match_trait_count(self):
        for key in ("init_stds", "mutation_stds"):
            path = _write_cfg(f"[traits]\ninit_means = [0.0, 1.0]\n" f"{key} = [0.1]\n")
            try:
                with self.assertRaises(ConfigError):
                    load_config(path)
            finally:
                os.remove(path)


class TestCliPrecedence(unittest.TestCase):
    """CLI flag > sim.toml value > built-in default."""

    def setUp(self):
        self.test_output = "test_precedence.csv"
        self.test_cfg = "test_precedence.toml"
        with open(self.test_cfg, "w") as f:
            f.write(
                "[run]\npop_size = 7\ngenerations = 2\n"
                "[traits]\ninit_means = [1.0, 2.0]\ninit_stds = [0.1, 0.1]\n"
                "mutation_stds = [0.05, 0.05]\n"
            )

    def tearDown(self):
        for path in (self.test_output, self.test_cfg):
            if os.path.exists(path):
                os.remove(path)

    def _capture(self, argv):
        import simulate

        original_multi = simulate.simulate_multi
        captured = {}

        def mock_multi(**kwargs):
            captured.update(kwargs)
            return [[1.0], [2.0]]

        simulate.simulate_multi = mock_multi
        old_argv = sys.argv.copy()
        try:
            sys.argv = ["simulate.py"] + argv
            simulate.main()
        finally:
            sys.argv = old_argv
            simulate.simulate_multi = original_multi
        return captured

    def test_config_values_used_when_flags_absent(self):
        captured = self._capture(["--config", self.test_cfg, "--output", self.test_output])
        self.assertEqual(captured["pop_size"], 7)
        self.assertEqual(captured["generations"], 2)
        self.assertEqual(captured["num_traits"], 2)
        self.assertEqual(captured["init_means"], [1.0, 2.0])
        self.assertEqual(captured["mutation_stds"], [0.05, 0.05])
        self.assertEqual(captured["trait_properties"], ["none", "none"])

    def test_cli_flags_override_config(self):
        captured = self._capture(
            [
                "--config",
                self.test_cfg,
                "--pop-size",
                "11",
                "--generations",
                "9",
                "--output",
                self.test_output,
            ]
        )
        self.assertEqual(captured["pop_size"], 11)
        self.assertEqual(captured["generations"], 9)

    def test_output_default_without_flag(self):
        captured = self._capture(["--config", self.test_cfg])
        self.assertEqual(captured["pop_size"], 7)
        # default output written; clean it up
        if os.path.exists("traits.csv"):
            os.remove("traits.csv")


class TestModelPropertyHooks(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        np.random.seed(self.seed)
        random.seed(self.seed)

    def test_survival_decay_shrinks_trait(self):
        """A survival_decay trait shrinks every generation."""
        result = model.simulate_multi(
            pop_size=10,
            generations=20,
            num_traits=1,
            init_means=[5.0],
            init_stds=[0.0],
            mutation_stds=[0.0],
            position_std=0.0,
            position_min=0.0,
            position_max=10.0,
            trait_properties=["survival_decay"],
        )
        self.assertLess(result[0][19], 5.0)  # decayed from 5
        self.assertGreater(result[0][19], 0.0)
        # strictly decreasing after generation 0
        self.assertTrue(all(b < a for a, b in zip(result[0], result[0][1:])))

    def test_no_decay_without_property(self):
        """Same params but properties=none: values do not shrink."""
        result = model.simulate_multi(
            pop_size=10,
            generations=20,
            num_traits=1,
            init_means=[5.0],
            init_stds=[0.0],
            mutation_stds=[0.0],
            position_std=0.0,
            position_min=0.0,
            position_max=10.0,
        )
        # stays at the initial mean (mutation 0, no selection)
        self.assertAlmostEqual(result[0][19], 5.0, places=6)

    def test_element_resist_blocks_boost(self):
        """element_resist=1 creatures receive no element boost."""
        common = {
            "pop_size": 5,
            "generations": 1,
            "num_traits": 1,
            "init_means": [1.0],
            "init_stds": [0.0],
            "mutation_stds": [0.0],
            "position_std": 0.0,
            "position_min": 0.0,
            "position_max": 10.0,
            "elements": True,
            "element_rate": 100.0,
            "element_radius": 10.0,
            "element_boost": 0.5,
        }
        resist = model.simulate_multi(trait_properties=["element_resist"], **common)
        bare = model.simulate_multi(trait_properties=None, **common)
        # resist value 1.0 -> immune: the trait stays at its initial 1.0
        self.assertAlmostEqual(resist[0][-1], 1.0, places=6)
        # without the property, creatures feel the full boost: they lift
        # above the initial 1.0
        self.assertGreater(bare[0][-1], 1.2)

    def test_repro_boost_biases_inheritance(self):
        """Larger-magnitude repro_boost parent wins the offspring trait."""
        # parent values 5.0 and 0.0; with repro_boost, offspring trait
        # should stay near 5.0 instead of averaging toward 0
        result = model.simulate_multi(
            pop_size=20,
            generations=10,
            num_traits=1,
            init_means=[2.5],
            init_stds=[2.5],
            mutation_stds=[0.0],
            position_std=0.0,
            position_min=0.0,
            position_max=10.0,
            trait_properties=["repro_boost"],
        )
        self.assertGreater(result[0][9], 1.0)
        # without the property, values average sharply downward
        plain = model.simulate_multi(
            pop_size=20,
            generations=10,
            num_traits=1,
            init_means=[2.5],
            init_stds=[2.5],
            mutation_stds=[0.0],
            position_std=0.0,
            position_min=0.0,
            position_max=10.0,
        )
        # biased parent choice keeps magnitudes higher than plain
        # random inheritance
        self.assertGreater(result[0][9], plain[0][9])

    def test_move_speed_scales_displacement(self):
        """move_speed multiplies how far a creature moves."""

        def run(move_v):
            pop = [{"traits": [move_v], "speed": 1.0, "size": 1.0, "pos": [5.0, 5.0]}]
            # a stationary neighbour as the attraction anchor
            pop.append({"traits": [move_v], "speed": 0.0, "size": 1.0, "pos": [5.0, 6.0]})
            model._move_toward_similar(
                pop,
                similarity_scale=0.5,
                move_step=1.0,
                position_std=0.0,
                position_min=0.0,
                position_max=10.0,
                trait_properties=["move_speed"],
            )
            return pop[0]["pos"][1]

        # displacement = move_step * attraction(=1) * dist(1) * speed
        # (1.0) * move_speed value; deterministic with no jitter
        self.assertAlmostEqual(run(3.0), 8.0, places=6)
        self.assertAlmostEqual(run(0.5), 5.5, places=6)
        self.assertAlmostEqual(run(0.0), 5.0, places=6)

    def test_steer_moves_toward_and_away(self):
        """steer > 0 approaches the anchor; steer < 0 flees it."""

        def run(steer_v):
            pop = [
                {"traits": [steer_v], "speed": 1.0, "size": 1.0, "pos": [0.0, 0.0]},
                {"traits": [steer_v], "speed": 1.0, "size": 1.0, "pos": [0.0, 0.0]},
                {"traits": [steer_v], "speed": 0.0, "size": 1.0, "pos": [1.0, 0.0]},
            ]
            model._move_toward_similar(
                pop,
                similarity_scale=0.5,
                move_step=1.0,
                position_std=0.0,
                position_min=0.0,
                position_max=10.0,
                trait_properties=["steer"],
            )
            return pop[0]["pos"][0]

        # attraction alone (steer 0) moves it halfway to the centroid
        self.assertAlmostEqual(run(0.0), 0.5, places=6)
        self.assertGreater(run(2.0), 0.5)
        self.assertLess(run(-2.0), 0.5)

    def test_repel_enlarges_personal_space(self):
        """A higher repel value widens the space a dissimilar creature keeps."""

        def run(repel_v):
            # trait 0 is the repel property; trait 1 makes B dissimilar.
            # B's own repel value is 0, so the pair radius is A's alone.
            pop = [
                {"traits": [repel_v, 0.0], "speed": 1.0, "size": 1.0, "pos": [5.0, 5.0]},
                {"traits": [0.0, 99.0], "speed": 1.0, "size": 1.0, "pos": [5.0, 5.0 - gap]},
            ]
            model._move_toward_similar(
                pop,
                similarity_scale=0.5,
                move_step=1.0,
                position_std=0.0,
                position_min=0.0,
                position_max=10.0,
                trait_properties=["repel", "none"],
            )
            return pop[0]["pos"][1]

        gap = 0.3
        pushed = run(3.0)  # radius 3 x baseline: pair is inside it
        baseline = run(0.0)  # radius 0 (and B's is 0): no repulsion
        self.assertGreater(pushed, 5.0)
        self.assertAlmostEqual(baseline, 5.0, places=12)

    def test_cluster_bias_scales_attraction(self):
        """cluster_bias 0 kills attraction; 1 keeps it (with speed 1)."""

        def run(bias_v):
            pop = [
                {"traits": [bias_v], "speed": 1.0, "size": 1.0, "pos": [0.0, 0.0]},
                {"traits": [bias_v], "speed": 1.0, "size": 1.0, "pos": [0.0, 0.0]},
            ]
            model._move_toward_similar(
                pop + [{"traits": [bias_v], "speed": 1.0, "size": 1.0, "pos": [2.0, 0.0]}],
                similarity_scale=0.5,
                move_step=0.5,
                position_std=0.0,
                position_min=0.0,
                position_max=10.0,
                trait_properties=["cluster_bias"],
            )
            return pop[0]["pos"][0]

        self.assertGreater(run(1.0), 0.0)
        self.assertAlmostEqual(run(0.0), 0.0, places=12)

    def test_life_expect_keeps_population_size(self):
        """Mortality replaces the dead; population size stays constant."""
        # tiny life values -> high death chance, many replacements
        result = model.simulate_multi(
            pop_size=15,
            generations=30,
            num_traits=1,
            init_means=[0.1],
            init_stds=[0.0],
            mutation_stds=[0.05],
            position_std=0.0,
            position_min=0.0,
            position_max=10.0,
            trait_properties=["life_expect"],
        )
        self.assertEqual(len(result[0]), 30)  # one average row per generation

    def test_legacy_positional_callers_still_work(self):
        """_move_toward_similar called positionally without properties."""
        pop = [{"traits": [1.0], "speed": 1.0, "size": 1.0, "pos": [1.0, 1.0]}]
        model._move_toward_similar(pop, 0.5, 0.1, 0.0, 0.0, 10.0)
        self.assertEqual(pop[0]["pos"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
