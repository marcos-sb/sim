"""Tests for controls.py: pure interactive state + snapshot writer."""

import os
import tempfile
import unittest

import config as config_mod
from controls import Controls, snapshot_text


class TestControls(unittest.TestCase):
    def setUp(self):
        self.c = Controls(3, default_radius=1.0, default_boost=0.05, default_lifetime=200)

    # ---- tools ----------------------------------------------------
    def test_set_tool_and_erase_reset(self):
        self.c.set_tool("beacon")
        self.c.toggle_erase()
        self.c.set_tool("food")  # selecting a tool leaves erase mode
        self.assertEqual(self.c.tool, "food")
        self.assertFalse(self.c.erase_mode)

    def test_unknown_tool_rejected(self):
        with self.assertRaises(ValueError):
            self.c.set_tool("laser")

    def test_cycle_trait_wraps(self):
        self.c.set_tool("toxin")  # tools do not matter here
        self.c.cycle_trait()
        self.assertEqual(self.c.target_trait, 1)
        self.c.cycle_trait()
        self.c.cycle_trait()
        self.assertEqual(self.c.target_trait, 0)
        single = Controls(1)
        single.cycle_trait()
        self.assertEqual(single.target_trait, 0)

    # ---- clicks ---------------------------------------------------
    def test_food_click_defaults(self):
        self.c.set_tool("food")
        self.c.click((3.0, 4.0))
        (req,) = self.c.pop_requests()
        self.assertEqual(req["op"], "spawn")
        self.assertEqual(req["kind"], "food")
        self.assertEqual(req["pos"], [3.0, 4.0])
        self.assertEqual(req["radius"], 1.0)
        self.assertEqual(req["boost"], 0.05)
        self.assertEqual(req["trait"], 0)
        self.assertEqual(req["lifetime"], 200)

    def test_toxin_click_negative_boost(self):
        self.c.set_tool("toxin")
        self.c.click((0.0, 0.0), radius=2.5)
        (req,) = self.c.pop_requests()
        self.assertEqual(req["kind"], "toxin")
        self.assertEqual(req["boost"], -0.05)
        self.assertEqual(req["radius"], 2.5)

    def test_beacon_click_defers_capture(self):
        self.c.set_tool("beacon")
        self.c.click((1.0, 1.0))
        (req,) = self.c.pop_requests()
        self.assertEqual(req["kind"], "beacon")
        self.assertIsNone(req["traits"])  # model captures the phenotype

    def test_erase_click(self):
        self.c.toggle_erase()
        self.c.click((7.0, 7.0))
        (req,) = self.c.pop_requests()
        self.assertEqual(req, {"op": "erase", "pos": [7.0, 7.0]})

    def test_pop_requests_clears(self):
        self.c.click((0.0, 0.0))
        self.c.click((1.0, 1.0))
        self.assertEqual(len(self.c.pop_requests()), 2)
        self.assertEqual(self.c.pop_requests(), [])

    # ---- dials ----------------------------------------------------
    def test_dial_multipliers_clamped(self):
        for _ in range(100):
            self.c.adjust("selection", 10.0)
        self.assertEqual(self.c.selection_mult, 8.0)
        for _ in range(100):
            self.c.adjust("mutation", -10.0)
        self.assertEqual(self.c.mutation_mult, 0.0)

    def test_speed_keys_scale_interval(self):
        self.assertEqual(self.c.interval_scale, 1.0)
        self.c.speed_up()  # smaller pause time = faster
        self.assertLess(self.c.interval_scale, 1.0)
        self.c.slow_down()
        self.c.slow_down()
        self.assertGreater(self.c.interval_scale, 1.0)
        for _ in range(20):
            self.c.slow_down()
        self.assertLessEqual(self.c.interval_scale, 5.0)
        for _ in range(60):
            self.c.speed_up()
        self.assertGreaterEqual(self.c.interval_scale, 0.05)

    # ---- run control ------------------------------------------------
    def test_pause_and_steps(self):
        self.c.toggle_pause()
        self.assertTrue(self.c.paused)
        self.c.request_step()
        self.assertEqual(self.c.steps, 1)
        self.c.steps -= 1  # the model/CLI consumes pending steps
        self.c.toggle_pause()
        self.assertFalse(self.c.paused)

    # ---- optimum / HUD ----------------------------------------------
    def test_optimum_override(self):
        self.c.set_optimum_override((0.5, 1.5, -2.0))
        self.assertEqual(self.c.optimum_override, [0.5, 1.5, -2.0])
        self.assertEqual(self.c.effective_optimum([9.0, 9.0, 9.0]), [0.5, 1.5, -2.0])
        single = Controls(1)
        self.assertIsNone(single.optimum_override)
        self.assertEqual(single.effective_optimum([3.0]), [3.0])

    def test_hud_mentions_state(self):
        self.c.set_tool("toxin")
        self.c.adjust("selection", 0.5)
        self.c.toggle_pause()
        self.c.toggle_erase()
        hud = self.c.hud_line()
        self.assertIn("toxin", hud)
        self.assertIn("erase", hud)
        self.assertIn("x1.5", hud)
        self.assertIn("PAUSED", hud)


class TestSnapshot(unittest.TestCase):
    def _snapshot(self, optimum=(1.0, 2.0)):
        return snapshot_text(
            averages=(1.5, 2.5),
            stds=(0.2, 0.3),
            base_mutation_stds=(0.05, 0.05),
            base_strength=1.0,
            properties=("none", "none"),
            optimum=optimum,
            pop_size=50,
            mut_mult=2.0,
            sel_mult=0.5,
        )

    def test_snapshot_is_valid_toml_and_roundtrips(self):
        text = self._snapshot()
        path = os.path.join(tempfile.gettempdir(), "test_snapshot.toml")
        try:
            with open(path, "w") as fh:
                fh.write(text)
            # the saved snapshot must load back as a runnable config
            cfg = config_mod.load_config(path)
        finally:
            if os.path.exists(path):
                os.remove(path)
        self.assertEqual(cfg["pop_size"], 50)
        self.assertEqual(cfg["traits"]["init_means"], [1.5, 2.5])
        self.assertEqual(cfg["traits"]["init_stds"], [0.2, 0.3])
        self.assertEqual(cfg["traits"]["mutation_stds"], [0.1, 0.1])  # x2 dial rolled in
        self.assertEqual(cfg["traits"]["selection_strength"], 0.5)  # x0.5 dial rolled in
        self.assertEqual(cfg["traits"]["optimum"], [1.0, 2.0])

    def test_snapshot_without_optimum(self):
        text = self._snapshot(optimum=None)
        self.assertNotIn("optimum", text)


if __name__ == "__main__":
    unittest.main()
