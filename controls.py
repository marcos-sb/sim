"""Player controls for interactive runs.

Pure state, no matplotlib: the live handlers push mouse/keyboard events
in, and the simulation loop pops pending requests and consults the dial
multipliers at the start of every generation.

Selection and mutation dials are MULTIPLIERS on the sim.toml values, so
the config file stays the canonical base. The optimum override set by
the slider panel is absolute — it is a position in trait space.
"""

DEFAULT_SNAPSHOT = "snapshot.toml"

TOOLS = ("food", "toxin", "mutagen", "beacon")
TOOL_KEYS = {"1": "food", "2": "toxin", "3": "mutagen", "4": "beacon"}

MAX_MULT = 8.0
MULT_STEP = 0.25


class Controls:
    """Interactive state shared between the matplotlib handlers, the CLI
    (pause/step/snapshot) and the simulation loop."""

    def __init__(self, num_traits, default_radius=1.0, default_boost=0.05, default_lifetime=200):
        self.num_traits = num_traits
        self.default_radius = default_radius
        self.default_boost = default_boost
        self.default_lifetime = default_lifetime
        self.tool = "food"
        self.target_trait = 0
        self.selection_mult = 1.0
        self.mutation_mult = 1.0
        self.interval_scale = 1.0  # speed keys; scales the pause time
        self.paused = False
        self.steps = 0  # pending single-step requests
        self.erase_mode = False
        self.requests = []
        self.optimum_override = None  # absolute vector, or None
        self.save_requested = False

    # ---- tools -----------------------------------------------------
    def set_tool(self, tool):
        if tool not in TOOLS:
            raise ValueError(f'unknown tool {tool!r} (valid: {", ".join(TOOLS)})')
        self.tool = tool
        self.erase_mode = False

    def toggle_erase(self):
        self.erase_mode = not self.erase_mode

    def cycle_trait(self):
        if self.num_traits > 1:
            self.target_trait = (self.target_trait + 1) % self.num_traits

    # ---- dials ------------------------------------------------------
    def adjust(self, dial, delta):
        """Move a 'selection' or 'mutation' multiplier by delta."""
        attr = f"{dial}_mult"
        value = getattr(self, attr)
        setattr(self, attr, max(0.0, min(MAX_MULT, value + delta)))

    def speed_up(self):
        self.interval_scale = max(0.05, self.interval_scale / 1.5)

    def slow_down(self):
        self.interval_scale = min(5.0, self.interval_scale * 1.5)

    # ---- run control ------------------------------------------------
    def toggle_pause(self):
        self.paused = not self.paused

    def request_step(self):
        self.steps += 1

    # ---- map clicks -------------------------------------------------
    def click(self, pos, radius=None):
        """Queue a map action at pos (drag radius overrides the default)."""
        if self.erase_mode:
            self.requests.append({"op": "erase", "pos": [float(pos[0]), float(pos[1])]})
            return
        req = {
            "op": "spawn",
            "kind": self.tool,
            "pos": [float(pos[0]), float(pos[1])],
            "radius": float(radius) if radius else self.default_radius,
            "trait": self.target_trait,
            "lifetime": self.default_lifetime,
        }
        if self.tool == "beacon":
            req["boost"] = self.default_boost
            req["traits"] = None  # captured from nearby creatures in the model
        elif self.tool == "mutagen":
            req["boost"] = self.default_boost  # |boost| = jitter magnitude
        elif self.tool == "toxin":
            req["boost"] = -self.default_boost
        else:
            req["boost"] = self.default_boost
        self.requests.append(req)

    def pop_requests(self):
        reqs = self.requests
        self.requests = []
        return reqs

    # ---- save / optimum ---------------------------------------------
    def request_save(self):
        self.requests.append({"op": "save"})

    def take_save(self):
        if self.save_requested:
            self.save_requested = False
            return True
        return False

    def set_optimum_override(self, vector):
        self.optimum_override = [float(v) for v in vector]

    def effective_optimum(self, fallback):
        return list(self.optimum_override) if self.optimum_override is not None else fallback

    # ---- HUD ----------------------------------------------------------
    def hud_line(self):
        parts = [f"tool {self.tool}"]
        if self.num_traits > 1:
            parts.append(f"trait {self.target_trait}")
        if self.erase_mode:
            parts.append("erase")
        parts.append(f"sel x{self.selection_mult:g}")
        parts.append(f"mut x{self.mutation_mult:g}")
        if self.paused:
            parts.append("PAUSED")
        return " · ".join(parts)


def snapshot_text(
    averages,
    stds,
    base_mutation_stds,
    base_strength,
    properties,
    optimum,
    pop_size,
    mut_mult,
    sel_mult,
):
    """TOML text capturing an evolved state as a runnable sim.toml.

    init_means/stds come from the live population stats; the mutation /
    selection dial multipliers are rolled into the written rates; the
    effective optimum (override or config) is written when set.
    """
    lines = [
        ("# Generated by the interactive run (key w): population snapshot + current dials"),
        "",
        "[run]",
        f"pop_size = {pop_size}",
        "generations = 300",
        "",
        "[traits]",
    ]
    for name, values in (
        ("init_means", averages),
        ("init_stds", stds),
        ("mutation_stds", [v * mut_mult for v in base_mutation_stds]),
    ):
        lines.append(f'{name} = [{", ".join(repr(float(v)) for v in values)}]')
    strength = base_strength * sel_mult
    lines.append(f"selection_strength = {float(strength)!r}")
    if optimum is not None:
        lines.append(f'optimum = [{", ".join(repr(float(v)) for v in optimum)}]')
    lines.append("properties = [" + ", ".join(repr(p) for p in properties) + "]")
    return "\n".join(lines) + "\n"
