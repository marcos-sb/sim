"""CLI for the creature-trait evolution simulation.

Config-first: `sim.toml` is the canonical home for everything evolvable
(see `config.py` for the schema). Remaining CLI flags only override or
toggle. Thin wrapper: model lives in `model.py`, live matplotlib
handlers in `live_viz.py`. This module re-exports the model API so
existing `import simulate` usage (including tests and
`@patch('simulate.np')`) keeps working.
"""

import argparse
import random
import sys

# numpy is optional at import time; --animate and --seed check for it
try:
    import numpy as np
except ImportError:
    np = None

import config as config_mod
from config import ConfigError, template
from live_viz import fix_axis_to_position_range, make_anim_handler, make_plot_handler
from model import (
    _format_stats,
    _print_trait_summary,
    _write_csv,
    simulate,
    simulate_multi,
)

__all__ = [
    "_format_stats",
    "_print_trait_summary",
    "_write_csv",
    "main",
    "np",
    "simulate",
    "simulate_multi",
]


def main():
    parser = argparse.ArgumentParser(
        description="Simulate evolution of creature traits.",
        epilog="Everything evolvable lives in sim.toml (see "
        "--config-template); CLI flags are overrides only.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the TOML configuration file "
        "(e.g. sim.toml). Required for a run; "
        "see --config-template.",
    )
    parser.add_argument(
        "--config-template",
        action="store_true",
        help="Print a fully-commented starter sim.toml and exit",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Override [run].generations (ignored when a plot window is shown; the simulation then runs until the window is closed)",
    )
    parser.add_argument("--pop-size", type=int, default=None, help="Override [run].pop_size")
    parser.add_argument(
        "--output", type=str, default=None, help="Override the output CSV path (default traits.csv)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (seeds both random and numpy)",
    )
    parser.add_argument(
        "--live-plot", action="store_true", help="Show trait evolution in real time (line chart)"
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Show real-time animation of positions colored by traits",
    )
    parser.add_argument(
        "--plot-block",
        action="store_true",
        help="With --live-plot/--animate: cap the run at "
        "--generations instead of running until the "
        "plot window closes. Use when no visible "
        "window is available (agents, scripts, SSH).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.05,
        help="Seconds between generations when a plot is shown (animation speed)",
    )
    parser.add_argument(
        "--elements", action="store_true", help="Enable map elements (overrides [elements].enabled)"
    )
    parser.add_argument("--element-rate", type=float, default=None, help="Override [elements].rate")
    parser.add_argument(
        "--element-radius", type=float, default=None, help="Override [elements].radius"
    )
    parser.add_argument(
        "--element-boost", type=float, default=None, help="Override [elements].boost"
    )
    parser.add_argument(
        "--element-lifetime", type=int, default=None, help="Override [elements].lifetime"
    )
    parser.add_argument("--element-max", type=int, default=None, help="Override [elements].max")
    args = parser.parse_args()

    if args.config_template:
        print(template())
        return

    # ---- resolve configuration: file -> CLI overrides -----------------
    try:
        if args.config is not None:
            cfg = config_mod.load_config(args.config)
        else:
            sys.exit(
                "Error: no config file given. Create one with "
                "--config-template, then run with --config sim.toml."
            )
    except ConfigError as e:
        sys.exit(f"Error: {e}")

    if args.generations is not None:
        if args.generations < 1:
            parser.error("--generations must be a positive integer")
        cfg["generations"] = args.generations
    if args.pop_size is not None:
        if args.pop_size < 1:
            parser.error("--pop-size must be a positive integer")
        cfg["pop_size"] = args.pop_size
    if args.output is not None:
        cfg["output"] = args.output
    if args.elements:
        cfg["enabled"] = True
    for flag, key in (
        ("element_rate", "rate"),
        ("element_radius", "radius"),
        ("element_boost", "boost"),
        ("element_lifetime", "lifetime"),
        ("element_max", "max"),
    ):
        override = getattr(args, flag)
        if override is not None:
            cfg[key] = override

    if args.seed is not None:
        random.seed(args.seed)
        if np is not None:
            np.random.seed(args.seed)

    traits = cfg.get("traits")
    if traits is None:
        sys.exit(
            "Error: the config file has no [traits] section. " "Add one (see --config-template)."
        )
    num_traits = len(traits["init_means"])
    properties = traits["properties"]

    plt_module = None
    show_plot = False
    plot_callback = None
    stats_texts = []  # text boxes showing live stats, one per plot window

    # Prepare live plot (line chart) if requested
    if args.live_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            sys.exit(
                "Error: matplotlib is required for --live-plot. Install it with 'pip install matplotlib'."
            )
        plt_module = plt
        show_plot = True
        plt.ion()
        _, ax = plt.subplots(figsize=(10, 6))
        plot_callback, text = make_plot_handler(ax, plt, args.interval)
        stats_texts.append(text)
    # Prepare animation if requested
    if args.animate:
        if np is None:
            sys.exit(
                "Error: numpy is required for --animate (comes with matplotlib). Install matplotlib with 'pip install matplotlib'."
            )
        import matplotlib

        # Try to switch to an interactive backend if we are using the
        # default Agg backend
        if matplotlib.get_backend() == "Agg":
            for candidate in ("TkAgg", "Qt5Agg"):
                try:
                    matplotlib.use(candidate)
                    break
                except ImportError:
                    continue
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
        except ImportError:
            sys.exit(
                "Error: matplotlib is required for --animate. Install it with 'pip install matplotlib'."
            )
        # If still non-interactive, we cannot show animation; warn and disable
        if matplotlib.get_backend() == "Agg":
            sys.stderr.write(
                "Warning: Animation will not be visible because no interactive matplotlib backend is available. Consider installing TkAgg or Qt5Agg backend. Disabling animation."
            )
            anim_callback = None
        else:
            plt_module = plt
            show_plot = True
            plt.ion()
            _, ax = plt.subplots(figsize=(8, 8))
            # Positions are clamped to [position_min, position_max] in the
            # simulation, so fix the axis once: a single stray creature would
            # otherwise stretch the whole view
            fix_axis_to_position_range(ax, cfg["position_min"], cfg["position_max"])
            anim_callback, text = make_anim_handler(
                ax, plt, np, cm, cfg["position_min"], cfg["position_max"], args.interval
            )
            stats_texts.append(text)
    else:
        anim_callback = None
    stats_cb = None
    if stats_texts:

        def stats_callback(gen, stats):
            line = _format_stats(gen, stats)
            for text in stats_texts:
                text.set_text(line)

        stats_cb = stats_callback

    # With a visible plot window, keep simulating until the user closes it;
    # otherwise cap the run at the requested generation count
    run_generations = cfg["generations"]
    stop_when = None
    if show_plot:
        import matplotlib
        import matplotlib.backends

        non_interactive = matplotlib.backends.backend_registry.list_builtin(
            matplotlib.backends.BackendFilter.NON_INTERACTIVE
        )
        backend = matplotlib.get_backend()
        # Only a known non-interactive backend (e.g. Agg) can never
        # close a window; everything else runs until the window is
        # closed. get_backend() can also return the unresolved
        # auto-backend sentinel (not a string) when pyplot is mocked
        # in tests, which counts as interactive here.
        if (
            not (isinstance(backend, str) and backend.lower() in non_interactive)
            and not args.plot_block
        ):
            # a real window exists and --plot-block was not given: run
            # until the user closes the window
            run_generations = None

            def stop_when():
                return not plt_module.get_fignums()

    # Record real speed/size history through the stats callback so the
    # CSV carries measured values (simulate_multi returns trait averages only).
    speed_hist, size_hist = [], []

    def record_stats(gen, stats):
        speed_hist.append(stats["speed_avg"])
        size_hist.append(stats["size_avg"])
        if stats_cb is not None:
            stats_cb(gen, stats)

    averages_per_trait = simulate_multi(
        pop_size=cfg["pop_size"],
        generations=run_generations,
        num_traits=num_traits,
        init_means=traits["init_means"],
        init_stds=traits["init_stds"],
        mutation_stds=traits["mutation_stds"],
        optimum=traits["optimum"],
        selection_strength=traits["selection_strength"],
        position_std=cfg["position_std"],
        position_min=cfg["position_min"],
        position_max=cfg["position_max"],
        similarity_scale=cfg["similarity_scale"],
        move_step=cfg["move_step"],
        plot_callback=plot_callback,
        anim_callback=anim_callback,
        stop_when=stop_when,
        stats_callback=record_stats,
        elements=cfg["enabled"],
        element_rate=cfg["rate"],
        element_radius=cfg["radius"],
        element_boost=cfg["boost"],
        element_lifetime=cfg["lifetime"],
        element_max=cfg["max"],
        speed_mean=cfg["speed_mean"],
        speed_std=cfg["speed_std"],
        speed_mut=cfg["speed_mutation"],
        size_mean=cfg["size_mean"],
        size_std=cfg["size_std"],
        size_mut=cfg["size_mutation"],
        trait_properties=properties,
    )

    # Write CSV: one column per trait plus speed/size averages
    header = (
        ["generation"] + [f"trait{i}_avg" for i in range(num_traits)] + ["speed_avg", "size_avg"]
    )
    # record_stats fires once per generation alongside the averages, so
    # the histories should align; min() keeps a mocked/shorter run safe.
    num_rows = min(len(averages_per_trait[0]), len(speed_hist), len(size_hist))
    rows = (
        [gen]
        + [averages_per_trait[i][gen] for i in range(num_traits)]
        + [speed_hist[gen], size_hist[gen]]
        for gen in range(num_rows)
    )
    _write_csv(cfg["output"], header, rows)
    print(f'Simulation complete. Results written to {cfg["output"]}')
    if averages_per_trait[0]:
        _print_trait_summary(averages_per_trait)

    # After simulation, if live plot or animation was shown, keep window open until user closes
    if show_plot:
        plt_module.ioff()
        plt_module.show()


if __name__ == "__main__":
    main()
