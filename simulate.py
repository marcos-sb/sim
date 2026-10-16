"""CLI for the creature-trait evolution simulation.

Thin wrapper: model lives in `model.py`, live matplotlib handlers in
`live_viz.py`. This module re-exports the model API so existing
`import simulate` / `simulate.simulate` / `simulate._*_` usage (including
tests and `@patch('simulate.np')`) keeps working.
"""

import argparse
import random
import sys

# Try to import numpy for color mapping; will be used only if animation requested
try:
    import numpy as np
except Exception:
    np = None  # will check later if needed

from live_viz import fix_axis_to_position_range, make_anim_handler, make_plot_handler
from model import (
    SPEED_MAX,
    SPEED_MIN,
    SIZE_MAX,
    SIZE_MIN,
    _clamp,
    _color_norm,
    _fire_callbacks,
    _format_stats,
    _inherit_body,
    _inherit_ranged,
    _mean_var,
    _move_toward_similar,
    _nearest_neighbour,
    _offspring_pos,
    _population_stats,
    _print_trait_summary,
    _random_body,
    _random_pos,
    _trait_columns,
    _trait_distance,
    _update_elements,
    _widen_scale,
    _write_csv,
    simulate,
    simulate_multi,
)

__all__ = [
    "SPEED_MIN", "SPEED_MAX", "SIZE_MIN", "SIZE_MAX",
    "_clamp", "_mean_var", "_trait_columns", "_trait_distance",
    "_random_pos", "_random_body", "_inherit_ranged", "_inherit_body",
    "_offspring_pos", "_fire_callbacks", "_nearest_neighbour",
    "_move_toward_similar", "_widen_scale", "_color_norm",
    "_population_stats", "_format_stats", "_update_elements",
    "_write_csv", "_print_trait_summary",
    "simulate", "simulate_multi", "main", "np",
]


def main():
    parser = argparse.ArgumentParser(description='Simulate evolution of creature traits.')
    # Backward compatibility arguments
    parser.add_argument('--generations', type=int, default=100, help='Number of generations (ignored when a plot window is shown; the simulation then runs until the window is closed)')
    parser.add_argument('--pop-size', type=int, default=100, help='Population size')
    parser.add_argument('--mutation-std', type=float, default=0.05, help='Mutation standard deviation (single trait)')
    parser.add_argument('--initial-mean', type=float, default=1.0, help='Initial mean trait value (single trait)')
    parser.add_argument('--initial-std', type=float, default=0.1, help='Initial standard deviation (single trait)')
    parser.add_argument('--speed-mean', type=float, default=1.0, help='Initial mean speed')
    parser.add_argument('--speed-std', type=float, default=0.2, help='Initial standard deviation of speed')
    parser.add_argument('--speed-mut', type=float, default=None, help='Mutation standard deviation for speed (defaults to the trait mutation std)')
    parser.add_argument('--size-mean', type=float, default=1.0, help='Initial mean size')
    parser.add_argument('--size-std', type=float, default=0.2, help='Initial standard deviation of size')
    parser.add_argument('--size-mut', type=float, default=None, help='Mutation standard deviation for size (defaults to the trait mutation std)')
    parser.add_argument('--output', type=str, default='traits.csv', help='Output CSV file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (seeds both random and numpy)')
    # New arguments for multi-trait and selection
    parser.add_argument('--num-traits', type=int, default=None, help='Number of traits to simulate (if set, enables multi-trait mode)')
    parser.add_argument('--init-means', type=str, default='', help='Comma-separated initial means for each trait')
    parser.add_argument('--init-stds', type=str, default='', help='Comma-separated initial standard deviations for each trait')
    parser.add_argument('--mutation-stds', type=str, default='', help='Comma-separated mutation standard deviations for each trait')
    parser.add_argument('--optimum', type=str, default='', help='Comma-separated optimum values for each trait (for selection)')
    parser.add_argument('--selection-strength', type=float, default=0.0, help='Strength of stabilizing selection (0 = no selection)')
    # New arguments for position and live plot
    parser.add_argument('--position-std', type=float, default=0.05, help='Standard deviation for positional mutation')
    parser.add_argument('--position-min', type=float, default=0.0, help='Minimum initial position coordinate')
    parser.add_argument('--position-max', type=float, default=10.0, help='Maximum initial position coordinate')
    parser.add_argument('--similarity-scale', type=float, default=0.5, help='Similarity radius in population standard deviations (creatures farther apart than 2x this do not attract)')
    parser.add_argument('--move-step', type=float, default=0.1, help='Fraction of the distance moved toward similar creatures per generation')
    parser.add_argument('--interval', type=float, default=0.05, help='Seconds between generations when a plot is shown (animation speed)')
    parser.add_argument('--elements', action='store_true', help='Scatter random elements on the map that boost a trait of nearby creatures')
    parser.add_argument('--element-rate', type=float, default=1.0, help='Expected number of new elements per generation')
    parser.add_argument('--element-radius', type=float, default=1.0, help="Radius of each element's effect")
    parser.add_argument('--element-boost', type=float, default=0.05, help='Per-generation trait boost at an element center (linear falloff to 0 at the edge; negative values harm)')
    parser.add_argument('--element-lifetime', type=int, default=200, help='Generations each element persists')
    parser.add_argument('--element-max', type=int, default=1, help='Maximum simultaneous elements (no new ones spawn while full)')

    parser.add_argument('--live-plot', action='store_true', help='Show trait evolution in real time (line chart)')
    parser.add_argument('--animate', action='store_true', help='Show real-time animation of positions colored by traits')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        if np is not None:
            np.random.seed(args.seed)

    plt_module = None
    show_plot = False
    plot_callback = None
    stats_texts = []  # text boxes showing live stats, one per plot window

    # Prepare live plot (line chart) if requested
    if args.live_plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            sys.exit("Error: matplotlib is required for --live-plot. Install it with 'pip install matplotlib'.")
        plt_module = plt
        show_plot = True
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_callback, text = make_plot_handler(ax, plt, args.interval)
        stats_texts.append(text)
    # Prepare animation if requested
    if args.animate:
        if np is None:
            sys.exit("Error: numpy is required for --animate (comes with matplotlib). Install matplotlib with 'pip install matplotlib'.")
        import matplotlib
        # Try to switch to an interactive backend if we are using the default Agg backend
        if matplotlib.get_backend() == 'Agg':
            try:
                matplotlib.use('TkAgg')
            except Exception:
                try:
                    matplotlib.use('Qt5Agg')
                except Exception:
                    pass
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
        except ImportError:
            sys.exit("Error: matplotlib is required for --animate. Install it with 'pip install matplotlib'.")
        # If still non-interactive, we cannot show animation; warn and disable
        if matplotlib.get_backend() == 'Agg':
            sys.stderr.write("Warning: Animation will not be visible because no interactive matplotlib backend is available. Consider installing TkAgg or Qt5Agg backend. Disabling animation.")
            anim_callback = None
        else:
            plt_module = plt
            show_plot = True
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 8))
            # Positions are clamped to [position_min, position_max] in the
            # simulation, so fix the axis once: a single stray creature would
            # otherwise stretch the whole view
            fix_axis_to_position_range(ax, args.position_min, args.position_max)
            anim_callback = make_anim_handler(ax, plt, np, cm,
                                              args.position_min, args.position_max,
                                              args.interval)
            stats_texts.append(ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                       va='top', family='monospace', fontsize=9,
                                       bbox=dict(boxstyle='round', facecolor='white',
                                                 alpha=0.7, edgecolor='none')))
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
    run_generations = args.generations
    stop_when = None
    if show_plot:
        import matplotlib
        import matplotlib.backends
        non_interactive = matplotlib.backends.backend_registry.list_builtin(
            matplotlib.backends.BackendFilter.NON_INTERACTIVE)
        backend = matplotlib.get_backend()
        # Only a known non-interactive backend (e.g. Agg) can never
        # close a window; everything else runs until the window is
        # closed. get_backend() can also return the unresolved
        # auto-backend sentinel (not a string) when pyplot is mocked
        # in tests, which counts as interactive here.
        if not (isinstance(backend, str)
                and backend.lower() in non_interactive):
            # a real window exists: run until the user closes it
            run_generations = None

            def stop_when():
                return not plt_module.get_fignums()
    # Arguments shared by both simulation modes.
    common = dict(
        position_std=args.position_std,
        position_min=args.position_min,
        position_max=args.position_max,
        similarity_scale=args.similarity_scale,
        move_step=args.move_step,
        plot_callback=plot_callback,
        stop_when=stop_when,
        anim_callback=anim_callback,
        elements=args.elements,
        element_rate=args.element_rate,
        element_radius=args.element_radius,
        element_boost=args.element_boost,
        element_lifetime=args.element_lifetime,
        element_max=args.element_max,
        stats_callback=stats_cb,
        speed_mean=args.speed_mean,
        speed_std=args.speed_std,
        speed_mut=args.speed_mut,
        size_mean=args.size_mean,
        size_std=args.size_std,
        size_mut=args.size_mut,
    )
    if args.num_traits is None:
        # Backward compatibility: use original single-trait simulation
        averages = simulate(pop_size=args.pop_size, generations=run_generations,
                            mutation_std=args.mutation_std,
                            initial_mean=args.initial_mean,
                            initial_std=args.initial_std,
                            **common)
        # Write CSV with single column
        _write_csv(args.output, ['generation', 'average_trait'],
                   ([i, avg] for i, avg in enumerate(averages)))
        print(f'Simulation complete. Results written to {args.output}')
        if averages:
            print(f'Initial average: {averages[0]:.4f}')
            print(f'Final average: {averages[-1]:.4f}')
            print(f'Difference: {abs(averages[-1] - averages[0]):.4f}')
    else:
        # Multi-trait mode
        num_traits = args.num_traits
        # Helper to parse comma-separated floats
        def parse_list(s, expected_len, name):
            if not s:
                raise ValueError(f'{name} must be provided when using multi-trait mode')
            parts = s.split(',')
            if len(parts) != expected_len:
                raise ValueError(f'{name} must have exactly {expected_len} values (got {len(parts)})')
            try:
                return [float(p.strip()) for p in parts]
            except ValueError:
                raise ValueError(f'{name} must be comma-separated numbers')
        try:
            init_means = parse_list(args.init_means, num_traits, '--init-means')
            init_stds = parse_list(args.init_stds, num_traits, '--init-stds')
            mutation_stds = parse_list(args.mutation_stds, num_traits, '--mutation-stds')
            optimum = parse_list(args.optimum, num_traits, '--optimum')
        except ValueError as e:
            parser.error(str(e))

        # Record real speed/size history through the stats callback so the
        # CSV carries measured values (simulate_multi returns trait averages only).
        speed_hist, size_hist = [], []

        def record_stats(gen, stats):
            speed_hist.append(stats['speed_avg'])
            size_hist.append(stats['size_avg'])
            if stats_cb is not None:
                stats_cb(gen, stats)

        # Run multi-trait simulation
        averages_per_trait = simulate_multi(pop_size=args.pop_size, generations=run_generations,
                                            num_traits=num_traits,
                                            init_means=init_means,
                                            init_stds=init_stds,
                                            mutation_stds=mutation_stds,
                                            optimum=optimum,
                                            selection_strength=args.selection_strength,
                                            **{**common, 'stats_callback': record_stats})
        # Write CSV with columns for each trait, speed, and size
        header = ['generation'] + [f'trait{i}_avg' for i in range(num_traits)] + ['speed_avg', 'size_avg']
        rows = ([gen] + [averages_per_trait[i][gen] for i in range(num_traits)]
                + [speed_hist[gen], size_hist[gen]]
                for gen in range(len(averages_per_trait[0])))
        _write_csv(args.output, header, rows)
        print(f'Simulation complete. Results written to {args.output}')
        if averages_per_trait[0]:
            _print_trait_summary(averages_per_trait)

    # After simulation, if live plot or animation was shown, keep window open until user closes
    if show_plot:
        plt_module.ioff()
        plt_module.show()


if __name__ == '__main__':
    main()
