"""Live matplotlib handlers: trait line chart + 2D creature animation.

Extracted from `simulate.main()` so the model (`model.py`) stays free of
matplotlib. Factories take the already-imported pyplot/numpy modules so
tests can inject mocks.
"""

from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from model import _color_norm, _widen_scale


def make_plot_handler(ax, plt, interval):
    """Return (plot_callback, stats_text) for `--live-plot`."""
    lines = []
    gen_data = []
    trait_data = []  # list of lists per trait
    text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                   va='top', family='monospace', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white',
                             alpha=0.7, edgecolor='none'))

    def plot_callback(gen, averages):
        gen_data.append(gen)
        if not trait_data:
            for _ in averages:
                trait_data.append([])
        for i, avg in enumerate(averages):
            trait_data[i].append(avg)
        if not lines:
            for i in range(len(averages)):
                (line,) = ax.plot([], [], label=f'Trait {i}')
                lines.append(line)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Average Trait Value')
        ax.set_title('Trait Evolution over Generations (live)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        for i, line in enumerate(lines):
            line.set_data(gen_data, trait_data[i])
        plt.pause(interval)

    return plot_callback, text


def make_anim_handler(ax, plt, np, cm, position_min, position_max, interval):
    """Return anim_callback for `--animate`.

    Colors: single trait via viridis; multi-trait with >=3 traits as RGB,
    else first trait via viridis. Marker area encodes size. Scales only
    widen so identical values keep identical colors across frames.
    """
    scat = None  # initialized on first callback
    elem_scat = None  # element scatter, init on first callback
    elem_circles = []  # radius circles, redrawn each frame
    trait_vmin = None
    trait_vmax = None
    rgb_mins = None
    rgb_maxs = None
    size_vmin = None
    size_vmax = None

    def anim_callback(gen, population, elements=None):
        nonlocal scat, trait_vmin, trait_vmax, rgb_mins, rgb_maxs, elem_scat
        nonlocal size_vmin, size_vmax
        xs = [c['pos'][0] for c in population]
        ys = [c['pos'][1] for c in population]
        sample = population[0]
        if 'trait' in sample:
            trait_vals = [c['trait'] for c in population]
            trait_vmin, trait_vmax = _widen_scale(trait_vmin, trait_vmax, trait_vals)
            norm = _color_norm(trait_vals, trait_vmin, trait_vmax)
            colors = cm.viridis(norm)
        else:
            tra = np.array([c['traits'] for c in population], dtype=float)
            if tra.shape[1] >= 3:
                rgb = tra[:, :3]
                if rgb_mins is None:
                    rgb_mins = rgb.min(axis=0)
                    rgb_maxs = rgb.max(axis=0)
                else:
                    rgb_mins = np.minimum(rgb_mins, rgb.min(axis=0))
                    rgb_maxs = np.maximum(rgb_maxs, rgb.max(axis=0))
                for ch in range(3):
                    if rgb_maxs[ch] - rgb_mins[ch] < 1e-12:
                        rgb[:, ch] = 0.5
                    else:
                        rgb[:, ch] = (rgb[:, ch] - rgb_mins[ch]) / (rgb_maxs[ch] - rgb_mins[ch])
                colors = rgb
            else:
                trait_vals = tra[:, 0].tolist()
                trait_vmin, trait_vmax = _widen_scale(trait_vmin, trait_vmax, trait_vals)
                norm = _color_norm(trait_vals, trait_vmin, trait_vmax)
                colors = cm.viridis(norm)
        sizes = [c.get('size', 1.0) for c in population]
        size_vmin, size_vmax = _widen_scale(size_vmin, size_vmax, sizes)
        areas = [15.0 + 105.0 * v for v in _color_norm(sizes, size_vmin, size_vmax)]
        if scat is None:
            scat = ax.scatter(xs, ys, c=colors, s=areas, edgecolors='k', linewidths=0.5)
            ax.set_xlabel('X position')
            ax.set_ylabel('Y position')
            ax.set_title('Creature positions (color = trait, size = size)')
            ax.grid(True, linestyle='--', alpha=0.5)
        else:
            scat.set_offsets(np.column_stack([xs, ys]))
            scat.set_color(colors)
            scat.set_sizes(areas)
        if elements:
            exs = [e['pos'][0] for e in elements]
            eys = [e['pos'][1] for e in elements]
            ecolors = [cm.tab10(e['trait'] % 10) for e in elements]
            esizes = [150 + 600 * abs(e['boost']) for e in elements]
            if elem_scat is None:
                elem_scat = ax.scatter(exs, eys, c=ecolors, s=esizes, marker='X',
                                       edgecolors='k', linewidths=1.5)
                ax.legend(handles=[
                    Line2D([0], [0], marker='X', color='w', markerfacecolor='gray',
                           markeredgecolor='k', markersize=8, linestyle='None',
                           label='element (color = trait, size = |boost|)'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
                           markeredgecolor='gray', markersize=10, linestyle=':',
                           label='effect radius')],
                    loc='upper right', fontsize=8, framealpha=0.7)
            else:
                elem_scat.set_offsets(np.column_stack([exs, eys]))
                elem_scat.set_color(ecolors)
                elem_scat.set_sizes(esizes)
            for circle in elem_circles:
                circle.remove()
            elem_circles.clear()
            for e in elements:
                elem_circles.append(ax.add_patch(Circle(
                    e['pos'], e['radius'], fill=False, linestyle=':',
                    linewidth=2, color=cm.tab10(e['trait'] % 10), alpha=0.6)))
        plt.pause(interval)

    return anim_callback


def fix_axis_to_position_range(ax, position_min, position_max):
    """Pin the animation axis once so strays cannot stretch the view."""
    pad = 0.05 * (position_max - position_min)
    if pad <= 0:
        pad = 0.5
    ax.set_xlim(position_min - pad, position_max + pad)
    ax.set_ylim(position_min - pad, position_max + pad)


__all__ = ["make_plot_handler", "make_anim_handler", "fix_axis_to_position_range"]
