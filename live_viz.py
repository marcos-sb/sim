"""Live matplotlib handlers: trait line chart + 2D creature animation.

Extracted from `simulate.main()` so the model (`model.py`) stays free of
matplotlib. Factories take the already-imported pyplot/numpy modules so
tests can inject mocks. `interval` may be a float or a zero-arg callable
(returning a float): the interactive CLI uses a callable so the -/= speed
keys can rerate the pause without rebuilding handlers.
"""

from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from controls import MULT_STEP, TOOL_KEYS
from model import _color_norm, _widen_scale

_CURVED_KIND_COLOR = {"mutagen": "crimson", "beacon": "gold"}


def _pause(plt, interval):
    plt.pause(interval() if callable(interval) else interval)


def make_plot_handler(ax, plt, interval):
    """Return (plot_callback, stats_text) for `--live-plot`."""
    lines = []
    gen_data = []
    trait_data = []  # list of lists per trait
    text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    def plot_callback(gen, averages):
        gen_data.append(gen)
        if not trait_data:
            for _ in averages:
                trait_data.append([])
        for i, avg in enumerate(averages):
            trait_data[i].append(avg)
        if not lines:
            for i in range(len(averages)):
                (line,) = ax.plot([], [], label=f"Trait {i}")
                lines.append(line)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Average Trait Value")
        ax.set_title("Trait Evolution over Generations (live)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        for i, line in enumerate(lines):
            line.set_data(gen_data, trait_data[i])
        # set_data() does not rescale the axes: without an explicit
        # re-autoscale the view stays at the empty (0,1) limits and the
        # line is drawn outside the visible area
        ax.relim()
        ax.autoscale_view()
        _pause(plt, interval)

    return plot_callback, text


def make_anim_handler(ax, plt, np, cm, position_min, position_max, interval):
    """Return (anim_callback, stats_text) for `--animate`.

    Colors: single trait via viridis; multi-trait with >=3 traits as RGB,
    else first trait via viridis. Marker area encodes size. Scales only
    widen so identical values keep identical colors across frames.

    Elements draw per kind: food/toxin keep the trait-colored 'X',
    mutagen a crimson 'P', beacon a gold '*'. Elements without a kind
    keep the legacy boost-sign interpretation. `hunter` (when passed)
    draws as a black 'P' star.
    """
    scat = None  # initialized on first callback
    elem_scats = {}  # (marker, color) -> element scatter, init on first use
    elem_circles = []  # radius circles, redrawn each frame
    hunter_scat = None
    trait_vmin = None
    trait_vmax = None
    rgb_mins = None
    rgb_maxs = None
    size_vmin = None
    size_vmax = None
    # stats text box lives with the handler, symmetrical with the plot one
    text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    def anim_callback(gen, population, elements=None, hunter=None):
        nonlocal scat, trait_vmin, trait_vmax, rgb_mins, rgb_maxs
        nonlocal size_vmin, size_vmax, hunter_scat
        xs = [c["pos"][0] for c in population]
        ys = [c["pos"][1] for c in population]
        sample = population[0]
        if "trait" in sample:
            trait_vals = [c["trait"] for c in population]
            trait_vmin, trait_vmax = _widen_scale(trait_vmin, trait_vmax, trait_vals)
            norm = _color_norm(trait_vals, trait_vmin, trait_vmax)
            colors = cm.viridis(norm)
        else:
            tra = np.array([c["traits"] for c in population], dtype=float)
            if tra.ndim == 2 and tra.shape[1] >= 3:
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
                if tra.ndim == 2:
                    trait_vals = tra[:, 0].tolist()
                else:
                    # mocked/odd shapes: flatten whatever came back
                    trait_vals = tra.reshape(-1).tolist()
                trait_vmin, trait_vmax = _widen_scale(trait_vmin, trait_vmax, trait_vals)
                norm = _color_norm(trait_vals, trait_vmin, trait_vmax)
                colors = cm.viridis(norm)
        sizes = [c.get("size", 1.0) for c in population]
        size_vmin, size_vmax = _widen_scale(size_vmin, size_vmax, sizes)
        areas = [15.0 + 105.0 * v for v in _color_norm(sizes, size_vmin, size_vmax)]
        if scat is None:
            scat = ax.scatter(xs, ys, c=colors, s=areas, edgecolors="k", linewidths=0.5)
            ax.set_xlabel("X position")
            ax.set_ylabel("Y position")
            ax.set_title("Creature positions (color = trait, size = size)")
            ax.grid(True, linestyle="--", alpha=0.5)
        else:
            scat.set_offsets(np.column_stack([xs, ys]))
            scat.set_color(colors)
            scat.set_sizes(areas)
        if elements:
            # group by visual kind: (marker, color); food/toxin share the
            # trait-colored 'X', mutagen/beacon get their own marker
            groups = {}
            for e in elements:
                kind = e.get("kind", "food" if e["boost"] >= 0 else "toxin")
                if kind in ("food", "toxin"):
                    key = ("X", None)
                elif kind == "mutagen":
                    key = ("P", "crimson")
                else:
                    key = ("*", "gold")
                groups.setdefault(key, []).append(e)
            for key, members in groups.items():
                marker, color = key
                mxs = [e["pos"][0] for e in members]
                mys = [e["pos"][1] for e in members]
                msizes = [150 + 600 * abs(e["boost"]) for e in members]
                if color is None:
                    mcolors = [cm.tab10(e["trait"] % 10) for e in members]
                else:
                    mcolors = color
                if key not in elem_scats:
                    elem_scats[key] = ax.scatter(
                        mxs,
                        mys,
                        c=mcolors,
                        s=msizes,
                        marker=marker,
                        edgecolors="k",
                        linewidths=1.5,
                    )
                    legend_handles = [
                        Line2D(
                            [0],
                            [0],
                            marker="X",
                            color="w",
                            markerfacecolor="gray",
                            markeredgecolor="k",
                            markersize=8,
                            linestyle="None",
                            label="element (color = trait, size = |boost|)",
                        ),
                        Line2D(
                            [0],
                            [0],
                            marker="P",
                            color="w",
                            markerfacecolor="crimson",
                            markeredgecolor="k",
                            markersize=8,
                            linestyle="None",
                            label="mutagen",
                        ),
                        Line2D(
                            [0],
                            [0],
                            marker="*",
                            color="w",
                            markerfacecolor="gold",
                            markeredgecolor="k",
                            markersize=12,
                            linestyle="None",
                            label="beacon (traits = captured)",
                        ),
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="w",
                            markerfacecolor="none",
                            markeredgecolor="gray",
                            markersize=10,
                            linestyle=":",
                            label="effect radius",
                        ),
                    ]
                    if hunter is not None:
                        legend_handles.append(
                            Line2D(
                                [0],
                                [0],
                                marker="P",
                                color="w",
                                markerfacecolor="k",
                                markeredgecolor="w",
                                markersize=8,
                                linestyle="None",
                                label="hunter",
                            )
                        )
                    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.7)
                else:
                    elem_scats[key].set_offsets(np.column_stack([mxs, mys]))
                    elem_scats[key].set_color(mcolors)
                    elem_scats[key].set_sizes(msizes)
            for key in list(elem_scats):
                if key not in groups:
                    elem_scats[key].set_offsets(np.column_stack([[], []]))
            for circle in elem_circles:
                circle.remove()
            elem_circles.clear()
            for e in elements:
                kind = e.get("kind", "food" if e["boost"] >= 0 else "toxin")
                ccolor = _CURVED_KIND_COLOR.get(kind) or cm.tab10(e["trait"] % 10)
                elem_circles.append(
                    ax.add_patch(
                        Circle(
                            e["pos"],
                            e["radius"],
                            fill=False,
                            linestyle=":",
                            linewidth=2,
                            color=ccolor,
                            alpha=0.6,
                        )
                    )
                )
        elif elem_scats:
            for sc in elem_scats.values():
                sc.set_offsets(np.column_stack([[], []]))
        if hunter is not None:
            hx, hy = hunter["pos"]
            if hunter_scat is None:
                hunter_scat = ax.scatter(
                    [hx],
                    [hy],
                    c="k",
                    s=220,
                    marker="P",
                    edgecolors="w",
                    linewidths=1.0,
                )
            else:
                hunter_scat.set_offsets(np.column_stack([hx, hy]))
        _pause(plt, interval)

    return anim_callback, text


def fix_axis_to_position_range(ax, position_min, position_max):
    """Pin the animation axis once so strays cannot stretch the view."""
    pad = 0.05 * (position_max - position_min)
    if pad <= 0:
        pad = 0.5
    ax.set_xlim(position_min - pad, position_max + pad)
    ax.set_ylim(position_min - pad, position_max + pad)


def bind_interactive_controls(fig, controls):
    """Wire the mouse/keyboard handlers onto the animation figure.

    Keys: 1 food, 2 toxin, 3 mutagen, 4 beacon (tool); x erase mode;
    n cycle target trait; space pause; s step one generation;
    [ / ] selection-strength dial; , / . mutation dial; - / = speed;
    w save a snapshot.toml.

    Mouse: click to place the current tool; press-drag-release places
    at the press point with the drag distance as the element radius.
    """
    canvas = fig.canvas
    press = [None]

    def on_key(event):
        key = event.key
        if key == " ":
            controls.toggle_pause()
        elif key == "s":
            controls.request_step()
        elif key in TOOL_KEYS:
            controls.set_tool(TOOL_KEYS[key])
        elif key == "x":
            controls.toggle_erase()
        elif key == "n":
            controls.cycle_trait()
        elif key == "[":
            controls.adjust("selection", -MULT_STEP)
        elif key == "]":
            controls.adjust("selection", MULT_STEP)
        elif key == ",":
            controls.adjust("mutation", -MULT_STEP)
        elif key == ".":
            controls.adjust("mutation", MULT_STEP)
        elif key in ("=", "+"):
            controls.speed_up()
        elif key in ("-", "_"):
            controls.slow_down()
        elif key == "w":
            controls.request_save()

    def on_press(event):
        if event.inaxes is not None and event.xdata is not None:
            press[0] = (event.xdata, event.ydata)

    def on_release(event):
        try:
            if press[0] is None or event.xdata is None:
                return
            radius = ((event.xdata - press[0][0]) ** 2 + (event.ydata - press[0][1]) ** 2) ** 0.5
            # a plain click (no drag) uses the configured default radius
            controls.click(press[0], radius if radius > 0.2 else None)
        finally:
            press[0] = None

    canvas.mpl_connect("key_press_event", on_key)
    canvas.mpl_connect("button_press_event", on_press)
    canvas.mpl_connect("button_release_event", on_release)


__all__ = [
    "bind_interactive_controls",
    "fix_axis_to_position_range",
    "make_anim_handler",
    "make_plot_handler",
]
