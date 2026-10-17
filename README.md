# sim

Agent-based simulation of evolving creature traits on a 2D map.

Each creature has evolvable traits plus `speed` / `size` body attributes and
an `(x, y)` position. Each generation, every creature interacts with its
nearest spatial neighbor: the offspring inherits one parent's value per
trait plus Gaussian mutation, and is placed at the parents' midpoint plus
Gaussian positional mutation. Creatures then move toward others with similar
traits. Optional stabilizing selection and map elements can bias evolution.

Traits are not inert numbers: each trait is assigned a *property* — what it
does to the creature (`move_speed`, `steer`, `survival_decay`,
`element_resist`, ...) — so evolution acts on behavior as well as on values.

## Features

- **Config-first**: everything evolvable lives in `sim.toml`; remaining CLI
  flags only override or toggle. Precedence: `CLI flag > sim.toml > default`.
- Trait properties (`[traits].properties = ["move_speed", "steer", ...]`):
  per-trait behaviors wired into movement, elements, reproduction, decay,
  and mortality (see [Trait properties](#trait-properties)).
- Single- and multi-trait: the length of `[traits].init_means` defines the
  trait count; one value = single-trait mode.
- Stabilizing selection: Gaussian fitness toward `optimum`, weighted
  resampling.
- Evolvable `speed` (`0.1–2.0`) and `size` (`0.3–3.0`): inherited from one
  parent, mutated, clamped. Speed scales movement; size scales marker area
  and personal-space repulsion.
- Spatial clustering: attraction to trait-similar centroid within
  `similarity_scale * trait-std` (cutoff `2σ`), repulsion from dissimilar
  creatures within mean spacing. Positions clamped to
  `[position_min, position_max]`.
- Map elements: randomly spawned boosters/harms with
  `rate / radius / boost / lifetime / max`, linear falloff to the edge.
- Optional `[hunter]`: a roaming predator (config-tuned, disabled by
  default) that kills the nearest creature in reach each generation.
- Interactive mode (`--interactive`): pause/step, live selection/mutation
  dials, mouse placement of food / toxins / mutagens / optimum beacons,
  per-trait optimum sliders, and snapshot saving (see
  [Interactive mode](#interactive-mode)).
- Live outputs: `--live-plot` trait-average line chart, `--animate` 2D map
  (`color = trait`, `size = size`, `X` = elements), live stats textbox.
- CSV output + `visualize.py` offline plotter.
- Reproducibility via `--seed` (`random` + `numpy`).

## Requirements

- Python 3.11+ (stdlib `tomllib` parses the config; no extra deps)
- `numpy`, `matplotlib` — required only for `--live-plot` / `--animate`
- `pytest`, `ruff`, `black` — dev only

```bash
pip install -e .          # runtime deps (numpy, matplotlib)
pip install -e ".[dev]"   # + pytest, ruff, black
```

## Quickstart

Generate a commented starter config, tailor it, and run:

```bash
python3 simulate.py --config-template > sim.toml
$EDITOR sim.toml
python3 simulate.py --config sim.toml --seed 42 --output traits.csv
```

Output:

```text
Simulation complete. Results written to traits.csv
Initial averages per trait:
  Trait 0: 0.0077
  ...
```

```csv
generation,trait0_avg,trait1_avg,speed_avg,size_avg
0,0.007675472498082147,1.0053550650449912,1.0176690239489357,1.0013563055707209
...
```

One trait = single-trait mode:

```toml
[run]
generations = 5
[traits]
init_means = [1.0]
```

```bash
python3 simulate.py --config sim.toml --pop-size 8 --seed 42
```

Offline plot of the results:

```bash
python3 visualize.py traits.csv --output traits.png
python3 visualize.py traits.csv   # show window instead
# multi-trait: filter traits, optionally include speed/size series
python3 visualize.py traits.csv --traits 0,1 --with-body --output traits.png
```

Note: CSVs store per-generation averages only, so offline plots show
average series (no std/spread bands — those appear in the live stats box).

## The configuration file

`sim.toml` (any path via `--config`) is the canonical home for everything
evolvable. Unknown sections and keys are rejected with precise messages
before the simulation starts; `[traits]` is the only required section.

```toml
[run]
pop_size = 100        # creatures per generation (>= 1)
generations = 300     # cap; ignored while a real plot window is open
# seed = 42           # reproducible run (also a CLI flag)
# output = "traits.csv"

[traits]              # required
init_means    = [0.0, 1.0, 2.0]     # length defines the trait count
init_stds     = [0.1, 0.1, 0.1]
mutation_stds = [0.05, 0.05, 0.05]
# mutation_std = 0.05               # shorthand: same rate for all traits
# optimum = [0.0, 1.0, 2.0]         # required when selection_strength > 0
selection_strength = 0.0            # 0.0 = pure drift
properties = ["move_speed", "steer", "repel"]  # one per trait

[body]                  # evolvable, separate from [traits]
speed_mean = 1.0
speed_std  = 0.2
# speed_mutation = 0.05 # defaults to trait 0's mutation rate
size_mean = 1.0
size_std  = 0.2
# size_mutation = 0.05  # defaults to trait 0's mutation rate

[space]
position_std     = 0.05 # positional jitter per generation
position_min     = 0.0
position_max     = 10.0
similarity_scale = 0.5  # similar-trait attract radius (pop std devs)
move_step        = 0.1  # fraction of distance moved per generation

[elements]
enabled  = false        # map elements boost nearby creatures' traits
rate     = 1.0          # expected new elements per generation
radius   = 1.0          # effect radius
boost    = 0.05         # gain at center, linear falloff to edge
lifetime = 200          # generations an element persists
max      = 5            # hard cap on simultaneous elements

# [hunter]               # optional roaming predator, off by default
# enabled = false
# speed   = 0.2         # fraction of the distance to its target per
#                        # generation (plus the usual positional jitter)
# radius  = 1.0          # kill radius
```

Notes:

- `selection_strength > 0` without `optimum` is a configuration error and is
  caught before the run starts.
- Fitness is `exp(-strength * squared-distance-to-optimum)` with weighted
  resampling.
- `similarity_scale=0` disables directed movement (only mutation jitter).

## CLI

```text
--config PATH         load sim.toml (required for a run)
--config-template     print a commented starter sim.toml and exit
--pop-size N          override [run].pop_size
--generations N       override [run].generations
--output PATH         override the output CSV path (default traits.csv)
--seed N              random + numpy seed for reproducibility
--live-plot           trait-average line chart while running
--animate             2D map animation (color = trait, size = size)
--plot-block          with --live-plot/--animate: cap the run at
                      --generations instead of running until the plot
                      window closes (agents, scripts, SSH)
--interval SEC        pause between generations when a plot is shown
--interactive         mouse/keyboard control of the run (see
                      "Interactive mode"); implies --animate and
                      refuses a non-interactive backend
--elements            enable [elements] (equivalent to enabled = true)
--element-rate / --element-radius / --element-boost /
--element-lifetime / --element-max
                      override the [elements] scalars
```

Everything else — trait lists, optimum, body, space — is config-only by
design. Run `python3 simulate.py` without `--config` for instructions
pointing at `--config-template`.

## Single- vs multi-trait

Historically the two modes had disjoint CLI surfaces; now they are the same
mechanism: `[traits]` lists of length 1 = single-trait, length N = multi.
The CSV always uses the unified multi-trait header:

```csv
generation,trait0_avg,...,trait{N-1}_avg,speed_avg,size_avg
```

The legacy `simulate()` API (single keyword-arg call, `trait` key, old CSV
header) is unchanged for existing callers and tests.

## Trait properties

A trait's *property* says what the trait value does. Each trait index gets
exactly one property; properties are fixed for the whole run (`none`
allowed). The registry lives in `model._prop_value` with validation in
`config.py`:

| property         | hook                                                                 |
|------------------|----------------------------------------------------------------------|
| `move_speed`     | extra displacement multiplier `max(v, 0)` on top of `speed`          |
| `steer`          | extra `move_step * v` toward the similar centroid (`v < 0` flees)    |
| `repel`          | personal-space radius multiplier `max(v, 0)` (`v=1` doubles it)      |
| `cluster_bias`   | attraction contribution scaled by `clamp(abs(v), 0, 1)`              |
| `survival_decay` | upkeep: value shrinks per generation by `min(0.5, abs(v) * 0.1)`      |
| `element_resist` | element boost scaled by `1 - clamp(abs(v), 0, 1)` (`v=1` is immune)  |
| `repro_boost`    | offspring of this trait inherit from the parent with larger magnitude |
| `life_expect`    | death chance `1 / (abs(v) * 20)` capped at 0.5; a death hands the slot to a mutated clone of the nearest neighbour |

Example: a trait that speeds movement but decays and attracts dangerous
elements:

```toml
[traits]
init_means = [1.0, 0.0]
init_stds = [0.1, 0.1]
mutation_stds = [0.05, 0.05]
properties = ["move_speed", "element_resist"]

[elements]
enabled = true
```

Trade-offs emerge: fast movers spread further, but more `move_speed` means
more decay per generation, and selection (if enabled) can favor a balance.

### Live visualization

```bash
python3 simulate.py --config sim.toml --live-plot --interval 0.05
python3 simulate.py --config sim.toml --animate --interval 0.05
python3 simulate.py --config sim.toml --live-plot --animate --elements
```

- Single trait color: `viridis` on a widen-only fixed scale.
- Multi-trait with ≥3 traits: first 3 traits as RGB on per-channel
  widen-only scales; otherwise first trait as scalar.
- Marker area `15–120` from `size` on a widen-only scale.
- Axis is fixed to `[position_min, position_max]` plus 5% padding.
- Stats box: `gen / avg / std / spread / elements / speed / size`.
- When a real plot window is open, the run continues until the window is
  closed (the configured/`--generations` cap is ignored). `--plot-block`
  restores the cap for headless contexts. In the Python API,
  `generations=None` requires `stop_when`.

## Interactive mode

`python3 simulate.py --config sim.toml --interactive` runs the same
engine with a live control surface. It implies `--animate` and refuses to
start without an interactive matplotlib backend (TkAgg/Qt5Agg/macOS).
Everything interactive is layered on top of the config: `sim.toml` stays
the base, the dials are *multipliers* on its values, and a saved snapshot
bakes the current state and multipliers into a runnable `sim.toml`.

### Tools (mouse + number keys)

| key | tool | effect |
|-----|------|--------|
| `1` | food | boosts the targeted trait of creatures in the radius (positive) |
| `2` | toxin | same but with a negative boost |
| `3` | mutagen | Gaussian jitter of every trait; magnitude scales with distance inside the radius |
| `4` | beacon | captures the current phenotype of nearby creatures and pulls selection inside its radius toward that snapshot (nearest beacon wins) |
| `x` | erase | the next click deletes the nearest element it lands on |

Click to place an element with `[elements].radius`; press-drag-release
places it with the drag distance as its radius. `n` cycles the targeted
trait (`trait {k}` in the HUD). Player elements bypass the auto-spawn
`max` cap but expire like any other element.

### Dials and run control (keyboard)

| key | action |
|-----|--------|
| `space` | pause / resume |
| `s` | run one generation while paused |
| `[` / `]` | selection-strength multiplier (-0.25 step, 0–8) |
| `,` / `.` | mutation-rate multiplier (-0.25 step, 0–8) |
| `-` / `=` | slower / faster pause time (animation speed) |
| `w` | write `snapshot.toml`: population means/stds + current multipliers + effective optimum, loadable with `--config` |

The per-trait optimum can also be tuned with sliders in a strip along
the bottom of the animation window (absolute trait values; raises `[ ]`
first if `selection_strength` is 0 — pure drift ignores the optimum).
They share the map's canvas so their events are always serviced.

### Hunter

`[hunter] enabled = true` turns on the predator even in headless runs:

- each generation it drifts `<speed>` of the way toward the nearest
  creature (plus the usual positional jitter);
- any creature within `radius` is killed — its slot is refilled by a
  mutated clone of its nearest neighbour, so the population size stays
  constant and the predator indirectly selects for survival, not just
  the optimum.
- With `--interactive` it is drawn as a black marker; the dials do not
  affect it.

### Notes

- Headless (no `--interactive`) runs behave exactly as before; the hunter
  is the only interactive-era feature that is config-driven.
- `--interactive --plot-block` is allowed: the run then stops at
  `--generations` instead of running until the window closes.
- Erase and element placement only apply to the element list — beacons
  that an erase click removes stop affecting selection immediately.

## Output format

`generation,trait0_avg,...,trait{N-1}_avg,speed_avg,size_avg` — one column
per trait plus measured speed/size averages.

`*.csv` and `*.png` are git-ignored outputs (see `.gitignore`).

## Performance

Two-tier cost per generation:

- Reproduction uses a uniform spatial grid (`_nearest_neighbours`): tiles
  the map with ~sqrt(N) cells, expands outward in rings until no
  unexplored ring can hold something closer. Expected O(N) for
  well-spread populations (pure stdlib — no scipy).
- Similarity movement and `spread` stats stay O(N²) pairwise, but with
  per-call precomputed arrays and C-speed `math.dist` on trait vectors.

Measured on this machine (5 generations, default movement, before →
after): N=100 0.042 → 0.016s, N=200 0.165 → 0.047s, N=400 0.654 → 0.176s,
N=1000 4.06 → 1.04s. N≤1000 is comfortable now; beyond that the O(N²)
movement dominates — a trait-space spatial index or numpy vectorization
would be the next step, intentionally left out to keep runtime deps to
`numpy` + `matplotlib`.

## Python API

Single and multi share one engine: `simulate()` is a single-trait call into
`simulate_multi(num_traits=1)` — same signature, same RNG stream, and its
`anim_callback` still receives creatures in the legacy flat
`{'trait': ..., 'pos': ...}` shape.

```python
from simulate import simulate, simulate_multi

avg = simulate(pop_size=100, generations=100,
               mutation_std=0.05, initial_mean=1.0, initial_std=0.1)

per_trait = simulate_multi(pop_size=100, generations=100, num_traits=2,
                           init_means=[0.0, 1.0], init_stds=[0.1, 0.1],
                           mutation_stds=[0.05, 0.05],
                           optimum=[0.0, 1.0], selection_strength=0.0,
                           trait_properties=['move_speed', 'steer'])
```

Optional callbacks per generation:

```python
simulate(..., plot_callback=lambda gen, averages: ...,
         anim_callback=lambda gen, population: ...,
         stats_callback=lambda gen, stats: ...,
         stop_when=lambda: False)
```

`simulate_multi` accepts everything `simulate()` does plus `num_traits`,
per-trait lists, `optimum` / `selection_strength`, `trait_properties`,
and the interactive/engine extras `hunter_enabled` / `hunter_speed` /
`hunter_radius`, `controls` (a `Controls` object from `controls.py`)
and `wait_callback` (a pause/step gate called before every generation).
With `elements=True`, `stats` includes `elements`, and `anim_callback`
receives `(gen, population, elements)`; when the hunter is on it also
receives the hunter dict as a fourth argument.

## Tests

131 tests, all passing:

```bash
python3 -m pytest -q
ruff check .
black --check .
```

Covers reproduction/inheritance, position bounds/midpoints, movement,
selection, trait-property hooks (`survival_decay`, `element_resist`,
`repro_boost`, `steer`, `move_speed`, `repel`, `cluster_bias`,
`life_expect`), config validation and CLI precedence, elements, stats
formatting, CSV output, seed reproducibility, `--live-plot` /
`--animate` callback wiring (mocked `matplotlib`), and the interactive
stack (`controls.py` state machine, snapshot TOML roundtrip,
controls-driven mutation/selection effects, hunter kills, beacon
optimum capture/selection, CLI wiring incl. snapshot save and backend
refusal, mouse/keyboard handler dispatch).

## Project structure

```text
model.py          # evolution model: reproduction, movement, selection,
                  # elements, trait-property hooks, hunter, beacons
controls.py       # interactive state (tools, dials, requests, snapshot)
config.py         # sim.toml load/validate (tomllib), starter template
live_viz.py       # live matplotlib handlers (line chart, 2D animation,
                  # mouse/keyboard event binding)
simulate.py       # config-first thin CLI (argparse, CSV output,
                  # --interactive wiring); re-exports the model API
visualize.py      # offline CSV -> matplotlib line chart (--traits, --with-body)
tests/            # behavior, callback/wiring, config, and controls tests
pyproject.toml    # packaging, deps, pytest/ruff/black config
LICENSE           # MIT
traits.csv        # example multi-trait output (ignored by git)
demo*.csv, test_*.csv, *.png  # example outputs (ignored by git)
```

## License

MIT — see `LICENSE`.
