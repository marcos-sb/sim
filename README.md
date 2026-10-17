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

## Output format

`generation,trait0_avg,...,trait{N-1}_avg,speed_avg,size_avg` — one column
per trait plus measured speed/size averages.

`*.csv` and `*.png` are git-ignored outputs (see `.gitignore`).

## Performance

Nearest-neighbor search and similarity movement are O(N²) per generation
(`_nearest_neighbour`, `_move_toward_similar`, pairwise `spread` stats).
Measured on this machine (5 generations, default movement): N=100 ~0.03s,
N=200 ~0.09s, N=400 ~0.36s. N≤500 is comfortable; N≥2000 gets slow. If you
need larger populations, consider a spatial index (`scipy.spatial.cKDTree`)
— left out intentionally to keep runtime deps to `numpy` + `matplotlib`.

## Python API

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
per-trait lists, `optimum` / `selection_strength`, and `trait_properties`.
With `elements=True`, `stats` includes `elements`, and `anim_callback`
receives `(gen, population, elements)`.

## Tests

87 tests, all passing:

```bash
python3 -m pytest -q
ruff check .
black --check .
```

Covers reproduction/inheritance, position bounds/midpoints, movement,
selection, trait-property hooks (`survival_decay`, `element_resist`,
`repro_boost`, `steer`, `move_speed`, `repel`, `cluster_bias`,
`life_expect`), config validation and CLI precedence, elements, stats
formatting, CSV output, seed reproducibility, and `--live-plot` /
`--animate` callback wiring (mocked `matplotlib`).

## Project structure

```text
model.py          # evolution model: reproduction, movement, selection,
                  # elements, trait-property hooks
config.py         # sim.toml load/validate (tomllib), starter template
live_viz.py       # live matplotlib handlers (line chart + 2D animation)
simulate.py       # config-first thin CLI (argparse, CSV output);
                  # re-exports the model API
visualize.py      # offline CSV -> matplotlib line chart (--traits, --with-body)
tests/            # behavior, callback/wiring, and config tests
pyproject.toml    # packaging, deps, pytest/ruff/black config
LICENSE           # MIT
traits.csv        # example multi-trait output (ignored by git)
demo*.csv, test_*.csv, *.png  # example outputs (ignored by git)
```

## License

MIT — see `LICENSE`.
