# sim

Agent-based simulation of evolving creature traits on a 2D map.

Each creature has evolvable traits plus `speed` / `size` body attributes and
an `(x, y)` position. Each generation, every creature interacts with its
nearest spatial neighbor: the offspring inherits one parent's trait value per
trait plus Gaussian mutation, and is placed at the parents' midpoint plus
Gaussian positional mutation. Creatures then move toward others with similar
traits. Optional stabilizing selection and map elements can bias evolution.

## Features

- Single-trait mode (`simulate()`): backward-compatible `trait` evolution.
- Multi-trait mode (`simulate_multi()`): `traits[]` vector with per-trait
  init / mutation, plus Gaussian stabilizing selection toward `--optimum`.
- Evolvable `speed` (`0.1–2.0`) and `size` (`0.3–3.0`): inherited from one
  parent, mutated, clamped. Speed scales movement; size scales marker area
  and personal-space repulsion.
- Spatial clustering (`_move_toward_similar()`): attraction to trait-similar
  centroid within `similarity_scale * trait-std` (cutoff `2σ`), repulsion
  from dissimilar creatures within mean spacing. Positions are clamped to
  `[position_min, position_max]`.
- Map elements (`--elements`): randomly spawned boosters/harms with
  `rate / radius / boost / lifetime / max`, linear falloff to the edge.
- Live outputs: `--live-plot` trait-average line chart, `--animate` 2D map
  (`color = trait`, `size = size`, `X` = elements), live stats textbox.
- CSV output + `visualize.py` offline plotter.
- Reproducibility via `--seed` (`random` + `numpy`).

## Requirements

- Python 3.9+
- `numpy`, `matplotlib` — required only for `--live-plot` / `--animate`
- `pytest` — for tests

No `requirements.txt`; install directly, e.g.:

```bash
pip install numpy matplotlib pytest
```

## Quickstart

Single trait, 100 generations (default output `traits.csv`):

```bash
python3 simulate.py --generations 5 --pop-size 10 --seed 42 --output traits.csv
```

Output:

```text
Simulation complete. Results written to traits.csv
Initial average: 1.0262
Final average: 1.0855
Difference: 0.0593
```

```csv
generation,average_trait
0,1.0261658948745924
1,1.020567999114006
...
```

Multi-trait with selection:

```bash
python3 simulate.py --num-traits 2 \
  --generations 3 --pop-size 10 \
  --init-means 0.0,1.0 \
  --init-stds 0.1,0.1 \
  --mutation-stds 0.05,0.05 \
  --optimum 0.0,1.0 \
  --seed 7 --output traits.csv
```

```csv
generation,trait0_avg,trait1_avg,speed_avg,size_avg
0,-0.014689099422148,0.9865920451175529,1.0416660239453939,1.0964995657780165
...
```

Offline plot:

```bash
python3 visualize.py traits.csv --output traits.png
python3 visualize.py traits.csv   # show window instead
```

## Usage

### Single-trait options

```bash
python3 simulate.py \
  --generations 100 --pop-size 100 \
  --mutation-std 0.05 --initial-mean 1.0 --initial-std 0.1 \
  --speed-mean 1.0 --speed-std 0.2 --speed-mut 0.05 \
  --size-mean 1.0 --size-std 0.2 --size-mut 0.05 \
  --output traits.csv --seed 42
```

`--speed-mut` / `--size-mut` default to `--mutation-std`.

### Multi-trait options

Set `--num-traits N` to enable. Then all four lists are required, each with
exactly `N` comma-separated floats:

```bash
--num-traits 2 \
--init-means 0.0,1.0 \
--init-stds 0.1,0.1 \
--mutation-stds 0.05,0.05 \
--optimum 0.0,1.0 \
--selection-strength 0.5
```

`--selection-strength 0` disables selection. Fitness is
`exp(-strength * squared-distance-to-optimum)` with weighted resampling.

### Space and movement

```bash
--position-std 0.05 --position-min 0.0 --position-max 10.0 \
--similarity-scale 0.5 --move-step 0.1
```

`similarity_scale=0` disables directed movement (only mutation jitter).

### Elements

```bash
--elements \
--element-rate 1.0 --element-radius 1.0 \
--element-boost 0.05 --element-lifetime 200 --element-max 1
```

`element_rate` is the expected spawns per generation. `element_max` is a hard
cap: no spawns while full. Negative `--element-boost` harms nearby creatures.
In `--animate`, elements are `X` markers colored by target trait (`tab10`)
with dotted radius circles; marker size encodes `|boost|`.

### Live visualization

```bash
python3 simulate.py --live-plot --interval 0.05
python3 simulate.py --animate --interval 0.05
python3 simulate.py --live-plot --animate --elements
```

- Single trait color: `viridis` on a widen-only fixed scale.
- Multi-trait with ≥3 traits: first 3 traits as RGB on per-channel
  widen-only scales; otherwise first trait as scalar.
- Marker area `15–120` from `size` on a widen-only scale.
- Axis is fixed to `[position_min, position_max]` plus 5% padding.
- Stats box: `gen / avg / std / spread / elements / speed / size`.
- When a plot window is open, the run continues until the window is closed
  (`--generations` is ignored). `generations=None` in the API requires
  `stop_when`.

## Output format

- Single-trait: `generation,average_trait`
- Multi-trait: `generation,trait0_avg,...,trait{N-1}_avg,speed_avg,size_avg`

`*.csv` and `*.png` are git-ignored outputs (see `.gitignore`).

## Python API

```python
from simulate import simulate, simulate_multi

avg = simulate(pop_size=100, generations=100,
               mutation_std=0.05, initial_mean=1.0, initial_std=0.1)

per_trait = simulate_multi(pop_size=100, generations=100, num_traits=2,
                           init_means=[0.0, 1.0], init_stds=[0.1, 0.1],
                           mutation_stds=[0.05, 0.05],
                           optimum=[0.0, 1.0], selection_strength=0.0)
```

Optional callbacks per generation:

```python
simulate(..., plot_callback=lambda gen, averages: ...,
         anim_callback=lambda gen, population: ...,
         stats_callback=lambda gen, stats: ...,
         stop_when=lambda: False)
```

With `elements=True`, `stats` includes `elements`, and `anim_callback`
receives `(gen, population, elements)`.

## Tests

51 tests, all passing:

```bash
python3 -m pytest -q
```

Covers reproduction/inheritance, position bounds/midpoints, movement,
selection, elements, stats formatting, CSV output, seed reproducibility,
and `--live-plot` / `--animate` callback wiring (mocked `matplotlib`).

## Project structure

```text
simulate.py       # simulation, CLI, live plot / animation
visualize.py      # offline CSV -> matplotlib line chart
test_simulate.py  # simulation behavior tests
test_callbacks.py # callback / CLI wiring tests
traits.csv        # example multi-trait output (ignored by git)
demo*.csv, test_*.csv, *.png  # example outputs (ignored by git)
```
