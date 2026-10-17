"""Config-first simulation settings: load and validate `sim.toml`.

The TOML file is the canonical home for everything evolvable. Remaining
CLI flags only override or toggle; when a flag is absent the config
decides, and when the config omits a key the built-in default applies.

Precedence:   CLI flag  >  sim.toml value  >  built-in default
"""

import tomllib

DEFAULT_OUTPUT = 'traits.csv'

TRAIT_PROPERTIES = frozenset({
    'move_speed', 'steer', 'repel', 'survival_decay',
    'element_resist', 'repro_boost', 'life_expect',
    'cluster_bias',
})

TRAITS_KEYS = ('init_means', 'init_stds', 'mutation_stds', 'mutation_std',
               'optimum', 'selection_strength', 'properties')


class ConfigError(ValueError):
    """Invalid simulation configuration (file, section, or value)."""


def _bad(name, expected):
    raise ConfigError(f'{name} must be {expected}')


def _int_value(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _bad(name, f'an integer >= {minimum}')
    return value


def _num_value(value, name, allow_none=False):
    if allow_none and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _bad(name, 'a number')
    return float(value)


def _bool_value(value, name):
    if not isinstance(value, bool):
        _bad(name, 'true or false')
    return value


def _check_list(value, name, length=None):
    """Validate a list of numbers; optionally enforce exact length."""
    if not isinstance(value, list) or not value:
        _bad(name, 'a non-empty list of numbers')
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            _bad(name, f'a list of numbers (bad entry: {v!r})')
    if length is not None and len(value) != length:
        _bad(name, f'{length} values, got {len(value)}')
    return [float(v) for v in value]


def _section(section_data, section_name, defaults, validators):
    """Validate one scalar section; yields (key, validated_value)."""
    if not isinstance(section_data, dict):
        _bad(f'[{section_name}]', 'a table')
    unknown = [k for k in section_data if k not in validators]
    if unknown:
        raise ConfigError(f'unknown keys in [{section_name}]: '
                          f'{", ".join(sorted(unknown))}')
    for key, validator in validators.items():
        yield key, validator(section_data.get(key, defaults[key]),
                             f'[{section_name}].{key}')


_RUN = {
    'pop_size':    lambda v, n: _int_value(v, n, 1),
    'generations': lambda v, n: _int_value(v, n, 1),
}
_SPACE = {
    'position_std':     lambda v, n: _num_value(v, n),
    'position_min':     lambda v, n: _num_value(v, n),
    'position_max':     lambda v, n: _num_value(v, n),
    'similarity_scale': lambda v, n: _num_value(v, n),
    'move_step':        lambda v, n: _num_value(v, n),
}
_BODY = {
    'speed_mean':     lambda v, n: _num_value(v, n),
    'speed_std':      lambda v, n: _num_value(v, n),
    'speed_mutation': lambda v, n: _num_value(v, n, allow_none=True),
    'size_mean':      lambda v, n: _num_value(v, n),
    'size_std':       lambda v, n: _num_value(v, n),
    'size_mutation':  lambda v, n: _num_value(v, n, allow_none=True),
}
_ELEMENTS = {
    'enabled':  _bool_value,
    'rate':     lambda v, n: _num_value(v, n),
    'radius':   lambda v, n: _num_value(v, n),
    'boost':    lambda v, n: _num_value(v, n),
    'lifetime': lambda v, n: _int_value(v, n, 1),
    'max':      lambda v, n: _int_value(v, n, 0),
}
_SCALAR_SECTIONS = {
    'run': ({'pop_size': 100, 'generations': 100}, _RUN),
    'space': ({
        'position_std': 0.05, 'position_min': 0.0, 'position_max': 10.0,
        'similarity_scale': 0.5, 'move_step': 0.1,
    }, _SPACE),
    'body': ({
        'speed_mean': 1.0, 'speed_std': 0.2, 'speed_mutation': None,
        'size_mean': 1.0, 'size_std': 0.2, 'size_mutation': None,
    }, _BODY),
    'elements': ({
        'enabled': False, 'rate': 1.0, 'radius': 1.0, 'boost': 0.05,
        'lifetime': 200, 'max': 1,
    }, _ELEMENTS),
}


def _check_traits(section_data):
    """Validate [traits]; returns (num_traits, validated dict).

    Trait count derives from len(init_means); all other lists must match.
    selection_strength > 0 requires an optimum. Unknown keys are
    rejected up front.
    """
    unknown = [k for k in section_data if k not in TRAITS_KEYS]
    if unknown:
        raise ConfigError('unknown keys in [traits]: '
                          f'{", ".join(sorted(unknown))} '
                          f'(valid: {", ".join(TRAITS_KEYS)})')
    if 'init_means' not in section_data:
        raise ConfigError('[traits].init_means is required '
                          '(its length defines the number of traits)')
    means = _check_list(section_data['init_means'], '[traits].init_means')
    num_traits = len(means)

    defaults = [0.1] * num_traits
    if 'mutation_std' in section_data:
        # shorthand: one std for all traits
        if 'mutation_stds' in section_data:
            raise ConfigError('[traits]: use either mutation_std or '
                              'mutation_stds, not both')
        one = _num_value(section_data['mutation_std'],
                         '[traits].mutation_std')
        mutation_stds = [one] * num_traits
    else:
        mutation_stds = _check_list(
            section_data.get('mutation_stds',
                             [section_data.get('mutation_std', 0.05)]
                             * num_traits),
            '[traits].mutation_stds', num_traits)

    optimum = section_data.get('optimum')
    strength = _num_value(section_data.get('selection_strength', 0.0),
                          '[traits].selection_strength')
    if strength < 0.0:
        _bad('[traits].selection_strength', '>= 0')
    if strength > 0.0 and optimum is None:
        raise ConfigError('[traits].optimum is required when '
                          '[traits].selection_strength > 0')

    props = section_data.get('properties', ['none'] * num_traits)
    if not isinstance(props, list) or len(props) != num_traits:
        raise ConfigError(f'[traits].properties must be a list of '
                          f'{num_traits} property names (one per trait, '
                          f'"none" allowed)')
    for p in props:
        if not isinstance(p, str):
            _bad('[traits].properties entries', 'strings')
        if p != 'none' and p not in TRAIT_PROPERTIES:
            raise ConfigError(
                f'unknown trait property {p!r} '
                f'(valid: none, {", ".join(sorted(TRAIT_PROPERTIES))})')

    validated = {
        'init_means': means,
        'init_stds': _check_list(
            section_data.get('init_stds', defaults),
            '[traits].init_stds', num_traits),
        'mutation_stds': mutation_stds,
        'optimum': (None if optimum is None
                    else _check_list(optimum, '[traits].optimum',
                                     num_traits)),
        'selection_strength': strength,
        'properties': props,
    }
    return num_traits, validated


def load_config(path):
    """Load and validate a TOML config file into a flat dict.

    Raises ConfigError with a precise key-path message on any problem.
    """
    try:
        with open(path, 'rb') as f:
            raw = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f'config file not found: {path}') from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f'invalid TOML in {path}: {e}') from None

    valid_sections = set(_SCALAR_SECTIONS) | {'traits'}
    unknown_sections = [k for k in raw if k not in valid_sections]
    if unknown_sections:
        raise ConfigError(f'unknown sections in {path}: '
                          f'{", ".join(sorted(unknown_sections))} '
                          f'(valid: {", ".join(sorted(valid_sections))})')

    resolved = {'traits': None, 'output': DEFAULT_OUTPUT}
    for section_name, (defaults, validators) in _SCALAR_SECTIONS.items():
        section_data = raw.get(section_name, {})
        for key, value in _section(section_data, section_name,
                                    defaults, validators):
            resolved[key] = value

    if 'traits' in raw:
        _, resolved['traits'] = _check_traits(raw['traits'])

    if resolved['position_max'] <= resolved['position_min']:
        raise ConfigError('[space] position_max must be > position_min')
    return resolved


def template() -> str:
    """A fully commented starter sim.toml."""
    return '''# sim.toml - simulation configuration (TOML)
# Remaining CLI flags override values here; missing keys use the
# defaults shown (commented = optional).

[run]
pop_size = 100        # creatures per generation (>= 1)
generations = 300     # cap; with --live-plot/--animate and a visible
                      # window the run ends when the window closes
# seed = 42           # reproducible run (also a CLI flag)
# output = "traits.csv"

# One entry per trait; the length of init_means defines the trait count.
# A single value = single-trait mode.
[traits]
init_means    = [0.0, 1.0, 2.0]     # starting population mean per trait
init_stds     = [0.1, 0.1, 0.1]     # starting spread per trait
mutation_stds = [0.05, 0.05, 0.05]  # per-trait mutation rate
# mutation_std = 0.05               # shorthand: same rate for all traits
# optimum = [0.0, 1.0, 2.0]         # selection target; required when
                                    # selection_strength > 0
selection_strength = 0.0            # 0.0 = pure drift
# Per-trait behavior: what each trait *does* to the creature.
# Valid: none, move_speed, steer, repel, survival_decay,
#        element_resist, repro_boost, life_expect, cluster_bias
properties = ["move_speed", "steer", "repel"]

[body]                  # evolvable body attributes (not traits)
speed_mean = 1.0        # movement multiplier
speed_std  = 0.2
# speed_mutation = 0.05 # defaults to mutation rate of trait 0
size_mean = 1.0         # marker size / personal space
size_std  = 0.2
# size_mutation = 0.05  # defaults to mutation rate of trait 0

[space]
position_std     = 0.05 # positional jitter per generation
position_min     = 0.0
position_max     = 10.0
similarity_scale = 0.5  # similar-trait attract radius (pop std devs)
move_step        = 0.1  # fraction of distance moved per generation

[elements]
enabled  = true         # map elements boost nearby creatures' traits
rate     = 1.0          # expected new elements per generation
radius   = 1.0          # effect radius
boost    = 0.05         # gain at center, linear falloff to edge
lifetime = 200          # generations an element persists
max      = 5            # hard cap on simultaneous elements
'''
