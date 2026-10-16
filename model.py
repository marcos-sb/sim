"""Core evolution model: reproduction, movement, selection, elements, stats.

No matplotlib / argparse here. Creamed off `simulate.py` so the model can
be tested and imported without pulling in visualization dependencies.
"""

import csv
import math
import random

# Bounds for the evolvable speed/size attributes.
SPEED_MIN, SPEED_MAX = 0.1, 2.0
SIZE_MIN, SIZE_MAX = 0.3, 3.0


def _clamp(value, lo, hi):
    """Clamp value into [lo, hi]."""
    return min(max(value, lo), hi)


def _mean_var(values):
    """Mean and population variance of values."""
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, var


def _trait_columns(population):
    """Per-trait value columns; works for single- and multi-trait populations."""
    if 'trait' in population[0]:
        return [[c['trait'] for c in population]]
    num_traits = len(population[0]['traits'])
    return [[c['traits'][k] for c in population] for k in range(num_traits)]


def _trait_distance(a, b):
    """Euclidean distance between two creatures in trait space."""
    if 'trait' in a:
        return abs(a['trait'] - b['trait'])
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a['traits'], b['traits'])))


def _random_pos(position_min, position_max):
    """A random position within the configured bounds."""
    return [random.uniform(position_min, position_max) for _ in range(2)]


def _random_body(speed_mean=1.0, speed_std=0.2, size_mean=1.0, size_std=0.2):
    """Speed/size for a newborn creature."""
    return {'speed': random.gauss(speed_mean, speed_std),
            'size': random.gauss(size_mean, size_std)}


def _inherit_ranged(own, partner_value, mutation_std, lo, hi):
    """Inherit a ranged attribute from one parent (clone if alone), mutate, clamp."""
    base = own if partner_value is None else random.choice((own, partner_value))
    return _clamp(base + random.gauss(0.0, mutation_std), lo, hi)


def _inherit_body(creature, partner, speed_mut, size_mut):
    """Offspring speed/size: inherit from one parent, mutate, clamp."""
    partner_speed = partner.get('speed', 1.0) if partner is not None else None
    partner_size = partner.get('size', 1.0) if partner is not None else None
    return {
        'speed': _inherit_ranged(creature.get('speed', 1.0), partner_speed,
                                 speed_mut, SPEED_MIN, SPEED_MAX),
        'size': _inherit_ranged(creature.get('size', 1.0), partner_size,
                                size_mut, SIZE_MIN, SIZE_MAX),
    }


def _offspring_pos(creature, partner, position_std, position_min, position_max):
    """Offspring position: midpoint of the parents (clone if alone), mutated, clamped."""
    if partner is None:
        raw = [p + random.gauss(0.0, position_std) for p in creature['pos']]
    else:
        raw = [(a + b) / 2.0 + random.gauss(0.0, position_std)
               for a, b in zip(creature['pos'], partner['pos'])]
    return [_clamp(p, position_min, position_max) for p in raw]


def _fire_callbacks(gen, averages, population, elements,
                    stats_callback, plot_callback, anim_callback):
    """Dispatch the per-generation stats/plot/animation callbacks.

    elements is the active element list, or None when disabled: stats then
    carry no element count and the animation gets no element overlay.
    """
    if stats_callback:
        stats = _population_stats(population)
        if elements is not None:
            stats['elements'] = len(elements)
        stats_callback(gen, stats)
    if plot_callback:
        plot_callback(gen, averages)
    if anim_callback:
        if elements is not None:
            anim_callback(gen, population, elements)
        else:
            anim_callback(gen, population)


def _nearest_neighbour(creature, population):
    """Return the closest other creature by Euclidean distance, or None if alone."""
    best = None
    best_dist = None
    for other in population:
        if other is creature:
            continue
        d = math.dist(creature['pos'], other['pos'])
        if best_dist is None or d < best_dist:
            best_dist = d
            best = other
    return best


def _move_toward_similar(population, similarity_scale, move_step, position_std,
                         position_min, position_max):
    """Move each creature toward the centroid of creatures with similar traits.

    The similarity scale is relative to the population's trait diversity:
    sigma = similarity_scale * (trait standard deviation), so "similar"
    means "within a few standard deviations" however diverse the
    population is. Creatures more than 2 sigma apart in trait space do
    not attract; instead they repel each other when spatially closer
    than the mean population spacing, which keeps the layout balanced
    instead of collapsing everything into one blob. Attraction strength
    is the total similarity to the other creatures (capped at 1), so
    creatures surrounded by similar traits move a full move_step toward
    their centroid while isolated creatures barely move. All creatures
    move simultaneously: displacements are computed from the same
    snapshot, so the result does not depend on iteration order.
    Positions are jittered by position_std and clamped to the bounds.

    Each creature's speed scales its displacement; its size scales its
    personal-space repulsion contribution.
    """
    # Similarity radius in trait units, scaled to the population's
    # trait diversity so the layout stays balanced whatever the
    # initial spread and mutation rates are
    sigma = 0.0
    if similarity_scale > 0.0:
        columns = _trait_columns(population)
        var = sum(_mean_var(col)[1] for col in columns) / len(columns)
        sigma = similarity_scale * math.sqrt(var)
        if sigma <= 0.0:
            # monomorphic population: only exactly equal traits match
            sigma = 1e-6
    cutoff = 2.0 * sigma
    # Personal space: dissimilar creatures repel when closer than the
    # mean spacing, so clusters stay separated and the layout stays
    # balanced at any population density
    repel_radius = 0.0
    if similarity_scale > 0.0 and position_max > position_min:
        repel_radius = (position_max - position_min) / math.sqrt(len(population))
    displacements = []
    for creature in population:
        dx = 0.0
        dy = 0.0
        rx = 0.0
        ry = 0.0
        speed = creature.get('speed', 1.0)
        size = creature.get('size', 1.0)
        if similarity_scale > 0.0:
            weight_sum = 0.0
            centroid_x = 0.0
            centroid_y = 0.0
            for other in population:
                if other is creature:
                    continue
                d = _trait_distance(creature, other)
                if d > cutoff:
                    # dissimilar: keep personal space instead of attracting
                    d_pos = math.dist(creature['pos'], other['pos'])
                    if 0.0 < d_pos < repel_radius:
                        f = (repel_radius - d_pos) / repel_radius
                        # size scales personal space repulsion
                        rx += f * (creature['pos'][0] - other['pos'][0]) / d_pos * size
                        ry += f * (creature['pos'][1] - other['pos'][1]) / d_pos * size
                    continue
                weight = math.exp(-(d * d) / (2.0 * sigma * sigma))
                weight_sum += weight
                centroid_x += weight * other['pos'][0]
                centroid_y += weight * other['pos'][1]
            if weight_sum > 0.0:
                attraction = min(1.0, weight_sum)
                dx = move_step * attraction * (centroid_x / weight_sum - creature['pos'][0])
                dy = move_step * attraction * (centroid_y / weight_sum - creature['pos'][1])
            # cap the total repulsion at one move_step
            repel_mag = math.hypot(rx, ry)
            if repel_mag > 1.0:
                rx /= repel_mag
                ry /= repel_mag
            dx += move_step * rx
            dy += move_step * ry
        # scale displacement by creature's speed
        dx *= speed
        dy *= speed
        displacements.append((dx, dy))
    for creature, (dx, dy) in zip(population, displacements):
        # random movement
        creature['pos'][0] += dx + random.gauss(0.0, position_std)
        creature['pos'][1] += dy + random.gauss(0.0, position_std)
        creature['pos'][0] = _clamp(creature['pos'][0], position_min, position_max)
        creature['pos'][1] = _clamp(creature['pos'][1], position_min, position_max)


def _widen_scale(vmin, vmax, values):
    """Widen a running [vmin, vmax] scale to include values (None = uninitialized)."""
    lo, hi = min(values), max(values)
    if vmin is None:
        return lo, hi
    return min(vmin, lo), max(vmax, hi)


def _color_norm(values, vmin, vmax):
    """Normalize values to [0, 1] against a fixed scale."""
    if vmax - vmin < 1e-12:
        return [0.0] * len(values)
    return [(v - vmin) / (vmax - vmin) for v in values]


def _population_stats(population):
    """Per-generation stats: per-trait averages and standard
    deviations, and the mean pairwise distance between creatures."""
    columns = _trait_columns(population)
    averages = []
    stds = []
    for values in columns:
        mean, var = _mean_var(values)
        averages.append(mean)
        stds.append(math.sqrt(var))
    # speed and size stats
    speed_mean, speed_var = _mean_var([c.get('speed', 1.0) for c in population])
    size_mean, size_var = _mean_var([c.get('size', 1.0) for c in population])
    spread = 0.0
    count = 0
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            spread += math.dist(population[i]['pos'], population[j]['pos'])
            count += 1
    if count > 0:
        spread /= count
    return {'averages': averages, 'stds': stds,
            'speed_avg': speed_mean, 'speed_std': math.sqrt(speed_var),
            'size_avg': size_mean, 'size_std': math.sqrt(size_var),
            'spread': spread}


def _format_stats(gen, stats):
    """One text block for the plot window: generation, per-trait
    averages and stds, and the mean pairwise distance."""
    avg = ', '.join(f'{v:.4f}' for v in stats['averages'])
    std = ', '.join(f'{v:.4f}' for v in stats['stds'])
    text = f'gen {gen}\navg {avg}\nstd {std}\nspread {stats["spread"]:.2f}'
    if 'elements' in stats:
        text += f'\nelements {stats["elements"]}'
    if 'speed_avg' in stats:
        text += f'\nspeed {stats["speed_avg"]:.4f}±{stats["speed_std"]:.4f}'
    if 'size_avg' in stats:
        text += f'\nsize {stats["size_avg"]:.4f}±{stats["size_std"]:.4f}'
    return text


def _update_elements(population, elements, num_traits, rate, radius,
                     boost, lifetime, position_min, position_max,
                     element_max):
    """Age map elements, spawn new ones, and boost creatures near them.

    Each element raises the trait it targets for creatures within its
    radius; the boost is strongest at the center and falls off
    linearly to zero at the edge. Elements expire after `lifetime`
    generations. The cap is a hard limit: while `element_max`
    elements exist, no new ones spawn, so elements are never
    evicted early.
    """
    for element in elements:
        element['lifetime'] -= 1
    elements[:] = [e for e in elements if e['lifetime'] > 0]
    # spawn: expected `rate` new elements per generation
    n = int(rate)
    if random.random() < rate - n:
        n += 1
    for _ in range(n):
        if len(elements) >= element_max:
            break  # at capacity: wait for an element to expire
        elements.append({
            'pos': [random.uniform(position_min, position_max)
                    for _ in range(2)],
            'trait': random.randrange(num_traits),
            'radius': radius,
            'boost': boost,
            'lifetime': lifetime,
        })
    # apply boosts with linear falloff inside the radius
    for creature in population:
        for element in elements:
            d = math.dist(creature['pos'], element['pos'])
            if d < element['radius']:
                amount = element['boost'] * (1.0 - d / element['radius'])
                if 'trait' in creature:
                    creature['trait'] += amount
                else:
                    creature['traits'][element['trait']] += amount


def simulate(pop_size=100, generations=100, mutation_std=0.05, initial_mean=1.0, initial_std=0.1,
             position_std=0.05, position_min=0.0, position_max=10.0,
             similarity_scale=0.5, move_step=0.1,
             plot_callback=None, anim_callback=None, stop_when=None, stats_callback=None,
             elements=False, element_rate=1.0, element_radius=1.0,
             element_boost=0.05, element_lifetime=200, element_max=1,
             speed_mean=1.0, speed_std=0.2, speed_mut=None,
             size_mean=1.0, size_std=0.2, size_mut=None):
    if generations is None and stop_when is None:
        raise ValueError('stop_when is required when generations is None')
    if pop_size < 1:
        raise ValueError('pop_size must be at least 1')
    if speed_mut is None:
        speed_mut = mutation_std
    if size_mut is None:
        size_mut = mutation_std
    # Initialize population
    population = []
    for _ in range(pop_size):
        population.append({
            'trait': random.gauss(initial_mean, initial_std),
            **_random_body(speed_mean, speed_std, size_mean, size_std),
            'pos': _random_pos(position_min, position_max),
        })
    averages = []
    element_list = []  # active map elements
    gen = 0
    while generations is None or gen < generations:
        if stop_when is not None and stop_when():
            break
        if elements:
            _update_elements(population, element_list, 1, element_rate,
                             element_radius, element_boost, element_lifetime,
                             position_min, position_max, element_max)
        avg = sum(c['trait'] for c in population) / len(population)
        averages.append(avg)
        _fire_callbacks(gen, [avg], population,
                        element_list if elements else None,
                        stats_callback, plot_callback, anim_callback)
        # Reproduce: each creature interacts with its nearest neighbour
        # (distance-based interaction); the offspring inherits one parent's
        # trait and is placed between the two parents, then mutates
        new_pop = []
        for creature in population:
            partner = _nearest_neighbour(creature, population)
            if partner is None:
                # alone: clone with mutation
                offspring_trait = creature['trait'] + random.gauss(0.0, mutation_std)
            else:
                # interaction: trait from one parent, then mutate
                offspring_trait = (random.choice((creature['trait'], partner['trait']))
                                   + random.gauss(0.0, mutation_std))
            new_pop.append({
                'trait': offspring_trait,
                **_inherit_body(creature, partner, speed_mut, size_mut),
                'pos': _offspring_pos(creature, partner, position_std,
                                      position_min, position_max),
            })
        population = new_pop
        # Move creatures toward others with similar traits
        _move_toward_similar(population, similarity_scale, move_step,
                             position_std, position_min, position_max)
        gen += 1
    return averages


def simulate_multi(pop_size=100, generations=100, num_traits=2, init_means=None, init_stds=None,
                   mutation_stds=None, optimum=None, selection_strength=0.0,
                   position_std=0.05, position_min=0.0, position_max=10.0,
                   similarity_scale=0.5, move_step=0.1,
                   plot_callback=None, anim_callback=None, stop_when=None, stats_callback=None,
                   elements=False, element_rate=1.0, element_radius=1.0,
                   element_boost=0.05, element_lifetime=200, element_max=1,
                   speed_mean=1.0, speed_std=0.2, speed_mut=None,
                   size_mean=1.0, size_std=0.2, size_mut=None):
    if generations is None and stop_when is None:
        raise ValueError('stop_when is required when generations is None')
    if pop_size < 1:
        raise ValueError('pop_size must be at least 1')
    if selection_strength > 0.0 and optimum is None:
        raise ValueError('optimum is required when selection_strength > 0')
    if speed_mut is None:
        speed_mut = mutation_stds[0]
    if size_mut is None:
        size_mut = mutation_stds[0]
    # Initialize population: list of creatures, each creature is dict with traits list and pos
    population = []
    for _ in range(pop_size):
        population.append({
            'traits': [random.gauss(init_means[i], init_stds[i]) for i in range(num_traits)],
            **_random_body(speed_mean, speed_std, size_mean, size_std),
            'pos': _random_pos(position_min, position_max),
        })
    # We'll track averages per trait per generation
    averages_per_trait = [[] for _ in range(num_traits)]
    element_list = []  # active map elements
    gen = 0
    while generations is None or gen < generations:
        if stop_when is not None and stop_when():
            break
        if elements:
            _update_elements(population, element_list, num_traits,
                             element_rate, element_radius, element_boost,
                             element_lifetime, position_min, position_max,
                             element_max)
        # compute average per trait
        for i in range(num_traits):
            trait_avg = sum(c['traits'][i] for c in population) / pop_size
            averages_per_trait[i].append(trait_avg)
        _fire_callbacks(gen, [averages_per_trait[i][gen] for i in range(num_traits)],
                        population, element_list if elements else None,
                        stats_callback, plot_callback, anim_callback)
        # Reproduce: each creature interacts with its nearest neighbour
        # (distance-based interaction); each trait is inherited from one of
        # the two parents, and the offspring is placed between the parents
        offspring_pop = []
        for creature in population:
            partner = _nearest_neighbour(creature, population)
            if partner is None:
                # alone: clone with mutation
                offspring_traits = [creature['traits'][i] + random.gauss(0.0, mutation_stds[i]) for i in range(num_traits)]
            else:
                # interaction: each trait from one parent, then mutate
                offspring_traits = [random.choice((creature['traits'][i], partner['traits'][i]))
                                    + random.gauss(0.0, mutation_stds[i]) for i in range(num_traits)]
            offspring_pop.append({
                'traits': offspring_traits,
                **_inherit_body(creature, partner, speed_mut, size_mut),
                'pos': _offspring_pos(creature, partner, position_std,
                                      position_min, position_max),
            })
        # Apply selection if needed
        if selection_strength > 0.0:
            # compute fitness for each offspring
            fitnesses = []
            for ind in offspring_pop:
                sq_dist = sum((ind['traits'][i] - optimum[i])**2 for i in range(num_traits))
                fitness = math.exp(-selection_strength * sq_dist)
                fitnesses.append(fitness)
            # Weighted random sampling to choose next generation.
            # Copy: choices() samples with replacement, and sharing one
            # dict between slots would move/boost that creature repeatedly.
            if sum(fitnesses) == 0:
                weights = [1.0] * pop_size
            else:
                weights = fitnesses
            population = [dict(ind, traits=list(ind['traits']), pos=list(ind['pos']))
                          for ind in random.choices(offspring_pop, weights=weights, k=pop_size)]
        else:
            population = offspring_pop
        # Move creatures toward others with similar traits
        _move_toward_similar(population, similarity_scale, move_step,
                             position_std, position_min, position_max)
        gen += 1
    return averages_per_trait


def _write_csv(path, header, rows):
    """Write a header plus rows to path as CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _print_trait_summary(averages_per_trait):
    """Print initial/final averages and drift for each trait series."""
    print('Initial averages per trait:')
    for i, series in enumerate(averages_per_trait):
        print(f'  Trait {i}: {series[0]:.4f}')
    print('Final averages per trait:')
    for i, series in enumerate(averages_per_trait):
        print(f'  Trait {i}: {series[-1]:.4f}')
    print('Differences per trait:')
    for i, series in enumerate(averages_per_trait):
        print(f'  Trait {i}: {abs(series[-1] - series[0]):.4f}')
