"""Core evolution model: reproduction, movement, selection, elements, stats.

No matplotlib / argparse here. Split out of `simulate.py` so the model can
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


def _prop_value(creature, trait_properties, name):
    """First trait value whose property is `name`, else None.

    trait_properties is a per-trait-index list of property names
    (e.g. ['move_speed', 'steer']); None disables all hooks.
    """
    if not trait_properties:
        return None
    for i, prop in enumerate(trait_properties):
        if prop == name:
            return float(creature['traits'][i])
    return None


def _move_toward_similar(population, similarity_scale, move_step, position_std,
                         position_min, position_max, trait_properties=None):
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

    Optional trait-property hooks (trait_properties maps trait index to
    a property name; a trait whose property is set drives behavior):
      move_speed     extra displacement multiplier max(v, 0)
      steer          extra move_step * v toward the similar centroid
                     (v < 0 flees it); no anchor = no steering
      repel          personal-space radius multiplier max(v, 0)
                     so v=1 doubles the baseline spacing demand
      cluster_bias   scales the attraction contribution by
                     clamp(|v|, 0, 1)
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
    # Per-creature personal-space radius (property-scaled); each pair
    # keeps the larger of the two creatures' demands.
    repel_radii = []
    for creature in population:
        v_repel = _prop_value(creature, trait_properties, 'repel')
        repel_radii.append(repel_radius if v_repel is None
                           else repel_radius * max(v_repel, 0.0))
    displacements = []
    for i, creature in enumerate(population):
        dx = 0.0
        dy = 0.0
        rx = 0.0
        ry = 0.0
        speed = creature.get('speed', 1.0)
        size = creature.get('size', 1.0)
        v_speed = _prop_value(creature, trait_properties, 'move_speed')
        v_steer = _prop_value(creature, trait_properties, 'steer')
        v_cluster = _prop_value(creature, trait_properties, 'cluster_bias')
        # attraction gain from cluster_bias
        cluster_gain = (min(abs(v_cluster), 1.0)
                        if v_cluster is not None else 1.0)
        if similarity_scale > 0.0:
            weight_sum = 0.0
            centroid_x = 0.0
            centroid_y = 0.0
            for j, other in enumerate(population):
                if j == i:
                    continue
                d = _trait_distance(creature, other)
                if d > cutoff:
                    # dissimilar: keep personal space instead of attracting
                    d_pos = math.dist(creature['pos'], other['pos'])
                    pair_radius = max(repel_radii[i], repel_radii[j])
                    if 0.0 < d_pos < pair_radius:
                        f = (pair_radius - d_pos) / pair_radius
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
                dx = move_step * attraction * cluster_gain * (
                    centroid_x / weight_sum - creature['pos'][0])
                dy = move_step * attraction * cluster_gain * (
                    centroid_y / weight_sum - creature['pos'][1])
            # cap the total repulsion at one move_step
            repel_mag = math.hypot(rx, ry)
            if repel_mag > 1.0:
                rx /= repel_mag
                ry /= repel_mag
            dx += move_step * rx
            dy += move_step * ry
            # steer: extra directed step toward (v > 0) or away (v < 0)
            # from the weighted centroid of similar creatures
            if v_steer is not None and weight_sum > 0.0:
                to_cx = centroid_x / weight_sum - creature['pos'][0]
                to_cy = centroid_y / weight_sum - creature['pos'][1]
                norm = math.hypot(to_cx, to_cy)
                if norm > 1e-12:
                    dx += move_step * v_steer * to_cx / norm
                    dy += move_step * v_steer * to_cy / norm
        # scale displacement by creature's speed
        dx *= speed
        dy *= speed
        # scale by the move_speed property (if present)
        if v_speed is not None:
            dx *= max(v_speed, 0.0)
            dy *= max(v_speed, 0.0)
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
                     element_max, trait_properties=None):
    """Age map elements, spawn new ones, and boost creatures near them.

    Each element raises the trait it targets for creatures within its
    radius; the boost is strongest at the center and falls off
    linearly to zero at the edge. Elements expire after `lifetime`
    generations. The cap is a hard limit: while `element_max`
    elements exist, no new ones spawn, so elements are never
    evicted early.

    Creatures carrying the element_resist property scale the boost by
    (1 - clamp(|v|, 0, 1)): v=0 feels the full boost, v=1 is immune.
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
        v_resist = _prop_value(creature, trait_properties, 'element_resist')
        for element in elements:
            d = math.dist(creature['pos'], element['pos'])
            if d < element['radius']:
                amount = element['boost'] * (1.0 - d / element['radius'])
                if v_resist is not None:
                    amount *= 1.0 - min(abs(v_resist), 1.0)
                if amount == 0.0:
                    continue
                if 'trait' in creature:
                    creature['trait'] += amount
                else:
                    creature['traits'][element['trait']] += amount


def _decay_traits(population, trait_properties):
    """survival_decay upkeep: each assigned trait shrinks every generation.

    Decay rate is proportional to the trait's own magnitude
    (rate = min(0.5, |v| * 0.1)), so large values are costlier to
    maintain: value *= (1 - rate). Self-referential by design — a
    runaway decay trait prices itself out.
    """
    if not trait_properties:
        return
    decay_idx = [i for i, p in enumerate(trait_properties)
                 if p == 'survival_decay']
    if not decay_idx:
        return
    for creature in population:
        for i in decay_idx:
            v = creature['traits'][i]
            decay = min(0.5, abs(v) * 0.1)
            creature['traits'][i] = v * (1.0 - decay)


def _apply_mortality(population, trait_properties, position_std,
                     position_min, position_max, mutation_stds,
                     speed_mut, size_mut):
    """life_expect mortality: probability of dying each generation.

    Death probability = clamp(1 / (|v| * 20), 0, 0.5), so v=1 means a
    5% chance per generation, larger values live longer, and the floor
    caps the death rate. A dead creature is replaced (population size
    constant) by a mutated clone of its nearest neighbour, so dying
    hands your slot to the neighborhood's winning lineage.
    """
    if not trait_properties:
        return False
    life_idx = [i for i, p in enumerate(trait_properties)
                if p == 'life_expect']
    if not life_idx:
        return False
    died = False
    for idx, creature in enumerate(population):
        v = creature['traits'][life_idx[0]]
        death_p = min(0.5, 1.0 / (abs(v) * 20.0))
        if random.random() < death_p:
            died = True
            donor = _nearest_neighbour(creature, population) or creature
            offspring_traits = [t + random.gauss(0.0, m) for t, m in
                                zip(donor['traits'], mutation_stds)]
            population[idx] = {
                'traits': offspring_traits,
                **_inherit_body(donor, None, speed_mut, size_mut),
                'pos': _offspring_pos(donor, None, position_std,
                                      position_min, position_max),
            }
    return died


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
                   size_mean=1.0, size_std=0.2, size_mut=None,
                   trait_properties=None):
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
                             element_max, trait_properties)
        # compute average per trait
        averages = []
        for i in range(num_traits):
            trait_avg = sum(c['traits'][i] for c in population) / pop_size
            averages_per_trait[i].append(trait_avg)
            averages.append(trait_avg)
        _fire_callbacks(gen, averages,
                        population, element_list if elements else None,
                        stats_callback, plot_callback, anim_callback)
        # Reproduce: each creature interacts with its nearest neighbour
        # (distance-based interaction); each trait is inherited from one of
        # the two parents (repro_boost biases the choice toward the parent
        # whose value of that trait is larger in magnitude), and the
        # offspring is placed between the parents
        repro_idx = [i for i, p in enumerate(trait_properties or [])
                     if p == 'repro_boost']
        offspring_pop = []
        for creature in population:
            partner = _nearest_neighbour(creature, population)
            if partner is None:
                # alone: clone with mutation
                offspring_traits = [creature['traits'][i] + random.gauss(0.0, mutation_stds[i]) for i in range(num_traits)]
            else:
                # interaction: each trait from one parent, then mutate
                offspring_traits = []
                for i in range(num_traits):
                    own_v = creature['traits'][i]
                    partner_v = partner['traits'][i]
                    if i in repro_idx:
                        # weighted choice: larger magnitude = fitter parent
                        w_own, w_partner = abs(own_v), abs(partner_v)
                        total = w_own + w_partner
                        p_own = w_own / total if total > 0.0 else 0.5
                        parent_value = (own_v if random.random() < p_own
                                        else partner_v)
                    else:
                        parent_value = random.choice((own_v, partner_v))
                    offspring_traits.append(parent_value
                                            + random.gauss(0.0,
                                                           mutation_stds[i]))
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
        # Property hooks running after selection, before movement:
        # upkeep decay, then mortality (dead slots take the neighbor's line)
        _decay_traits(population, trait_properties)
        _apply_mortality(population, trait_properties, position_std,
                         position_min, position_max, mutation_stds,
                         speed_mut, size_mut)
        # Move creatures toward others with similar traits
        _move_toward_similar(population, similarity_scale, move_step,
                             position_std, position_min, position_max,
                             trait_properties)
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
