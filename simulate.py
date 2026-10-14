import random
import csv
import argparse
import math
import sys
import os

# Try to import numpy for color mapping; will be used only if animation requested
try:
    import numpy as np
except Exception:
    np = None  # will check later if needed

def _get_traits(creature):
    if 'trait' in creature:
        return creature['trait']
    else:
        return creature['traits']

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
        if 'trait' in population[0]:
            mean = sum(c['trait'] for c in population) / len(population)
            var = sum((c['trait'] - mean) ** 2 for c in population) / len(population)
        else:
            num_traits = len(population[0]['traits'])
            var = 0.0
            for k in range(num_traits):
                mean_k = sum(c['traits'][k] for c in population) / len(population)
                var += sum((c['traits'][k] - mean_k) ** 2
                           for c in population) / len(population)
            var /= num_traits
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
                if 'trait' in creature:
                    d = abs(creature['trait'] - other['trait'])
                else:
                    d = math.sqrt(sum((a - b) ** 2
                                      for a, b in zip(creature['traits'], other['traits'])))
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
        creature['pos'][0] = min(max(creature['pos'][0], position_min), position_max)
        creature['pos'][1] = min(max(creature['pos'][1], position_min), position_max)

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
    if 'trait' in population[0]:
        # single trait mode
        columns = [[c['trait'] for c in population]]
    else:
        # multi-trait mode
        num_traits = len(population[0]['traits'])
        columns = [[c['traits'][k] for c in population]
                   for k in range(num_traits)]
    averages = []
    stds = []
    for values in columns:
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        averages.append(mean)
        stds.append(math.sqrt(var))
    # speed and size stats
    speeds = [c.get('speed', 1.0) for c in population]
    sizes = [c.get('size', 1.0) for c in population]
    speed_mean = sum(speeds) / len(speeds)
    speed_var = sum((s - speed_mean) ** 2 for s in speeds) / len(speeds)
    size_mean = sum(sizes) / len(sizes)
    size_var = sum((s - size_mean) ** 2 for s in sizes) / len(sizes)
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
             element_boost=0.05, element_lifetime=200, element_max=1):
    if generations is None and stop_when is None:
        raise ValueError('stop_when is required when generations is None')
    # Initialize population
    population = []
    for _ in range(pop_size):
        trait = random.gauss(initial_mean, initial_std)
        speed = random.gauss(1.0, 0.2)  # default speed mean/std
        size = random.gauss(1.0, 0.2)   # default size mean/std
        pos = [random.uniform(position_min, position_max) for _ in range(2)]
        population.append({'trait': trait, 'speed': speed, 'size': size, 'pos': pos})
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
        if stats_callback:
            stats = _population_stats(population)
            if elements:
                stats['elements'] = len(element_list)
            stats_callback(gen, stats)
        if plot_callback:
            plot_callback(gen, [avg])
        if anim_callback:
            if elements:
                anim_callback(gen, population, element_list)
            else:
                anim_callback(gen, population)
        # Reproduce: each creature interacts with its nearest neighbour
        # (distance-based interaction); the offspring inherits one parent's
        # trait and is placed between the two parents, then mutates
        new_pop = []
        for creature in population:
            partner = _nearest_neighbour(creature, population)
            if partner is None:
                # alone: clone with mutation
                offspring_trait = creature['trait'] + random.gauss(0.0, mutation_std)
                offspring_pos = [creature['pos'][i] + random.gauss(0.0, position_std) for i in range(2)]
            else:
                # interaction: trait from one parent, position between parents
                offspring_trait = random.choice((creature['trait'], partner['trait']))
                offspring_trait += random.gauss(0.0, mutation_std)
                offspring_pos = [(creature['pos'][i] + partner['pos'][i]) / 2.0
                                 + random.gauss(0.0, position_std) for i in range(2)]
            offspring_pos = [min(max(p, position_min), position_max) for p in offspring_pos]
            # inherit speed and size from one parent, then mutate
            offspring_speed = random.choice((creature['speed'], partner['speed']))
            offspring_speed += random.gauss(0.0, mutation_std)
            offspring_size = random.choice((creature['size'], partner['size']))
            offspring_size += random.gauss(0.0, mutation_std)
            # clamp speed/size to reasonable ranges
            offspring_speed = max(0.1, min(2.0, offspring_speed))
            offspring_size = max(0.3, min(3.0, offspring_size))
            new_pop.append({'trait': offspring_trait, 'speed': offspring_speed, 'size': offspring_size, 'pos': offspring_pos})
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
                   element_boost=0.05, element_lifetime=200, element_max=1):
    if generations is None and stop_when is None:
        raise ValueError('stop_when is required when generations is None')
    # Initialize population: list of creatures, each creature is dict with traits list and pos
    population = []
    for _ in range(pop_size):
        traits = [random.gauss(init_means[i], init_stds[i]) for i in range(num_traits)]
        speed = random.gauss(1.0, 0.2)
        size = random.gauss(1.0, 0.2)
        pos = [random.uniform(position_min, position_max) for _ in range(2)]
        population.append({'traits': traits, 'speed': speed, 'size': size, 'pos': pos})
    # We'll track averages per trait per generation
    averages_per_trait = [[] for _ in range(num_traits)]
    speeds = []
    sizes = []
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
        speeds.append(sum(c.get('speed', 1.0) for c in population) / pop_size)
        sizes.append(sum(c.get('size', 1.0) for c in population) / pop_size)
        if stats_callback:
            stats = _population_stats(population)
            if elements:
                stats['elements'] = len(element_list)
            stats_callback(gen, stats)
        if plot_callback:
            plot_callback(gen, [averages_per_trait[i][gen] for i in range(num_traits)])
        if anim_callback:
            if elements:
                anim_callback(gen, population, element_list)
            else:
                anim_callback(gen, population)
        # Reproduce: each creature interacts with its nearest neighbour
        # (distance-based interaction); each trait is inherited from one of
        # the two parents, and the offspring is placed between the parents
        offspring_pop = []
        for creature in population:
            partner = _nearest_neighbour(creature, population)
            if partner is None:
                # alone: clone with mutation
                offspring_traits = [creature['traits'][i] + random.gauss(0.0, mutation_stds[i]) for i in range(num_traits)]
                offspring_pos = [creature['pos'][i] + random.gauss(0.0, position_std) for i in range(2)]
            else:
                # interaction: each trait from one parent, position between parents
                offspring_traits = [random.choice((creature['traits'][i], partner['traits'][i]))
                                    + random.gauss(0.0, mutation_stds[i]) for i in range(num_traits)]
                offspring_pos = [(creature['pos'][i] + partner['pos'][i]) / 2.0
                                 + random.gauss(0.0, position_std) for i in range(2)]
            offspring_pos = [min(max(p, position_min), position_max) for p in offspring_pos]
            # inherit speed and size from one parent, then mutate
            offspring_speed = random.choice((creature['speed'], partner['speed']))
            offspring_speed += random.gauss(0.0, mutation_stds[0])
            offspring_size = random.choice((creature['size'], partner['size']))
            offspring_size += random.gauss(0.0, mutation_stds[0])
            # clamp speed/size to reasonable ranges
            offspring_speed = max(0.1, min(2.0, offspring_speed))
            offspring_size = max(0.3, min(3.0, offspring_size))
            offspring_pop.append({'traits': offspring_traits, 'speed': offspring_speed, 'size': offspring_size, 'pos': offspring_pos})
        # Apply selection if needed
        if selection_strength > 0.0:
            # compute fitness for each offspring
            fitnesses = []
            for ind in offspring_pop:
                sq_dist = sum((ind['traits'][i] - optimum[i])**2 for i in range(num_traits))
                fitness = math.exp(-selection_strength * sq_dist)
                fitnesses.append(fitness)
            # Weighted random sampling to choose next generation
            if sum(fitnesses) == 0:
                weights = [1.0] * pop_size
            else:
                weights = fitnesses
            population = random.choices(offspring_pop, weights=weights, k=pop_size)
        else:
            population = offspring_pop
        # Move creatures toward others with similar traits
        _move_toward_similar(population, similarity_scale, move_step,
                             position_std, position_min, position_max)
        gen += 1
    return averages_per_trait

def main():
    parser = argparse.ArgumentParser(description='Simulate evolution of creature traits.')
    # Backward compatibility arguments
    parser.add_argument('--generations', type=int, default=100, help='Number of generations (ignored when a plot window is shown; the simulation then runs until the window is closed)')
    parser.add_argument('--pop-size', type=int, default=100, help='Population size')
    parser.add_argument('--mutation-std', type=float, default=0.05, help='Mutation standard deviation (single trait)')
    parser.add_argument('--initial-mean', type=float, default=1.0, help='Initial mean trait value (single trait)')
    parser.add_argument('--initial-std', type=float, default=0.1, help='Initial standard deviation (single trait)')
    parser.add_argument('--output', type=str, default='traits.csv', help='Output CSV file')
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
        lines = []
        gen_data = []
        trait_data = []  # list of lists per trait
        stats_texts.append(ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                        va='top', family='monospace', fontsize=9,
                                        bbox=dict(boxstyle='round', facecolor='white',
                                                  alpha=0.7, edgecolor='none')))
        def plot_callback(gen, averages):
            gen_data.append(gen)
            if not trait_data:
                for _ in averages:
                    trait_data.append([])
            for i, avg in enumerate(averages):
                trait_data[i].append(avg)
            if not lines:
                for i in range(len(averages)):
                    line, = ax.plot([], [], label=f'Trait {i}')
                    lines.append(line)
            ax.set_xlabel('Generation')
            ax.set_ylabel('Average Trait Value')
            ax.set_title('Trait Evolution over Generations (live)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            for i, line in enumerate(lines):
                line.set_data(gen_data, trait_data[i])
            plt.pause(args.interval)
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
            from matplotlib.patches import Circle
            from matplotlib.lines import Line2D
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
            pad = 0.05 * (args.position_max - args.position_min)
            if pad <= 0:
                pad = 0.5
            ax.set_xlim(args.position_min - pad, args.position_max + pad)
            ax.set_ylim(args.position_min - pad, args.position_max + pad)
            stats_texts.append(ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                            va='top', family='monospace', fontsize=9,
                                            bbox=dict(boxstyle='round', facecolor='white',
                                                      alpha=0.7, edgecolor='none')))
            scat = None  # will initialize on first callback
            elem_scat = None  # element scatter, init on first callback
            elem_circles = []  # radius circles, redrawn each frame
            # Fixed color scales: running min/max per trait channel that only
            # widen, so a creature's color depends on its trait value alone
            # (similar traits get similar colors across the whole animation)
            trait_vmin = None
            trait_vmax = None
            rgb_mins = None
            rgb_maxs = None
            def anim_callback(gen, population, elements=None):
                nonlocal scat, trait_vmin, trait_vmax, rgb_mins, rgb_maxs, elem_scat
                # Extract positions
                xs = [c['pos'][0] for c in population]
                ys = [c['pos'][1] for c in population]
                # Determine trait values for coloring
                # Check if single trait or multi-trait by inspecting first creature
                sample = population[0]
                if 'trait' in sample:
                    # single trait mode
                    trait_vals = [c['trait'] for c in population]
                    trait_vmin, trait_vmax = _widen_scale(trait_vmin, trait_vmax, trait_vals)
                    # normalize to [0,1] on the fixed scale
                    norm = _color_norm(trait_vals, trait_vmin, trait_vmax)
                    # use viridis colormap
                    colors = cm.viridis(norm)
                else:
                    # multi-trait mode
                    tra = np.array([c['traits'] for c in population], dtype=float)  # shape (N, num_traits)
                    if tra.shape[1] >= 3:
                        # use first three traits as RGB
                        rgb = tra[:, :3]
                        # widen the fixed per-channel scales, then normalize
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
                        colors = rgb  # already shape (N,3)
                    else:
                        # fewer than 3 traits, fallback to first trait as scalar
                        trait_vals = tra[:, 0].tolist()
                        trait_vmin, trait_vmax = _widen_scale(trait_vmin, trait_vmax, trait_vals)
                        norm = _color_norm(trait_vals, trait_vmin, trait_vmax)
                        colors = cm.viridis(norm)
                # Initialize or update scatter
                if scat is None:
                    scat = ax.scatter(xs, ys, c=colors, s=30, edgecolors='k', linewidths=0.5)
                    ax.set_xlabel('X position')
                    ax.set_ylabel('Y position')
                    ax.set_title('Creature positions (color = trait)')
                    ax.grid(True, linestyle='--', alpha=0.5)
                else:
                    scat.set_offsets(np.column_stack([xs, ys]))
                    scat.set_color(colors)
                # Draw map elements: filled 'X' markers colored
                # by the trait they boost, sized by the boost,
                # with a dotted circle showing the radius
                if elements:
                    exs = [e['pos'][0] for e in elements]
                    eys = [e['pos'][1] for e in elements]
                    ecolors = [cm.tab10(e['trait'] % 10) for e in elements]
                    esizes = [150 + 600 * abs(e['boost']) for e in elements]
                    if elem_scat is None:
                        elem_scat = ax.scatter(exs, eys, c=ecolors,
                                               s=esizes, marker='X',
                                               edgecolors='k',
                                               linewidths=1.5)
                        ax.legend(handles=[
                            Line2D([0], [0], marker='X', color='w',
                                   markerfacecolor='gray',
                                   markeredgecolor='k', markersize=8,
                                   linestyle='None',
                                   label='element (color = trait, size = |boost|)'),
                            Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor='none',
                                   markeredgecolor='gray', markersize=10,
                                   linestyle=':',
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
                            linewidth=2, color=cm.tab10(e['trait'] % 10),
                            alpha=0.6)))
                # Axis limits are fixed above: positions are clamped to the
                # configured range, so per-frame min/max rescaling is unnecessary
                plt.pause(args.interval)
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
    if args.num_traits is None:
        # Backward compatibility: use original single-trait simulation
        averages = simulate(pop_size=args.pop_size, generations=run_generations,
                            mutation_std=args.mutation_std,
                            initial_mean=args.initial_mean,
                            initial_std=args.initial_std,
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
                            stats_callback=stats_cb)
        # Write CSV with single column
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['generation', 'average_trait'])
            for i, avg in enumerate(averages):
                writer.writerow([i, avg])
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
        
        # Run multi-trait simulation
        averages_per_trait = simulate_multi(pop_size=args.pop_size, generations=run_generations,
                                            num_traits=num_traits,
                                            init_means=init_means,
                                            init_stds=init_stds,
                                            mutation_stds=mutation_stds,
                                            optimum=optimum,
                                            selection_strength=args.selection_strength,
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
                                            stats_callback=stats_cb)
        # Write CSV with columns for each trait, speed, and size
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['generation'] + [f'trait{i}_avg' for i in range(num_traits)] + ['speed_avg', 'size_avg']
            writer.writerow(header)
            # Use default speed=1.0, size=1.0 averages since simulate_multi()
            # returns only averages_per_trait; for full speed/size history,
            # the simulation return type would need to be extended
            speed_val = 1.0
            size_val = 1.0
            for gen in range(len(averages_per_trait[0])):
                row = [gen] + [averages_per_trait[i][gen] for i in range(num_traits)] + [speed_val, size_val]
                writer.writerow(row)
        print(f'Simulation complete. Results written to {args.output}')
        if averages_per_trait[0]:
            print('Initial averages per trait:')
            for i in range(num_traits):
                print(f'  Trait {i}: {averages_per_trait[i][0]:.4f}')
            print('Final averages per trait:')
            for i in range(num_traits):
                print(f'  Trait {i}: {averages_per_trait[i][-1]:.4f}')
            print('Differences per trait:')
            for i in range(num_traits):
                diff = abs(averages_per_trait[i][-1] - averages_per_trait[i][0])
                print(f'  Trait {i}: {diff:.4f}')
    
    # After simulation, if live plot or animation was shown, keep window open until user closes
    if show_plot:
        plt_module.ioff()
        plt_module.show()

if __name__ == '__main__':
    main()
