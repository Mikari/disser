from __future__ import division, absolute_import
from scipy.special import expit
import numpy
from copy import deepcopy
import random

from rep_ef.estimators._matrixnetapplier import MatrixnetApplier

__author__ = 'Egor Khairullin'


def apply_mx_trees(events, bias, trees):
        """
        :param events: numpy.array (or DataFrame) of shape [n_samples, n_features]
        :return: each time yields numpy.array predictions of shape [n_samples]
            which is output of a particular tree
        """
        # result of first iteration
        result = numpy.zeros(len(events), dtype=float) + bias

        # extending the data so the number of events is divisible by 8
        n_events = len(events)
        n_extended64 = (n_events + 7) // 8
        n_extended = n_extended64 * 8

        # using Fortran order (surprisingly doesn't seem to influence speed much)
        features = numpy.zeros([n_extended, events.shape[1]], dtype='float32', order='F')
        features[:n_events, :] = events

        for tree_features, tree_cuts, leaf_values in trees:
            leaf_indices = numpy.zeros(n_extended64, dtype='uint64')
            for tree_level, (feature, cut) in enumerate(zip(tree_features, tree_cuts)):
                leaf_indices |= (features[:, feature] > cut).view('uint64') << tree_level
            result += leaf_values[leaf_indices.view('uint8')[:n_events]]

        return result


def convert_lookup_index_to_bins(points_in_bins, lookup_indices):
    result = numpy.zeros([len(lookup_indices), len(points_in_bins)], dtype=float)
    lookup_indices = lookup_indices.copy()
    for i, points_in_variable in list(enumerate(points_in_bins))[::-1]:
        # print(points_in_variable)
        n_columns = len(points_in_variable)
        result[:, i] = points_in_variable[lookup_indices % n_columns]
        lookup_indices //= n_columns

    assert numpy.prod([len(x) for x in points_in_bins]) == len(lookup_indices)

    return result


def compute_cubes_size(cubes):
    ans = 0
    for cube in cubes:
        ans += compute_cube_size(cube)
    return ans


def compute_cube_size(used_cuts):
    ans = 1
    for c in used_cuts:
        ans *= (len(c) + 1)
    return ans


def compute_raw_cube_size(trees, indicies):
    cuts = []
    for i in indicies:
        fs, cs, ls = trees[i]
        for f, c in zip(fs, cs):
            while f >= len(cuts):
                cuts.append(set())
            cuts[f].add(c)
    size = 1
    for c in cuts:
        size *= (len(c) + 1)
    return size


def extract_trees(mx):
    trees = []
    for _, _, iterator in mx.iterate_trees():
        for features, cuts, leafs in iterator:
            trees.append((features, cuts, leafs))
    return trees


def greedy_split_to_cube_packs(mx, tree_count):
    trees = extract_trees(mx)
    return greedy_split_to_cube_packs_raw(trees, tree_count, len(mx.features))


def greedy_split_to_cube_packs_raw(trees, tree_count, feature_count):
    cubes = [[i] for i in xrange(len(trees))]
    while len(cubes) > tree_count:
        optimal_pair = (None, None)
        optimal_increase = None
        for i in xrange(len(cubes)):
            for j in xrange(len(cubes)):
                if i >= j:
                    continue
                increase = (
                    compute_raw_cube_size(trees, cubes[i] + cubes[j]) -
                    compute_raw_cube_size(trees, cubes[i]) - compute_raw_cube_size(trees, cubes[j])
                )
                if optimal_increase is None or optimal_increase > increase:
                    optimal_pair = (i, j)
                    optimal_increase = increase

        cubes[optimal_pair[0]] += cubes[optimal_pair[1]]
        cubes[optimal_pair[1]] = cubes[-1]
        cubes.pop()

    all_cuts = []
    all_trees = []
    for cube in cubes:
        cuts = [set() for _ in xrange(feature_count)]
        used_trees = []
        for i in cube:
            used_trees.append(trees[i])
            fs, cs, ls = trees[i]
            for f, c in zip(fs, cs):
                cuts[f].add(c)
        all_cuts.append(cuts)
        all_trees.append(used_trees)
    return all_cuts, all_trees, compute_cubes_size(all_cuts)


def greedy_split_to_cube_packs_varianted(mx, tree_count):
    trees = extract_trees(mx)
    result = []
    for i in xrange(len(trees)):
        print "current iteration: {}".format(i)
        result.append(greedy_split_to_cube_packs_raw(trees[:i + 1], tree_count, len(mx.features)))
    return result


def random_split_to_cube_packs(mx, tree_count, attempts=10):
    trees = []
    for _, _, iterator in mx.iterate_trees():
        for features, cuts, leafs in iterator:
            trees.append((features, cuts, leafs))
    best_cubes = None
    best_size = None
    for attempt in xrange(attempts):
        initial_indicies = random.sample(range(len(trees)), tree_count)
        used = [False for _ in trees]
        for i in initial_indicies:
            used[i] = True
        cubes = [[i] for i in initial_indicies]
        for i in xrange(len(trees) - tree_count):
            best_candidate = (None, None)
            lowest_increase = None
            for candidate in xrange(len(trees)):
                if used[candidate]:
                    continue
                for c in xrange(len(cubes)):
                    current_size = compute_raw_cube_size(trees, cubes[c])
                    possible_size = compute_raw_cube_size(trees, cubes[c] + [i])
                    increase = possible_size - current_size
                    if lowest_increase is None or lowest_increase > increase:
                        best_candidate = (candidate, c)
                        lowest_increase = increase
            used[best_candidate[0]] = True
            cubes[best_candidate[1]].append(best_candidate[0])

        current_size = 0
        for cube in cubes:
            current_size += compute_raw_cube_size(trees, cube)
        if best_size is None or current_size < best_size:
            best_size = current_size
            best_cubes = cubes
        print "iteration: {}, size: {}".format(attempt, best_size)
    all_cuts = []
    all_trees = []
    for cube in best_cubes:
        cuts = [set() for _ in xrange(len(mx.features))]
        used_trees = []
        for i in cube:
            used_trees.append(trees[i])
            fs, cs, ls = trees[i]
            for f, c in zip(fs, cs):
                cuts[f].add(c)
        all_cuts.append(cuts)
        all_trees.append(used_trees)
    return all_cuts, all_trees, compute_cubes_size(all_cuts)


def split_to_cube_packs(mx, max_tree_size):
    cubes = []
    trees = []

    current_trees = []
    current_used_cuts = [set() for _ in xrange(len(mx.features))]
    for _, _, iterator in mx.iterate_trees():
        for features, cuts, leafs in iterator:
            previous_cuts = deepcopy(current_used_cuts)
            for f, c in zip(features, cuts):
                current_used_cuts[f].add(c)
            if compute_cube_size(current_used_cuts) > max_tree_size:
                cubes.append(previous_cuts)
                trees.append(current_trees)
                current_used_cuts = [set() for _ in xrange(len(mx.features))]
                current_trees = [(features, cuts, leafs)]
                for f, c in zip(features, cuts):
                    current_used_cuts[f].add(c)
            else:
                current_trees.append((features, cuts, leafs))

    if current_trees:
        cubes.append(current_used_cuts)
        trees.append(current_trees)
    return cubes, trees, compute_cubes_size(cubes)


def write_formula(inp_file, out_file, threshold, max_tree_size):
    with open(inp_file) as inp_stream:
        with open(out_file, "w") as out_stream:
            write_formula_stream(inp_stream, out_stream, threshold, max_tree_size)


def write_formula_stream(inp_stream, out_stream, threshold, max_tree_size):
    mx = MatrixnetApplier(inp_stream)

    cubes, trees = split_to_cube_packs(mx, max_tree_size)

    print cubes
    print [[len(_) for _ in cube] for cube in cubes]
    print [len(_) for _ in trees]

    out_stream.write(str(len(cubes)) + "\n")
    out_stream.write(str(len(mx.features)) + "\n")
    out_stream.write(" ".join([str(f) for f in mx.features]) + " # features\n")
    for i in xrange(len(cubes)):
        bias = 0
        if i == 0:
            bias = mx.bias
        write_cube_to_stream(cubes[i], trees[i], bias, out_stream, threshold)


def write_cube_to_stream(cube, trees, bias, out_stream, threshold):
    bins = [
        sorted(list(_)) for _ in cube
    ]
    for i in xrange(len(bins)):
        m = 0
        if bins[i]:
            m = -10 * abs(bins[i][0])
        bins[i] = [m] + bins[i]

    bins_quantities = numpy.array([len(_) for _ in bins])
    count = numpy.prod(bins_quantities)

    points_in_bins = []
    for i in xrange(len(bins)):
        edges = numpy.array(bins[i])
        points_in = (edges[1:] + edges[:-1]) / 2.
        points_in = numpy.array(list(points_in) + [edges[-1] + 1.])
        points_in_bins.append(points_in)

    print "Total event count: " + str(count)
    out_stream.write(" ".join([str(b) for b in bins_quantities]) + "\n")
    for fbins in bins:
        out_stream.write(" ".join([str(b) for b in fbins]) + "\n")
        fbins.append(abs(fbins[-1]) * 3)

    divider = 10000
    out_stream.write(str(divider) + "\n")

    events = convert_lookup_index_to_bins(points_in_bins, lookup_indices=numpy.arange(count))

    predictions = expit(apply_mx_trees(events, bias, trees))
    assert len(predictions) == count
    for q, pred in enumerate(predictions):
        if pred > threshold:
            out_stream.write(str(q) + " " + str(int(pred * divider)) + "\n")


def build_cubes(mxs, qualities, tree_count):
    results = []
    for mx, quals in zip(mxs, qualities):
        for t, q in zip(greedy_split_to_cube_packs_varianted(mx, tree_count), quals):
            results.append(tuple(list(t) + [q]))
    results.sort(key=lambda x: x[2])
    return results


def clean_cubes(cube_results):
    cube_results.sort(key=lambda x: x[2])
    clean_results = []
    for result in cube_results:
        if len(clean_results) == 0 or clean_results[-1][3] < result[3]:
            clean_results.append(result)
    return clean_results
