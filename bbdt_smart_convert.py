from __future__ import division, absolute_import
from scipy.special import expit
import numpy
from copy import deepcopy
import random
import struct

# from rep_ef.estimators._matrixnetapplier import MatrixnetApplier

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

def apply_staged_mx_trees(events, bias, trees):
        result = []

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
            result.append(leaf_values[leaf_indices.view('uint8')[:n_events]])

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
        print len(cubes)
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
    return all_cuts, all_trees, compute_cubes_size(all_cuts), "greedy"


def split_to_cube_packs_varianted(trees, feature_count, tree_count, quals, step=1):
    result = []
    print "step", step, "trees: ", len(trees), "quals", len(quals)
    for i in xrange(0, len(trees) + 1, step):
        if i < tree_count:
            continue
        print "tree iteration: ", i
        r = [
            # greedy_split_to_cube_packs_raw(trees[:i], tree_count, feature_count),
            random_split_to_cube_packs_raw(trees[:i], tree_count, feature_count, 10, True),
            # random_split_to_cube_packs_raw(trees[:i], tree_count, feature_count, 10, False)
        ]
        for l in r:
            result.append(tuple(list(l) + [quals[i]]))
    return result


def random_split_to_cube_packs_raw(trees, tree_count, feature_count, attempts=10, choose_best=True):
    best_cubes = None
    best_size = None
    for attempt in xrange(attempts):
        initial_indicies = random.sample(range(len(trees)), tree_count)
        not_used_trees = set(xrange(len(trees)))
        for i in initial_indicies:
            not_used_trees.remove(i)
        cubes = [[i] for i in initial_indicies]
        all_sizes = [compute_raw_cube_size(trees, c) for c in cubes]
        for i in xrange(len(trees) - tree_count):
            best_candidate = (None, None)
            lowest_increase = None
            if choose_best:
                for candidate in not_used_trees:
                    for c in xrange(len(cubes)):
                        current_size = all_sizes[c]
                        possible_size = compute_raw_cube_size(trees, cubes[c] + [candidate])
                        increase = possible_size - current_size
                        if lowest_increase is None or lowest_increase > increase:
                            best_candidate = (candidate, c)
                            lowest_increase = increase
            else:
                candidate = random.sample(not_used_trees, 1)[0]
                for c in xrange(len(cubes)):
                    current_size = compute_raw_cube_size(trees, cubes[c])
                    possible_size = compute_raw_cube_size(trees, cubes[c] + [candidate])
                    increase = possible_size - current_size
                    if lowest_increase is None or lowest_increase > increase:
                        best_candidate = (candidate, c)
                        lowest_increase = increase
            not_used_trees.remove(best_candidate[0])
            cubes[best_candidate[1]].append(best_candidate[0])
            all_sizes[best_candidate[1]] = compute_raw_cube_size(trees, cubes[best_candidate[1]])

        current_size = 0
        for cube in cubes:
            current_size += compute_raw_cube_size(trees, cube)
        if best_size is None or current_size < best_size:
            best_size = current_size
            best_cubes = cubes
        # print "iteration: {}, size: {}".format(attempt, best_size)
    all_cuts = []
    all_trees = []
    for cube in best_cubes:
        cuts = [set() for _ in xrange(feature_count)]
        used_trees = []
        for i in cube:
            used_trees.append(trees[i])
            fs, cs, ls = trees[i]
            for f, c in zip(fs, cs):
                cuts[f].add(c)
        all_cuts.append(cuts)
        all_trees.append(used_trees)
    name = "random"
    if choose_best:
        name += "-best"
    return all_cuts, all_trees, compute_cubes_size(all_cuts), name


def split_to_cube_packs(trees, feature_count, max_tree_size):
    cubes = []
    trees = []

    current_trees = []
    current_used_cuts = [set() for _ in xrange(feature_count)]
    for features, cuts, leafs in trees:
        previous_cuts = deepcopy(current_used_cuts)
        for f, c in zip(features, cuts):
            current_used_cuts[f].add(c)
        if compute_cube_size(current_used_cuts) > max_tree_size:
            cubes.append(previous_cuts)
            trees.append(current_trees)
            current_used_cuts = [set() for _ in xrange(feature_count)]
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


# def write_formula_stream(inp_stream, out_stream, threshold, max_tree_size):
#     mx = MatrixnetApplier(inp_stream)

#     cubes, trees = split_to_cube_packs(mx, max_tree_size)

#     print cubes
#     print [[len(_) for _ in cube] for cube in cubes]
#     print [len(_) for _ in trees]

#     out_stream.write(str(len(cubes)) + "\n")
#     out_stream.write(str(len(mx.features)) + "\n")
#     out_stream.write(" ".join([str(f) for f in mx.features]) + "\n")
#     for i in xrange(len(cubes)):
#         bias = 0
#         if i == 0:
#             bias = mx.bias
#         write_cube_to_stream(cubes[i], trees[i], bias, out_stream, threshold)


def write_fast_cubes(pool, cubes, trees, bias, f):
    print len(cubes), len(trees)
    with open(f, "w") as out_stream:
        out_stream.write(str(len(cubes)) + "\n")
        for i in xrange(len(cubes)):
            write_cube_to_stream(pool, cubes[i], trees[i], bias, out_stream)
            if i == 0:
                bias = 0



def write_cube_to_stream(pool, cube, trees, bias, out_stream):
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

    print "Total features: " + str(len(bins_quantities))
    out_stream.write(str(len(bins_quantities)) + "\n")
    print "Total event count: " + str(count)
    for fbins in bins:
        out_stream.write(str(len(fbins)) + "\n")
        out_stream.write(" ".join([str(b) for b in fbins]) + "\n")
        fbins.append(abs(fbins[-1]) * 3)

    divider = 10000
    out_stream.write(str(divider) + "\n")

    events = convert_lookup_index_to_bins(points_in_bins, lookup_indices=numpy.arange(count))
    print len(events)
    chunk_size = len(events) / 32

    workers = []
    i = 0
    while i < len(events):
        workers.append(pool.apply_async(apply_mx_trees, (events[i:i+chunk_size], bias, trees)))
        i += chunk_size
    predictions = []
    c = 0
    for w in workers:
        predictions.append(w.get())
        c += 1
        print c
    # predictions = apply_mx_trees(events, bias, trees)
    for preds in predictions:
        for pred in preds:
            out_stream.write(str(int(pred * divider)) + "\n")


def build_cubes(all_trees, feature_count, qualities, tree_count, step):
    results = []
    i = 0
    for trees, quals in zip(all_trees, qualities):
        i += 1
        print "number: ", i
        results += split_to_cube_packs_varianted(trees, feature_count, tree_count, quals, step)
    results.sort(key=lambda x: x[2])
    return results


def clean_cubes(cube_results):
    cube_results.sort(key=lambda x: x[2])
    clean_results = []
    for result in cube_results:
        if len(clean_results) == 0 or clean_results[-1][4] < result[4]:
            clean_results.append(result)
    return clean_results


def dump_data_set(data, feature_order, out):
    columns = list(data.columns)
    matrix = data.as_matrix()
    order = []
    for feature in feature_order:
        order.append(columns.index(feature))
    with open(out, "w") as f:
        for record in matrix:
            f.write("\t".join(map(str, [record[o] for o in order])))
            f.write("\n")

def serialize_mx(dump_file, factors, formula_file):
    '''
    Will merged dump_file with mx formula and list of factors into one file - formula_file
    '''

    with open(formula_file, "wb") as formula:
        with open(dump_file, "rb") as dump:
            formula.write(struct.pack('i', len(factors)))
            for factor in factors:
                formula.write(struct.pack('i', len(factor)))
                formula.write(factor)
            bytes = dump.read()
            formula.write(struct.pack('i', len(bytes)))
            formula.write(bytes)


def deserialize_mx(formula_file, dump_file):
    '''
    Will split formula_file into dump_file and list of factors
    Returns list of factors
    '''

    with open(formula_file, "rb") as formula:
        with open(dump_file, "wb") as dump:
            bytes = formula.read(4)
            factors_quantity = struct.unpack('i', bytes)[0]
            factors = []
            for index in xrange(0, factors_quantity):
                bytes = formula.read(4)
                factor_length = struct.unpack('i', bytes)[0]
                factor = formula.read(factor_length)
                factors.append(factor)
            bytes = formula.read(4)
            dump_length = struct.unpack('i', bytes)[0]
            dump.write(formula.read(dump_length))
    return factors

