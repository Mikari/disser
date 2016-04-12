import pyGPs
import numpy as np
from math import sqrt


def simple_function(point):
    return -5 * (point[0] + 0.3) ** 2 - 10 * (point[1] - 0.3) ** 2


def optimize_max_possible_value(x, y, grid, func):
    model = pyGPs.GPR()
    np_x = np.array(x)
    np_y = np.array(y)
    np_z = np.array(z)
    model.getPosterior(np_x, np_y)
    model.optimize(np_x, np_y)

    used_points = set()
    for step in xrange(100):
        l = model.predict(np_z)
        possible_max_point = None
        possible_max_value = None
        possible_max_index = None
        N = 2
        for i in xrange(len(z)):
            point = z[i]
            value = l[0][i][0]
            variance = sqrt(l[1][i][0])
            if possible_max_value is None or possible_max_value < value + variance * N:
                possible_max_point = point
                possible_max_value = value + variance * N
                possible_max_index = i
        print possible_max_index, possible_max_point, possible_max_value
        if possible_max_index in used_points:
            return possible_max_point, func(possible_max_point)
        used_points.add(possible_max_index)

        x.append(possible_max_point)
        y.append(func(possible_max_point))
        np_x = np.array(x)
        np_y = np.array(y)
        np_z = np.array(z)
        model = pyGPs.GPR()
        model.getPosterior(np_x, np_y)
        model.optimize(np_x, np_y)
    return possible_max_point, func(possible_max_point)


if __name__ == "__main__":

    x = [[-1.0, -1.0]]
    y = [simple_function(a) for a in x]
    z = [[-3.0 + a / 100.0 * 6.0, -3.0 + b / 100.0 * 6.0] for a in range(0, 101) for b in range(0, 101)]

    print optimize_max_possible_value(x, y, z, simple_function)
