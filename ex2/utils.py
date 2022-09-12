import numpy as np
import random


def test_function(x1, x2):
    return 1 - 0.8 * x1 ** 2 + x2 + 8 * np.sin(0.8 * x2) + 10


def cost_function(x1, x2):
    return (x1 - 1) ** 2 + (x2 - 5) ** 2


def get_bound_vals_mesh(rto, XX, YY):
    ZZ = np.zeros_like(XX)
    (s1, s2) = XX.shape
    for i in range(s1):
        for j in range(s2):
            point = np.array([XX[i, j], YY[i, j]])
            rto.lambda_, rto.nu = rto.alt_opt(point.reshape(1, -1))
            ZZ[i, j] = -rto.bd_constraint(point)[0]
    return ZZ


def get_bound_vals_mesh_prim(opt_mod, XX, YY):
    ZZ = np.zeros_like(XX)
    (s1, s2) = XX.shape
    for i in range(s1):
        for j in range(s2):
            point = np.array([XX[i, j], YY[i, j]]).reshape((1, -1))
            ZZ[i, j] = opt_mod.get_upper_bound(point)
    return ZZ


def solve_grid(opt_mod, cost, lb, ub, num_points, steps, rel_size=0.1):
    current_min = np.inf
    z_min = None
    for i in range(steps):
        z1s = np.linspace(lb[0], ub[0], num_points)
        z2s = np.linspace(lb[1], ub[1], num_points)
        for z1 in z1s:
            for z2 in z2s:
                point = np.array([z1, z2])
                constr = opt_mod.get_upper_bound(point.reshape((1, -1)))
                if constr <= 0:
                    if cost(point) < current_min:
                        current_min = cost(point)
                        z_min = point
                        print(
                            f"new min {z_min} with val {current_min} and constr {constr}"
                        )
        z1_len = ub[0] - lb[0]
        z2_len = ub[1] - lb[1]
        lb[0] = z_min[0] - rel_size * z1_len
        ub[0] = z_min[0] + rel_size * z1_len
        lb[1] = z_min[1] - rel_size * z2_len
        ub[1] = z_min[1] + rel_size * z2_len
        print(lb)
    return current_min, z_min


def sample_function_2d(func, lb, ub, num_points, method="grid", eps=0):
    if method == "grid":
        X1 = np.linspace(lb[0], ub[0], num_points)
        X2 = np.linspace(lb[1], ub[1], num_points)
        points = [[x1, x2] for x1 in X1 for x2 in X2]
        samples = [[func(x[0], x[1]) + random.uniform(-eps, eps)] for x in points]
    elif method == "random":
        points = []
        samples = []
        for _ in range(num_points):
            x1 = random.uniform(lb[0], ub[0])
            x2 = random.uniform(lb[1], ub[1])
            sample = func(x1, x2) + random.uniform(-eps, eps)
            points.append([x1, x2])
            samples.append([sample])
    points = np.array(points)
    samples = np.array(samples)
    return points, samples