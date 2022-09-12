import numpy as np
import random
import matplotlib.pyplot as plt


def test_function(x1, x2):
    return 1 - 0.8 * x1 ** 2 + x2 + 8 * np.sin(0.8 * x2)


def get_scaling(
    steps, num_points, lower_pt, upper_pt, lower_sc, upper_sc, optimod, heurmod
):
    """Bisection method to get scaling for distance based heuristic"""
    old_scaling = heurmod.scaling
    points = []
    lower_pt = lower_pt.flatten()
    upper_pt = upper_pt.flatten()
    for _ in range(num_points):
        point = np.array(
            [random.uniform(lower_pt[i], upper_pt[i]) for i in range(len(lower_pt))]
        )
        points.append(point)
    lower_avg = avg_bound_diff(points, lower_sc, optimod, heurmod)
    upper_avg = avg_bound_diff(points, upper_sc, optimod, heurmod)
    for i in range(steps):
        print("#############################")
        print(f"# Step {i+1}")
        print(lower_avg, upper_avg)
        scal = (upper_sc + lower_sc) / 2
        avg_err = avg_bound_diff(points, scal, optimod, heurmod)
        if lower_avg < upper_avg:
            upper_avg = avg_err
            upper_sc = scal
        else:
            lower_avg = avg_err
            lower_sc = scal
    heurmod.scaling = old_scaling
    return scal


def avg_bound_diff(X, scaling, optimod, heurmod):
    sum_diff = 0
    heurmod.scaling = scaling
    for x in X:
        sum_diff += heurmod.get_upper_bound(np.array([x])) - optimod.get_upper_bound(
            np.array([x])
        )
    return sum_diff / len(X)


def sample_function_2d(
    func, lb, ub, num_points, method="grid", eps=0, distr="uniform", std=0
):
    if method == "grid":
        X1 = np.linspace(lb[0], ub[0], num_points)
        X2 = np.linspace(lb[1], ub[1], num_points)
        points = [[x1, x2] for x1 in X1 for x2 in X2]
        if distr == "uniform":
            samples = [[func(x[0], x[1]) + random.uniform(-eps, eps)] for x in points]
        elif distr == "gaussian":
            samples = [
                [func(x[0], x[1]) + clip(random.normalvariate(0, std), -eps, eps)]
                for x in points
            ]
    elif method == "random":
        points = []
        samples = []
        for _ in range(num_points):
            x1 = random.uniform(lb[0], ub[0])
            x2 = random.uniform(lb[1], ub[1])
            if distr == "uniform":
                sample = func(x1, x2) + random.uniform(-eps, eps)
            elif distr == "gaussian":
                sample = func(x1, x2) + clip(random.normalvariate(0, std), -eps, eps)
            points.append([x1, x2])
            samples.append([sample])
    points = np.array(points)
    samples = np.array(samples)
    return points, samples


def plot_3d(ax, points, samples, **kwargs):
    surf = ax.plot_trisurf(points.T[0], points.T[1], samples.flatten(), **kwargs)
    return ax, surf


def plot_2d_comparison(
    ax1,
    ax2,
    points,
    samples,
    lb_opt_grid,
    ub_opt_grid,
    lb_opt_random,
    ub_opt_random,
    lb_mod_grid,
    ub_mod_grid,
    lb_mod_random,
    ub_mod_random,
    color1,
    color2,
    alpha,
):
    ax1.plot(points, samples, "--k")
    ax1.plot(points, lb_opt_grid, color1)
    ax1.plot(points, ub_opt_grid, color1)
    ax1.fill_between(points, ub_opt_grid, lb_opt_grid, color=color1, alpha=alpha)
    ax1.plot(points, lb_mod_grid, color2)
    ax1.plot(points, ub_mod_grid, color2)
    ax1.fill_between(points, ub_mod_grid, ub_opt_grid, color=color2, alpha=alpha)
    ax1.fill_between(points, lb_opt_grid, lb_mod_grid, color=color2, alpha=alpha)

    ax2.plot(points, samples, "--k")
    ax2.plot(points, lb_opt_random, color1)
    ax2.plot(points, ub_opt_random, color1)
    ax2.fill_between(points, ub_opt_random, lb_opt_random, color=color1, alpha=alpha)
    ax2.plot(points, lb_mod_random, color2)
    ax2.plot(points, ub_mod_random, color2)
    ax2.fill_between(points, ub_mod_random, ub_opt_random, color=color2, alpha=alpha)
    ax2.fill_between(points, lb_opt_random, lb_mod_random, color=color2, alpha=alpha)
    return ax1, ax2


def clip(u, lb, ub):
    if u < lb:
        u = lb
    elif u > ub:
        u = ub
    return u