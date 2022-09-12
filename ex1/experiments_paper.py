import numpy as np
import random
import math
import json
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from sklearn.gaussian_process.kernels import RBF
from models import KRR, OptiBoundDualApprox, OptiBound, GPRegressor
from utils import (
    test_function,
    sample_function_2d,
    plot_3d,
    plot_2d_comparison,
)


##################################
# Experiment Settings (Edit Here!)
##################################

exptype = "gp"  # "3d", "2d" ,"krr", "opt"
evalpoints_sqrt = 50  # Evaluation will be done on a grid with evalpoints_sqrt^2 points
slice_index = 10  # Only for 2d experiments. Has to be in [0, evalpoints_sqrt)
eps = 1  # Noise level of the samples
lambda_0 = 0.001  # Initial guess for lambda in the bound approximation
approx_steps = 7  # Maximum number of alternating steps in the approximation method
# noise bound approximations for bound computation
dbars = [
    1,
    1.5,
    2,
]  # [5, 7.5, 10]
gammas = [
    1200,
    1800,
    2400,
]  # RKHS norm bound approximations for bound computation
distr = "gaussian"  # "uniform"
delta = 0.01  # GP bound violation probability

#########################
# Hyperparameter Settings
#########################

approx_method = "alt"
random.seed(1)
l = 5
lb = [-10, -10]
ub = [10, 10]
Gamma = 1200
krr_reg = 0.0001
std = eps / 2.58  # 99% confidence interval

###############
# Plot Settings
###############

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
blue = "#0072BD"
green = "#00C651"
magenta = "#AD0764"
yellow = "#E6F13E"
alpha = 0.15

###################
# Function sampling
###################


train_points, train_samples = sample_function_2d(
    test_function, lb, ub, 100, method="random", eps=eps, distr=distr, std=std
)
train_points2, train_samples2 = sample_function_2d(
    test_function, lb, ub, 10, method="grid", eps=eps, distr=distr, std=std
)
test_points, test_samples = sample_function_2d(test_function, lb, ub, evalpoints_sqrt)

################
# Model training
################

opt_grid = OptiBound(l, eps, Gamma)
opt_random = OptiBound(l, eps, Gamma)
opt_grid.fit(train_points2, train_samples2)
opt_random.fit(train_points, train_samples)
if exptype == "2d":
    krr_grid = KRR(krr_reg, l, eps, Gamma)
    krr_random = KRR(krr_reg, l, eps, Gamma)
    opt_approx_grid = OptiBoundDualApprox(l, eps, Gamma)
    opt_approx_random = OptiBoundDualApprox(l, eps, Gamma)

    krr_grid.fit(train_points2, train_samples2)
    krr_random.fit(train_points, train_samples)
    opt_approx_grid.fit(train_points2, train_samples2)
    opt_approx_random.fit(train_points, train_samples)

################
# 2d Experiments
################

if exptype == "2d":
    lb_krr_grid = []
    ub_krr_grid = []
    lb_krr_random = []
    ub_krr_random = []
    lb_opt_grid = []
    ub_opt_grid = []
    lb_opt_random = []
    ub_opt_random = []
    lb_opt_approx_grid = []
    ub_opt_approx_grid = []
    lb_opt_approx_random = []
    ub_opt_approx_random = []
    i = 1
    for x in test_points[
        slice_index * evalpoints_sqrt : (slice_index + 1) * evalpoints_sqrt
    ]:
        print(f"point {i}")
        i += 1
        inp = np.array([x])
        lb_krr_grid.append(krr_grid.get_lower_bound(inp)[0])
        ub_krr_grid.append(krr_grid.get_upper_bound(inp)[0])
        lb_krr_random.append(krr_random.get_lower_bound(inp)[0])
        ub_krr_random.append(krr_random.get_upper_bound(inp)[0])
        lb_opt_grid.append(opt_grid.get_lower_bound(inp)[0])
        ub_opt_grid.append(opt_grid.get_upper_bound(inp)[0])
        lb_opt_random.append(opt_random.get_lower_bound(inp)[0])
        ub_opt_random.append(opt_random.get_upper_bound(inp)[0])
        lb_opt_approx_grid.append(
            opt_approx_grid.get_lower_bound(
                inp, approx_method, lam=lambda_0, steps=approx_steps
            )[0]
        )
        ub_opt_approx_grid.append(
            opt_approx_grid.get_upper_bound(
                inp, approx_method, lam=lambda_0, steps=approx_steps
            )[0]
        )
        lb_opt_approx_random.append(
            opt_approx_random.get_lower_bound(
                inp, approx_method, lam=lambda_0, steps=approx_steps
            )[0]
        )
        ub_opt_approx_random.append(
            opt_approx_random.get_upper_bound(
                inp, approx_method, lam=lambda_0, steps=approx_steps
            )[0]
        )
    points = test_points[
        slice_index * evalpoints_sqrt : (slice_index + 1) * evalpoints_sqrt
    ].T[1]
    samples = test_samples[
        slice_index * evalpoints_sqrt : (slice_index + 1) * evalpoints_sqrt
    ]

    #############################################
    # Plotting Opt vs. KRR and Opt vs. Opt Approx
    #############################################

    fig1, (ax11, ax12) = plt.subplots(2, 1, figsize=(10, 10))
    fig2, (ax21, ax22) = plt.subplots(2, 1, figsize=(10, 10))
    ax11, ax12 = plot_2d_comparison(
        ax11,
        ax12,
        points,
        samples,
        lb_opt_grid,
        ub_opt_grid,
        lb_opt_random,
        ub_opt_random,
        lb_krr_grid,
        ub_krr_grid,
        lb_krr_random,
        ub_krr_random,
        blue,
        green,
        alpha,
    )
    ax21, ax22 = plot_2d_comparison(
        ax21,
        ax22,
        points,
        samples,
        lb_opt_grid,
        ub_opt_grid,
        lb_opt_random,
        ub_opt_random,
        lb_opt_approx_grid,
        ub_opt_approx_grid,
        lb_opt_approx_random,
        ub_opt_approx_random,
        blue,
        yellow,
        alpha,
    )

    ax11.set_xlim(-10, 10)
    ax12.set_xlim(-10, 10)
    ax21.set_xlim(-10, 10)
    ax22.set_xlim(-10, 10)
    ax11.set_xlabel(r"$z_2$")
    ax12.set_xlabel(r"$z_2$")
    ax21.set_xlabel(r"$z_2$")
    ax22.set_xlabel(r"$z_2$")
    ax11.set_ylabel(r"$f(z_1,z_2)$")
    ax12.set_ylabel(r"$f(z_1,z_2)$")
    ax21.set_ylabel(r"$f(z_1,z_2)$")
    ax22.set_ylabel(r"$f(z_1,z_2)$")
    ax11.set_title("Grid Sampling")
    ax12.set_title("Random Sampling")
    ax21.set_title("Grid Sampling")
    ax22.set_title("Random Sampling")
    # fig1.tight_layout()
    fig1.suptitle("Optimal Bound vs. KRR", y=0.92)
    # fig2.tight_layout()
    fig2.suptitle("Optimal Bound vs. Approximation", y=0.92)
    plt.show()

################
# 3d Experiments
################

elif exptype == "3d":
    ub_grid = []
    ub_random = []
    i = 1
    for x in test_points:
        print(f"point {i}")
        i += 1
        inp = np.array([x])
        ub_grid.append(opt_grid.get_upper_bound(inp))
        ub_random.append(opt_random.get_upper_bound(inp))
    ub_grid = np.array(ub_grid).flatten()
    ub_random = np.array(ub_random).flatten()
    vmin = min(np.min(ub_random), np.min(ub_grid))
    vmax = max(np.max(ub_random), np.max(ub_grid))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, subplot_kw={"projection": "3d"}, figsize=(15, 6)
    )
    ax1, surf1 = plot_3d(
        ax1,
        test_points,
        test_samples,
        cmap=cm.Spectral,
        linewidth=0,
        antialiased=False,
        alpha=0.8,
    )
    ax1, surf2 = plot_3d(
        ax1,
        test_points,
        ub_random,
        cmap=cm.cool,
        linewidth=0,
        antialiased=False,
        alpha=0.7,
        vmin=vmin,
        vmax=vmax,
    )

    ax2, surf = plot_3d(
        ax2,
        test_points,
        test_samples,
        cmap=cm.Spectral,
        linewidth=0,
        antialiased=False,
        alpha=0.8,
    )
    ax2, surf2 = plot_3d(
        ax2,
        test_points,
        ub_grid,
        cmap=cm.cool,
        linewidth=0,
        antialiased=False,
        alpha=0.7,
        vmin=vmin,
        vmax=vmax,
    )

    ax1.set_xlim3d(-10, 10)
    ax1.set_ylim3d(-10, 10)
    ax2.set_xlim3d(-10, 10)
    ax2.set_ylim3d(-10, 10)
    mpl.rc("text.latex", preamble=r"\usepackage{sfmath}")
    ax1.set_xlabel(r"$z_1$")
    ax1.set_ylabel(r"$z_2$")
    ax1.set_zlabel(r"$f(z_1,z_2)$")
    ax2.set_xlabel(r"$z_1$")
    ax2.set_ylabel(r"$z_2$")
    ax2.set_zlabel(r"$f(z_1,z_2)$")
    ax1.set_zlim3d(-100, 50)
    ax2.set_zlim3d(-100, 50)
    ax1.set_xticks(np.array([-10, -5, 0, 5, 10]))
    ax1.set_yticks(np.array([-10, -5, 0, 5, 10]))
    ax2.set_xticks(np.array([-10, -5, 0, 5, 10]))
    ax2.set_yticks(np.array([-10, -5, 0, 5, 10]))
    fig.tight_layout()
    plt.show()


############################################
# Delta, Gamma Overapproximation Experiments
############################################

elif exptype == "opt":
    res_dict = {}
    for dbar in dbars:
        for gamma in gammas:
            opt_grid.dbar = dbar
            opt_grid.Gamma = gamma
            opt_random.dbar = dbar
            opt_random.Gamma = gamma
            avg_dist_grid = 0
            avg_dist_rand = 0
            for i, x in enumerate(test_points):
                # print(f"point {i+1}"
                if i % 100 == 0:
                    print(f"point {i}")
                inp = np.array([x])
                avg_dist_grid += opt_grid.get_upper_bound(
                    inp
                ) - opt_grid.get_lower_bound(
                    inp
                )  # test_samples[i]
                avg_dist_rand += opt_random.get_upper_bound(
                    inp
                ) - opt_random.get_lower_bound(
                    inp
                )  # test_samples[i]
            avg_dist_grid /= len(test_points)
            avg_dist_rand /= len(test_points)
            name = f"true_dbar_{eps}_true_gamma_{Gamma}_dbar_{dbar}_gamma_{gamma}_std_{(dbar / 2.58)}"
            res_dict[name + "_grid"] = avg_dist_grid.tolist()
            res_dict[name + "_random"] = avg_dist_rand.tolist()
            print("type | dbar | Gamma | ap. dbar | ap. Gamma | avg dist ")
            print(f"grid | {eps} | {Gamma} | {dbar} | {gamma} | {avg_dist_grid} ")
            print(f"rand | {eps} | {Gamma} | {dbar} | {gamma} | {avg_dist_rand} ")

    print("type | dbar | Gamma | ap. dbar | ap. Gamma | avg dist ")
    for dbar in dbars:
        for gamma in gammas:
            name = f"true_dbar_{eps}_true_gamma_{Gamma}_dbar_{dbar}_gamma_{gamma}_std_{(dbar / 2.58)}"
            print(
                f"grid |  {eps}   | {Gamma}  |    {dbar}     |   {gamma}    | {res_dict[name + '_grid']} "
            )
            print(
                f"rand |  {eps}   | {Gamma}  |    {dbar}     |   {gamma}    | {res_dict[name + '_random']} "
            )
    with open(f"opt_data_dbar_{eps}_Gamma_{Gamma}.json", "w") as fp:
        json.dump(res_dict, fp)

#################################################
# Delta, Gamma Overapproximation Experiments (GP)
#################################################

elif exptype == "gp":
    res_dict = {}
    kernel = RBF(length_scale=l, length_scale_bounds="fixed")
    for dbar in dbars:
        gp_grid = GPRegressor(kernel, dbar / 2.58)
        gp_rand = GPRegressor(kernel, dbar / 2.58)
        gp_grid.fit(train_points2, train_samples2)
        gp_rand.fit(train_points, train_samples)
        for gamma in gammas:
            gam_n = (
                1
                / 2
                * math.log(
                    np.linalg.det(
                        np.eye(10 ** 2) + 1 / (dbar / 2.58) ** 2 * kernel(train_points2)
                    )
                )
            )
            beta = gamma + dbar * math.sqrt(2 * (gam_n + 1 + math.log(1 / delta)))
            avg_dist_grid = 0
            avg_dist_rand = 0
            for i, x in enumerate(test_points):
                # print(f"point {i+1}"
                if i % 100 == 0:
                    print(f"point {i}")
                inp = np.array([x])
                lb_grid, ub_grid = gp_grid.get_bounds(inp, beta)
                lb_rand, ub_rand = gp_rand.get_bounds(inp, beta)
                avg_dist_grid += ub_grid - lb_grid  # test_samples[i]
                avg_dist_rand += ub_rand - lb_rand  # test_samples[i]
                if ub_grid < test_samples[i]:
                    print(ub_grid, test_samples[i])
                    input("..")
            avg_dist_grid /= len(test_points)
            avg_dist_rand /= len(test_points)
            name = f"true_dbar_{eps}_true_gamma_{Gamma}_dbar_{dbar}_gamma_{gamma}_std_{(dbar / 2.58)}"
            res_dict[name + "_grid"] = avg_dist_grid.tolist()
            res_dict[name + "_random"] = avg_dist_rand.tolist()
            print("type | dbar | Gamma | ap. dbar | ap. Gamma | avg dist ")
            print(f"grid | {eps} | {Gamma} | {dbar} | {gamma} | {avg_dist_grid} ")
            print(f"rand | {eps} | {Gamma} | {dbar} | {gamma} | {avg_dist_rand} ")

    print("type | dbar | Gamma | ap. dbar | ap. Gamma | avg dist ")
    for dbar in dbars:
        for gamma in gammas:
            name = f"true_dbar_{eps}_true_gamma_{Gamma}_dbar_{dbar}_gamma_{gamma}_std_{(dbar / 2.58)}"
            print(
                f"grid |  {eps}   | {Gamma}  |    {dbar}     |   {gamma}    | {res_dict[name + '_grid']} "
            )
            print(
                f"rand |  {eps}   | {Gamma}  |    {dbar}     |   {gamma}    | {res_dict[name + '_random']} "
            )
    with open(f"gp_data_dbar_{eps}_Gamma_{Gamma}.json", "w") as fp:
        json.dump(res_dict, fp)


#################################################
# Delta, Gamma Overapproximation Experiments (KRR)
#################################################

elif exptype == "krr":
    res_dict = {}
    krr_grid = KRR(krr_reg, l, eps, Gamma)
    krr_random = KRR(krr_reg, l, eps, Gamma)
    krr_grid.fit(train_points2, train_samples2)
    krr_random.fit(train_points, train_samples)
    for dbar in dbars:
        for gamma in gammas:
            krr_grid.dbar = dbar
            krr_grid.Gamma = gamma
            krr_random.dbar = dbar
            krr_random.Gamma = gamma
            avg_dist_grid = 0
            avg_dist_rand = 0
            for i, x in enumerate(test_points):
                # print(f"point {i+1}"
                if i % 100 == 0:
                    print(f"point {i}")
                inp = np.array([x])
                ub_grid = krr_grid.get_upper_bound(inp)[0]
                ub_rand = krr_random.get_upper_bound(inp)[0]
                lb_grid = krr_grid.get_lower_bound(inp)[0]
                lb_rand = krr_random.get_lower_bound(inp)[0]
                avg_dist_grid += ub_grid - lb_grid  # test_samples[i]
                avg_dist_rand += ub_rand - lb_rand  # test_samples[i]
                if ub_grid < test_samples[i]:
                    print(ub_grid, test_samples[i])
                    input("..")
            avg_dist_grid /= len(test_points)
            avg_dist_rand /= len(test_points)
            name = f"true_dbar_{eps}_true_gamma_{Gamma}_dbar_{dbar}_gamma_{gamma}_std_{(dbar / 2.58)}"
            res_dict[name + "_grid"] = avg_dist_grid.tolist()
            res_dict[name + "_random"] = avg_dist_rand.tolist()
            print("type | dbar | Gamma | ap. dbar | ap. Gamma | avg dist ")
            print(f"grid | {eps} | {Gamma} | {dbar} | {gamma} | {avg_dist_grid} ")
            print(f"rand | {eps} | {Gamma} | {dbar} | {gamma} | {avg_dist_rand} ")

    print("type | dbar | Gamma | ap. dbar | ap. Gamma | avg dist ")
    for dbar in dbars:
        for gamma in gammas:
            name = f"true_dbar_{eps}_true_gamma_{Gamma}_dbar_{dbar}_gamma_{gamma}_std_{(dbar / 2.58)}"
            print(
                f"grid |  {eps}   | {Gamma}  |    {dbar}     |   {gamma}    | {res_dict[name + '_grid']} "
            )
            print(
                f"rand |  {eps}   | {Gamma}  |    {dbar}     |   {gamma}    | {res_dict[name + '_random']} "
            )
    with open(f"krr_data_dbar_{eps}_Gamma_{Gamma}.json", "w") as fp:
        json.dump(res_dict, fp)