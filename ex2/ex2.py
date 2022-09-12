import numpy as np
import random
import matplotlib.pyplot as plt
from models import OptiBound
from utils import (
    test_function,
    cost_function,
    sample_function_2d,
    get_bound_vals_mesh,
    solve_grid,
    get_bound_vals_mesh_prim,
)


#########################
# Hyperparameter Settings
#########################

eps = 1  # Noise level of the samples
approx_method = "alt"
random.seed(1)
l = 5
lb = [-10, -10]
ub = [10, 10]
Gamma = 1200
krr_reg = 0.0001
x0 = np.array([[10, -10]])
sampling_method = "grid"  # "random"
num_points = 100  # 64, 81

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


z1 = np.linspace(-10, 10, 50)
z2 = np.linspace(-10, 10, 50)
XX, YY = np.meshgrid(z1, z2)
ZZ = cost_function(XX, YY)
ZZ_constr = test_function(XX, YY)

########################
# Running the experiment
########################

cost = lambda x: cost_function(x[0], x[1])
constr = lambda x: -test_function(x[0], x[1])

vals = []
if sampling_method == "random":
    for i in range(10):
        print(
            f"""##############
# Run {i+1} of 10
############## """
        )
        train_points, train_samples = sample_function_2d(
            test_function, lb.copy(), ub, num_points, method=sampling_method, eps=eps
        )
        opt_grid = OptiBound(l, eps, Gamma)
        opt_grid.fit(train_points, train_samples)
        val, z = solve_grid(opt_grid, cost, lb.copy(), ub.copy(), 10, 4)
        vals.append(val)
elif sampling_method == "grid":
    train_points, train_samples = sample_function_2d(
        test_function,
        lb.copy(),
        ub,
        int(np.sqrt(num_points)),
        method=sampling_method,
        eps=eps,
    )
    opt_grid = OptiBound(l, eps, Gamma)
    opt_grid.fit(train_points, train_samples)
    val, z = solve_grid(opt_grid, cost, lb.copy(), ub.copy(), 10, 4)
    vals.append(val)
print(f"Cost value(s) of solution(s): {vals}")
ZZ_con_ker = get_bound_vals_mesh_prim(opt_grid, XX, YY)

fig, ax = plt.subplots()
CS = ax.contourf(XX, YY, ZZ, levels=30, cmap="cool")
CS = ax.contour(XX, YY, ZZ_constr, levels=[0.0], colors="k")
ax.plot(z[0], z[1], "or")
plt.title("Solution with true constraints")
plt.show()
fig, ax = plt.subplots()
CS = ax.contourf(XX, YY, ZZ, levels=30, cmap="cool")
CS = ax.contour(XX, YY, ZZ_con_ker, levels=[0.0], colors="k")
ax.set_xlabel("$z_1$")
ax.set_ylabel("$z_2$")
ax.plot(z[0], z[1], "or")
plt.title("Solution with approximated constraints")
plt.show()
