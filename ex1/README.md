# Example 1

Code for Example 1 of the paper "Robust Uncertainty Bounds in Reproducing Kernel Hilbert Spaces:  A Convex Optimization Approach".

Authors: P. Scharnhorst, E. T. Maddalena, Y. Jiang and C. N. Jones.

## Experiments (Python)

To replicate the experiments of the paper, see the file `experiments_paper.py`. The parameter settings
- `exptype = "3d"`
- `evalpoints_sqrt = 50`

together with `eps = 1` and `eps = 5` produce 3d plots of the optimal bound comparison. The settings

- `exptype = "2d"`
- `evalpoints_sqrt = 100`
- `slice_index = 10`
- `eps = 1`
- `lambda_0 = 0.001`
- `distr = "uniform"`

together with `approx_steps = 5` and `approx_steps = 10` produce results on the comparison of the optimal bounds, the alternating minimization approach, and the subptimal alternatives on a slice.

- `exptype = "opt"`, `"krr"` or `"gp"` 
- `dbars = [1, 1.5, 2]`, or `[5, 7.5, 10]`
- `gammas = [1200, 1800, 2400]`
- `distr = "gaussian"`

together with the corresponding `eps = 1` or `eps = 5`, produce the average bound tightness comparison. 

