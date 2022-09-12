from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model._ridge import _solve_cholesky_kernel
import numpy as np
import cvxpy as cp
from scipy.spatial import distance
import scipy.optimize as op

# import jax.numpy as jnp
import abc
import random


class BaseModel(KernelRidge):
    def __init__(self, alpha, lengthscale, noisebound, Gamma, kernel="rbf"):
        if lengthscale is None or lengthscale == 0:
            gamma = None
        else:
            gamma = 1 / (2 * lengthscale ** 2)
        super().__init__(alpha=alpha, gamma=gamma, kernel=kernel)
        self.dbar = noisebound
        self.Gamma = Gamma
        self.dual_coef_ = None

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        self.K = self._get_kernel(X)
        self.y = y
        self.N = len(y)

    def get_norm(self):
        assert self.dual_coef_ is not None
        return np.sqrt(np.dot(self.y.T, self.dual_coef_)).flatten()

    @abc.abstractmethod
    def get_lower_bound(self, X):
        pass

    @abc.abstractmethod
    def get_upper_bound(self, X):
        pass


class OptiBound(BaseModel):
    def __init__(self, lengthscale, noisebound, Gamma, kernel="rbf", precision=1e-8):
        super().__init__(
            alpha=precision,
            lengthscale=lengthscale,
            noisebound=noisebound,
            Gamma=Gamma,
            kernel=kernel,
        )
        self.precision = precision

    def get_bound(self, X, btype):
        dist = distance.cdist(X, self.X_fit_).min()
        if dist <= 1e-16:
            cost_index = np.where(distance.cdist(X, self.X_fit_) <= self.precision)[1]
            Kn = self.K + self.precision * np.eye(self.N)
            In = np.eye(self.N)
            cost_len = self.N
        else:
            Xn = np.append(self.X_fit_, X, axis=0)
            Kn = self._get_kernel(Xn) + self.precision * np.eye(self.N + 1)
            In = np.append(np.eye(self.N), np.zeros((self.N, 1)), axis=1)
            cost_len = self.N + 1
            cost_index = -1
        costvec = np.zeros(cost_len)
        costvec[cost_index] = 1
        c = cp.Variable(cost_len)
        if btype == "min":
            cost = cp.Minimize(costvec.T * c)
        elif btype == "max":
            cost = cp.Maximize(costvec.T * c)
        else:
            raise ValueError(f"Unknown bound type {btype}.")
        constraints = [
            cp.matrix_frac(c, Kn) <= self.Gamma ** 2,
            In * c - self.y.flatten() <= np.ones(self.N) * self.dbar,
            -In * c + self.y.flatten() <= np.ones(self.N) * self.dbar,
        ]
        prob = cp.Problem(cost, constraints)
        prob.solve()
        return prob.value

    def get_lower_bound(self, X):
        return self.get_bound(X, "min")

    def get_upper_bound(self, X):
        return self.get_bound(X, "max")
