# -*- coding: utf-8 -*-
# Author: Binh Nguyen <tuan-binh.nguyen@inria.fr>
"""
Source code of Zhang & Zhang (2014) desparsified lasso borrowed from Jerome-Alexis, used for FDR
control using desparsified Lasso by Javanmard & Javandi (2018).
"""
import numpy as np
from numpy.linalg import inv, norm
from scipy import stats
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from .utils import _reid


def desparsified_lasso(X, y, centered=True, tol=1e-4, method="lasso", c=0.01, n_jobs=1):
    """Desparsified Lasso with pvalues

    Parameters
    -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary
        confidence : float, optional
            Confidence level used to compute the confidence intervals.
            Each value should be in the range [0, 1].
        tol : float, optional
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``.
        method : string, optional
            The method for the nodewise lasso: "lasso", "lasso_cv" or
            "zhang_zhang"
        c : float, optional
            Only used if method="lasso". Then alpha = c * alpha_max.
    """

    X = np.asarray(X)

    if centered:
        X = StandardScaler().fit_transform(X)

    n_samples, n_features = X.shape

    Z = np.zeros((n_samples, n_features))
    omega_diag = np.zeros(n_features)

    if method == "lasso":
        Gram = np.dot(X.T, X)

        k = c * (1.0 / n_samples)
        alpha = k * np.max(np.abs(Gram - np.diag(np.diag(Gram))), axis=0)

    elif method == "lasso_cv":
        clf_lasso_loc = LassoCV(tol=tol, n_jobs=n_jobs)

    # Calculating Omega Matrix
    for i in range(n_features):
        if method == "lasso":
            Gram_loc = np.delete(np.delete(Gram, obj=i, axis=0), obj=i, axis=1)
            clf_lasso_loc = Lasso(alpha=alpha[i], precompute=Gram_loc, tol=tol)

        if method == "lasso" or method == "lasso_cv":
            X_new = np.delete(X, i, axis=1)
            clf_lasso_loc.fit(X_new, X[:, i])

            Z[:, i] = X[:, i] - clf_lasso_loc.predict(X_new)

        elif method == "zhang_zhang":
            print("i = ", i)
            X_new = np.delete(X, i, axis=1)
            alpha, z, eta, tau = _lpde_regularizer(X_new, X[:, i])

            Z[:, i] = z

        omega_diag[i] = (
            n_samples * np.sum(Z[:, i] ** 2) / np.sum(Z[:, i] * X[:, i]) ** 2
        )

    # Lasso regression
    clf_lasso_cv = LassoCV(n_jobs=n_jobs)
    clf_lasso_cv.fit(X, y)
    beta_lasso = clf_lasso_cv.coef_

    # Estimating the coefficient vector
    beta_bias = y.T.dot(Z) / np.sum(X * Z, axis=0)

    P = ((Z.T.dot(X)).T / np.sum(X * Z, axis=0)).T
    P_nodiag = P - np.diag(np.diag(P))

    beta_hat = beta_bias - P_nodiag.dot(beta_lasso)

    sigma_hat = _reid(X, y)

    zscore = np.sqrt(n_samples) * beta_hat / (sigma_hat * np.sqrt(omega_diag))
    pval = 2 * stats.norm.sf(np.abs(zscore))

    return beta_hat, zscore, pval


def _lpde_regularizer(
    X, y, grid=100, alpha_max=None, kappa_0=0.25, kappa_1=0.5, c_max=0.99, eps=1e-3
):
    X = np.asarray(X)
    n_samples, n_features = X.shape

    eta_star = np.sqrt(2 * np.log(n_features))

    z_grid = np.zeros(grid * n_samples).reshape(grid, n_samples)
    eta_grid = np.zeros(grid)
    tau_grid = np.zeros(grid)

    if alpha_max is None:
        alpha_max = np.max(np.dot(X.T, y)) / n_samples

    alpha_0 = eps * c_max * alpha_max
    z_grid[0, :], eta_grid[0], tau_grid[0] = _lpde_regularizer_substep(X, y, alpha_0)

    if eta_grid[0] > eta_star:
        eta_star = (1 + kappa_1) * eta_grid[0]

    alpha_1 = c_max * alpha_max
    z_grid[-1, :], eta_grid[-1], tau_grid[-1] = _lpde_regularizer_substep(X, y, alpha_1)

    alpha_grid = _alpha_grid(X, y, eps=eps, n_alphas=grid)[::-1]
    alpha_grid[0] = alpha_0
    alpha_grid[-1] = alpha_1

    for i, alpha in enumerate(alpha_grid[1:-1], 1):
        z_grid[:, i], eta_grid[i], tau_grid[i] = _lpde_regularizer_substep(X, y, alpha)

    # tol_factor must be inferior to (1 - 1 / (1 + kappa_1)) = 1 / 3 (default)
    index_1 = (grid - 1) - (eta_grid <= eta_star)[-1].argmax()

    tau_star = (1 + kappa_0) * tau_grid[index_1]

    index_2 = (tau_grid <= tau_star).argmax()

    return (
        alpha_grid[index_2],
        z_grid[:, index_2],
        eta_grid[index_2],
        tau_grid[index_2],
    )


def _lpde_regularizer_substep(X, y, alpha):
    clf_lasso = Lasso(alpha=alpha)
    clf_lasso.fit(X, y)

    z = y - clf_lasso.predict(X)
    z_norm = np.linalg.norm(z)
    eta = np.max(np.dot(X.T, z)) / z_norm
    tau = z_norm / np.sum(y * z)

    return z, eta, tau
