"""Knockoff lambda statistics follows Weinstein et at. (2017)
"""

import numpy as np
from sklearn.linear_model import lasso_path
from sklearn.utils.validation import check_memory


def stat_lambda_path(X, X_tilde, y, eps=5e-3, n_alphas=1000,
                     fit_intercept=False, memory=None):

    memory = check_memory(memory)

    X_ko = np.column_stack([X, X_tilde])
    lambdas, coefs, _ = memory.cache(lasso_path)(X_ko, y, eps=eps,
                                                 n_alphas=n_alphas)

    # test_scores = np.apply_along_axis(_sup_lambda, 1, coefs, lambdas)
    test_scores = [_sup_lambda(coefs[i, :], lambdas)
                   for i in range(X_ko.shape[1])]

    return np.array(test_scores)


def _sup_lambda(coef_path, lambda_path):
    """Find the maximum l1-reg that has coef > 0
    (or the lambda position where coef enters the model)
    """
    for i in range(len(lambda_path) - 1, -1, -1):
        if coef_path[i] == 0:
            try:
                # coef decreasing when lamba increasing
                return lambda_path[i + 1]
            # coefs can all be 0 so just return the smallest lambdas possible
            except IndexError:
                return lambda_path[-1]
    return lambda_path[-1]


def _fdp_hat_lambda(lbd, test_score):

    n_features = len(test_score) // 2
    h = test_score[:n_features]
    k0 = test_score[n_features:]
    v1 = np.sum(k0 >= lbd)

    def pi_0(t):
        return (1 + np.sum(h <= t)) / np.sum(k0 <= t)

    t0 = np.min(test_score)
    return (1 + v1) * pi_0(t0) / np.maximum(1, np.sum(h >= lbd))


def _lambda_threshold(test_score_lambda, fdr=0.1):
    test_score_sorted = np.sort(test_score_lambda)
    result = np.inf

    for t in test_score_sorted:
        if _fdp_hat_lambda(t, test_score_lambda) <= fdr:
            return t

    return result
