"""Implementation of Javanmard & Montanari (2014a,b) in Python
TODO: cythonize solve precision for speeding up process
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import Lasso, LassoCV


def debiased_lasso(X, y, cv=3, alpha=1.0, mu=None, resol=1.3, max_iter=50,
                   tol=1e-2, inference=False, q=0.05, verbose=False, n_jobs=1):
    """Debiased Lasso of Javanmard & Montanari (2014a,b) in Python

    :param X: 
    :param y: 
    :param cv: 
    :param alpha: float, optional
         regularization parameter for Lasso problem, adaptively chosen if cv > 1
    :param mu: 
    :param resol: 
    :param max_iter: 
    :param tol: 
    :param inference: 
    :param q: float, optional
        Degree of siginificance for calculation of confidence intervals &
        pvalues, only used when inference is True
    :param verbose: 
    :param n_jobs: 

    """
    n_samples, n_features = X.shape
    Sigma_hat = X.T.dot(X) / n_samples
    col_norm = 1 / np.sqrt(np.diag(Sigma_hat))

    X = X.dot(np.diag(col_norm))  # normalizing the column of X

    # Solving Lasso problem & finding biased coefs
    if cv > 1:
        clf = LassoCV(max_iter=max_iter, n_jobs=n_jobs, cv=cv)
    else:
        clf = Lasso(max_iter=max_iter, alpha=alpha)

    clf.fit(X, y)
    coef = np.ravel(clf.coef_)

    # Classical regime of low-dimension
    if n_samples >= 2 * n_features:
        tmp = np.linalg.eigvalsh(Sigma_hat)
        tmp = np.min(tmp) / np.max(tmp)  # ratio of lambda_min / lambda_max
    else:
        tmp = 0

    if n_samples >= 2 * n_features and tmp >= 1e-4:
        M = np.linalg.inv(Sigma_hat)  # if we are in nice regime
    else:
        M = _solve_precision(Sigma_hat, n_samples, resol=resol, mu=mu,
                             max_iter=max_iter, tol=tol, verbose=verbose)

    # Bias correction
    unbiased_coef = coef + M.dot(X.T.dot(y - X.dot(coef))) / n_samples

    # Doing inference
    if inference:
        A = M.dot(Sigma_hat.dot(M.T))
        s_hat, s0 = _noise_estimation(unbiased_coef, A, n_samples)

        interval_size = stats.norm.ppf(1 - q / 2) * s_hat * \
            np.sqrt(np.diag(A)) / np.sqrt(n_samples)

        add_length = np.zeros(n_features)

        MM = M.dot(Sigma_hat) - np.eye(n_features)
        for i in range(n_features):
            effective_mu_vec = np.sort(np.abs(-MM[i, :]))  # sort decreasing
            effective_mu_vec = effective_mu_vec[s0 - 1]
            add_length[i] = np.sqrt(np.sum(effective_mu_vec ** 2)) * clf.alpha_

        interval_size = interval_size * col_norm
        add_length = add_length * col_norm

    # Normalizing
    coef = coef * col_norm
    unbiased_coef = unbiased_coef * col_norm

    if inference:
        pvals = 2 * (1 - stats.norm.cdf(
            np.sqrt(n_samples) * np.abs(unbiased_coef) / (
                s_hat * col_norm * np.sqrt(np.diag(A)))))

        lower_interval = unbiased_coef - interval_size - add_length
        upper_interval = unbiased_coef + interval_size + add_length

        return (unbiased_coef, pvals, lower_interval, upper_interval,
                M, s_hat, s0)

    return unbiased_coef, coef, M


def _solve_precision(Sigma, n_samples, resol=1.5, mu=None, max_iter=50,
                     tol=1e-2, verbose=True):

    is_given = False if mu is None else True
    n_features = Sigma.shape[0]

    M = np.zeros((n_features, n_features))
    xperc = 0
    xp = round(n_features / 10)

    for i in range(n_features):
        beta = np.zeros(n_features)
        # Print progress
        if verbose:
            if i % xp == 0:
                xperc = xperc + 10
                print(f'Finding M: {xperc}% done...')
        # Reset the value of mu for each i
        if not is_given:
            mu = stats.norm.ppf(
                1 - (0.1 / (n_features ** 2))) / np.sqrt(n_samples)
        try_no = 1
        increasing = False
        while try_no <= 10:
            last_beta = beta
            beta, iteration = _solve_precision_one_row(
                Sigma, i, mu, max_iter=max_iter, tol=tol)
            if is_given is True:
                break
            else:
                if try_no == 1:
                    if iteration == max_iter + 1:
                        increasing = True
                        mu = mu * resol
                    else:
                        increasing = False
                        mu = mu / resol
                elif try_no > 1:
                    if increasing and iteration == (max_iter + 1):
                        mu = mu * resol
                    if increasing and iteration < (max_iter + 1):
                        break
                    if not increasing and iteration < (max_iter + 1):
                        mu = mu / resol
                    if not increasing and iteration == (max_iter + 1):
                        mu = mu * resol
                        beta = last_beta
                        break
            try_no += 1
            print(
                f'feature: {i}, try_no: {try_no}, mu: {mu}, increasing: {increasing}, iteration: {iteration}')
        M[i, :] = beta

    return M


def _solve_precision_one_row(Sigma, i, mu, max_iter=50, tol=1e-2):
    n_features = Sigma.shape[0]
    row_minus_i = np.delete(Sigma[i, :].copy(), i)
    rho = np.max(np.abs(row_minus_i)) / Sigma[i, i]
    mu_init = rho / (1 + rho)
    beta = np.zeros(n_features)

    diff_norm_2 = 1
    last_norm_2 = 1
    iteration = 1
    iteration_old = 1
    beta[i] = (1 - mu_init) / Sigma[i, i]

    if mu >= mu_init:
        return beta, 0

    beta_old = beta
    Sigma_tilde = Sigma.copy()
    np.fill_diagonal(Sigma_tilde, 0)
    vs = -Sigma_tilde.dot(beta)

    while (iteration <= max_iter) and (diff_norm_2 >= tol * last_norm_2):
        for j in range(n_features):
            old_val = beta[j]
            v = vs[j]
            if j == i:
                v = v + 1
            beta[j] = _soft_threshold(v, mu) / Sigma[j, j]
            if old_val != beta[j]:
                vs = vs + (old_val - beta[j]) * Sigma_tilde[:, j]

        iteration = iteration + 1
        if iteration == 2 * iteration_old:
            d = beta - beta_old
            diff_norm_2 = np.sqrt(np.sum(d ** 2))
            last_norm_2 = np.sqrt(np.sum(beta ** 2))
            iteration_old = iteration
            beta_old = beta

            if iteration > 10:
                vs = -Sigma_tilde.dot(beta)

    return beta, iteration


def _noise_estimation(beta_hat, A, n_samples):
    beta_hat_norm = np.sqrt(n_samples) * beta_hat / np.sqrt(np.diag(A))
    sd_hat_0 = stats.median_absolute_deviation(beta_hat_norm)
    # check the meaning of these later
    zeros = np.abs(beta_hat_norm) < 3 * sd_hat_0

    beta_hat_2_norm = np.sum(beta_hat_norm[zeros] ** 2)
    sd_hat_1 = np.sqrt(n_samples * beta_hat_2_norm / np.trace(A))
    ratio = sd_hat_0 / sd_hat_1

    if max(ratio, 1 / ratio) > 2:
        print("Warnding: problem in noise estimation")

    s0 = np.sum(zeros == False)

    return sd_hat_1, s0


def _soft_threshold(x, lbda):
    return np.sign(x) * max(0, np.abs(x) - lbda)
