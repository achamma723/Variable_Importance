import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

from .gaussian_knockoff import _cov_to_corr, _estimate_distribution, _is_posdef


def _s_equi_simultaneous(Sigma, n_bootstraps=2):
    """Following Gimenez et al (2019) - Appendix A1 
    """
    n_features = Sigma.shape[0]

    G = _cov_to_corr(Sigma)
    eig_value = np.linalg.eigvalsh(G)
    lambda_min = np.min(eig_value[0])
    S = np.ones(n_features) * min(
        (n_bootstraps + 1) / n_bootstraps * lambda_min, 1)

    psd = False
    s_eps = 0

    while psd is False:
        # if all eigval > 0 then the matrix is psd
        psd = _is_posdef(2 * G - np.diag(S * (1 - s_eps)))
        if not psd:
            if s_eps == 0:
                s_eps = 1e-08
            else:
                s_eps *= 10

    S = S * (1 - s_eps)

    return S, S * np.diag(Sigma)


def _knockoff_equi_simultaneous(X, mu, Sigma, n_bootstraps=2, seed=None):

    if n_bootstraps < 2:
        raise(ValueError('n_bootstraps must be greater than 1.'))

    n_samples, n_features = X.shape

    Diag_s_equi = np.diag(_s_equi_simultaneous(Sigma, n_bootstraps)[1])
    Sigma_inv_s = np.linalg.solve(Sigma, Diag_s_equi)

    Mu_tilda = X - np.dot(X - mu, Sigma_inv_s)
    Mu_tilda_kappa = np.tile(Mu_tilda, n_bootstraps)

    C = 2 * Diag_s_equi - Diag_s_equi.dot(Sigma_inv_s.dot(Diag_s_equi))
    C_cholesky = np.linalg.cholesky(C)
    D_cholesky = np.linalg.cholesky(Diag_s_equi)

    Sigma_kappa = np.tile(
        C_cholesky - D_cholesky, reps=(n_bootstraps, n_bootstraps))

    # Diagonal element of block matrix Sigma_kappa is equal to C
    for i in range(n_bootstraps):
        Sigma_kappa[i * n_features:(i + 1) * n_features,
                    i * n_features:(i + 1) * n_features] = C_cholesky

    np.random.seed(seed)
    U_tilde = np.random.randn(n_samples, n_features * n_bootstraps)
    # Following formula in page 4 - Barber & Candes (2015)
    X_tilde = Mu_tilda_kappa + np.dot(U_tilde, np.tril(Sigma_kappa))

    return X_tilde


def _knockoff_stat_simultaneous(X, X_tilde, y, loss='least_square',
                                cv=5, solver='liblinear',
                                n_jobs=1, return_T=False):
    """Following Gimenez et al. (2019)
    """
    if (solver == 'saga') or (loss == 'least_square'):
        n_jobs = 2
    else:
        n_jobs = 1

    n_features = X.shape[1]
    n_bootstraps = int(X_tilde.shape[1] / n_features)
    X_ko = np.column_stack([X, X_tilde])

    if loss == 'least_square':
        clf = LassoCV(n_jobs=n_jobs, max_iter=1e5, cv=cv)
        clf.fit(X_ko, y)
        coef = clf.coef_
    elif loss == 'logistic':
        clf = LogisticRegressionCV(
            penalty='l1', max_iter=1e5,
            solver=solver, cv=cv, n_jobs=n_jobs, tol=1e-8)
        clf.fit(X_ko, y)
        coef = clf.coef_[0]
    else:
        raise ValueError("'loss' must be either 'least_square' or 'logistic'")

    T = np.abs(coef).reshape(n_bootstraps + 1, n_features)
    T_sorted = np.sort(T, axis=0)
    kappa = np.argmax(T, axis=0)
    # correct kappa to not take into account of original W_i == 0
    for i in range(n_features):
        # check if all coefficient of a variable equal 0
        if not np.any(T[:, i]):
            kappa[i] = np.random.randint(1, n_bootstraps)
    tau = T_sorted[-1, :] - T_sorted[-2, :]

    if return_T:
        return kappa, tau, T

    return kappa, tau


def _knockoff_simultaneous_threshold(kappa, tau, n_bootstraps,
                                     offset=1, fdr=0.1):
    def _fdp_hat_kappa(t, offset, n_bootstraps):
        fp = offset + np.sum(np.logical_and(kappa >= 1, tau >= t))
        s_hat = np.maximum(1, np.sum(np.logical_and(kappa == 0, tau >= t)))
        return fp / (n_bootstraps * s_hat)

    tau_nonzero = np.sort(tau[tau.nonzero()])
    fdps = np.array(
        [_fdp_hat_kappa(t, offset, n_bootstraps) for t in tau_nonzero])
    under_index = np.where(fdps <= fdr)[0]

    if under_index.size == 0:
        threshold_ = np.inf
    else:
        threshold_ = tau_nonzero[under_index[0]]

    return threshold_


def knockoff_simultaneous(X, y, n_bootstraps=2, fdr=0.1,
                          centered=True, shrink=True,
                          offset=1,
                          cov_estimator='ledoit_wolf',
                          n_jobs=1,
                          method='equi',
                          verbose=False,
                          seed=None):
    if centered:
        X = StandardScaler().fit_transform(X)

    mu, Sigma = _estimate_distribution(
        X, shrink=shrink, cov_estimator=cov_estimator)
    if method == 'equi':
        X_tilde = _knockoff_equi_simultaneous(X, mu, Sigma, n_bootstraps, seed)
    else:
        raise ValueError(
            'Only equi-correlated knockoff is available in this version.')

    kappa, tau = _knockoff_stat_simultaneous(X, X_tilde, y, n_jobs=n_jobs)
    threshold_ = _knockoff_simultaneous_threshold(
        kappa, tau, n_bootstraps, offset, fdr)

    selected = np.logical_and(kappa == 0, tau >= threshold_)
    selected = np.where(selected)[0]

    if verbose:
        return selected, threshold_, tau, kappa, X_tilde

    return selected
