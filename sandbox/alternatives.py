import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

from .gaussian_knockoff import (_estimate_distribution,
                                gaussian_knockoff_generation)
from .knockoff_simultaneous import _knockoff_equi_simultaneous
from .stat_lambda_path import stat_lambda_path


def holden_knockoff_aggregation(X, y, n_bootstraps=12,
                                fdr=0.1, offset=0,
                                cov_estimator='ledoit_wolf',
                                construct_method='equi',
                                loss='least_square',
                                random_state=None,
                                centered=True,
                                shrink=True,
                                verbose=False,
                                n_jobs=1):
    """
    Following idea of Holden et. al. 2018 - Multiple Model-Free Knockoffs
    Default offset for fdr threshold is 0, different from vanilla KO
    """
    holden_statistics = _holden_knockoff_statistic(
        X, y, n_bootstraps=n_bootstraps,
        cov_estimator=cov_estimator,
        loss=loss,
        construct_method=construct_method,
        random_state=random_state,
        centered=centered,
        shrink=shrink,
        n_jobs=n_jobs
    )

    threshold = _holden_knockoff_threshold(holden_statistics,
                                           n_bootstraps=n_bootstraps,
                                           fdr=fdr, offset=offset)
    W_true = holden_statistics[0]
    selected = np.where(W_true >= threshold)[0]
    if verbose:
        return selected, W_true, threshold

    return selected


def _holden_knockoff_statistic(X, y, n_bootstraps=12,
                               construct_method='equi',
                               cov_estimator='ledoit_wolf',
                               loss='least_square',
                               random_state=None,
                               centered=True,
                               shrink=True,
                               n_jobs=1):

    _, n_features = X.shape
    # unnecessary to have n_jobs > number of k = (2 * bootstraps - 1) or total
    # number of knockoff bootstraps (follow the paper)
    n_bootstraps = 2 * n_bootstraps - 1
    n_jobs = min(n_bootstraps, n_jobs)
    parallel = Parallel(n_jobs=n_jobs)

    if centered:
        X = StandardScaler().fit_transform(X)

    mu, Sigma = _estimate_distribution(
        X, shrink=shrink, cov_estimator=cov_estimator)

    if isinstance(random_state, (int, np.int32, np.int64)):
        rng = check_random_state(random_state)
        seeds = rng.randint(1, np.iinfo(np.int32).max, n_bootstraps)
        X_tildes = parallel(delayed(gaussian_knockoff_generation)(
            X, mu, Sigma, method=construct_method,
            seed=seed) for seed in seeds)
    elif random_state is None:
        X_tildes = parallel(delayed(gaussian_knockoff_generation)(
            X, mu, Sigma, method=construct_method,
            seed=seed) for seed in range(n_bootstraps))
    else:
        raise TypeError('Wrong type for random_state')

    X_kos = [np.column_stack([X, X_tildes[i]])
             for i in range(n_bootstraps)]

    if loss == 'least_square':
        clf = LassoCV(cv=5, max_iter=1e4)
        clfs = parallel(delayed(clone(clf).fit)(
            X_kos[i], y) for i in range(n_bootstraps))
        coefs = [clfs[i].coef_ for i in range(n_bootstraps)]

    elif loss == 'logistic':
        clf = LogisticRegressionCV(
            penalty='l1', max_iter=1e4,
            solver='liblinear', cv=5, tol=1e-6)
        clfs = parallel(delayed(clone(clf).fit)(
            X_kos[i], y) for i in range(n_bootstraps))
        coefs = [clfs[i].coef_[0] for i in range(n_bootstraps)]
    else:
        raise ValueError

    middle_index = int((n_bootstraps + 1) / 2)
    abs_true_coef = np.abs(coefs[0][:n_features])
    abs_knockoff_coefs = [
        np.abs(coefs[i][n_features:]) for i in range(1, middle_index)]
    compared_knockoff_coefs = np.mean(
        [coefs[i][n_features:] for i in range(middle_index, n_bootstraps)], 0)

    W_true = abs_true_coef - compared_knockoff_coefs
    W_ko = np.ravel([abs_coef - compared_knockoff_coefs
                     for abs_coef in abs_knockoff_coefs])

    return W_true, W_ko


def _holden_knockoff_threshold(holden_statistics, n_bootstraps=12,
                               fdr=0.1, offset=1):
    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    W_true, W_ko = holden_statistics
    threshold = np.inf
    t_grid = np.sort(np.abs(W_true[W_true != 0]))

    for t in t_grid:
        fp = offset + np.sum(W_ko >= t)
        s = np.sum(W_true >= t)
        fdp = fp / (s * (n_bootstraps - 1))
        if fdp <= fdr:
            threshold = t
            break

    return threshold


def emery_knockoff_aggregation(X, y, fdr=0.1, offset=1,
                               threshold_method='max',
                               verbose=False,
                               n_bootstraps=3, seed=None):
    """Procedure following Emery and Keich (2019); with threshold_method to
    control c and lambda defined in beginning of page 11.
    """
    if threshold_method == 'mirror':
        c = lamb = 0.5
    elif threshold_method == 'max':
        c = lamb = 1 / (n_bootstraps + 1)
    else:
        raise ValueError('invalid threshold_method')

    n_features = X.shape[1]
    mu, Sigma = _estimate_distribution(X)
    X_tilde = _knockoff_equi_simultaneous(
        X, mu, Sigma,
        n_bootstraps=n_bootstraps, seed=seed)
    ko_scores = stat_lambda_path(X, X_tilde, y).reshape(-1, n_features).T
    selected, threshold = _emery_knockoff_threshold(
        ko_scores, fdr=fdr, offset=offset, c=c, lamb=lamb)

    if verbose:
        return selected, ko_scores, threshold, X_tilde

    return selected


def _emery_knockoff_threshold(statistics, fdr=0.1, offset=1, c=0.5, lamb=0.5):
    """FDR control threshold with decoy follow Emery et al. (2019)
    statistics: ndarray (n_features, n_bootstraps + 1)
    """

    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    # define these variables follow section 4 of the paper
    n_features, d1 = statistics.shape  # d1 = n_bootstraps + 1
    ic = int(c * d1)
    i_lambda = int(lamb * d1)

    def find_rank_original_score(score_vectors):
        # score_vectors: (n_bootstraps + 1, )
        # return rank of original score each
        return int(rankdata(score_vectors, method='min')[0])

    def calculate_L(r, d1, ic, i_lambda):
        # r: rank of original score in the vector of scores bootstraps in the
        # statistics
        if r >= d1 - ic + 1:
            return 1  # original win
        elif r < d1 - ic + 1 and r > d1 - i_lambda:
            return 0  # ignored hypythesis
        else:
            return -1  # decoy/knockoff win

    def assign_score(statistics, L):
        """
        statistics: (n_features, n_bootstraps + 1)
        L: n_features: vector of association
        """
        ascore = []
        for i in range(len(L)):
            if L[i] == 1:
                ascore.append(statistics[i, 0])
            elif L[i] == 0:
                ascore.append(np.random.choice(statistics[i, (d1 - i):]))
            else:
                # assign max of knockoff score when L == -1
                ascore.append(np.max(statistics[i, 1:]))

        return np.array(ascore)

    original_ranks = np.array([find_rank_original_score(statistics[i, :])
                               for i in range(n_features)])
    Ls = np.array([calculate_L(original_ranks[i], d1, ic, i_lambda)
                   for i in range(n_features)])

    ascore = assign_score(statistics, Ls)
    sorted_ascore_index = np.argsort(-ascore)  # index when sort decreasing
    Ls_sorted = Ls[sorted_ascore_index]

    # step down procedure
    count = n_features - 1
    found = False
    while (not found) and (count >= 0):
        temp = Ls_sorted[:count]  # consider only j <= i
        count -= 1
        fp = offset + np.sum(temp == -1)
        R = np.maximum(np.sum(temp == 1), 1)
        fdphat = fp / R * c / (1 - lamb)
        if fdphat <= fdr:
            found = True

    if found:
        threshold = count
        selected = sorted_ascore_index[
            np.where(Ls_sorted[:threshold] == 1)[0]]
    else:
        threshold = None
        selected = np.array([])

    return selected, threshold
