import numpy as np
from group_lasso import GroupLasso
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso, LinearRegression, MultiTaskLassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state, resample
from sklearn.utils.validation import check_memory

from .gaussian_knockoff import (_estimate_distribution,
                                gaussian_knockoff_generation)
from .stat_coef_diff import _coef_diff_threshold
from .utils import (_cluster_data, _empirical_pval, fdr_threshold,
                    quantile_aggregation)


def cluster_group_lasso_cv(X, y, n_clusters, groups=None, linkage='ward',
                           connectivity=None, use_sklearn_backend=False,
                           do_clustering=True, joblib_verbose=0, memory=None,
                           train_size=0.7, group_reg=1e-3, use_l1_reg=False,
                           n_iter=100, warm_start=True, l1_reg=5e-2, tol=1e-5,
                           cv=5, n_jobs=1, rand=None):

    if do_clustering or groups is None and not use_sklearn_backend:
        _, groups = _cluster_data(X, n_clusters, train_size, linkage=linkage,
                                  memory=memory, connectivity=connectivity,
                                  rand=rand)

    if use_sklearn_backend:
        clf = MultiTaskLassoCV(cv=cv, n_alphas=10, n_jobs=n_jobs,
                               verbose=joblib_verbose, random_state=rand)
        clf.fit(X, y.reshape(-1, 1))

        return np.ravel(clf.coef_)

    estimator = GroupLasso(groups=groups, group_reg=group_reg, l1_reg=l1_reg,
                           warm_start=warm_start, n_iter=n_iter, tol=tol,
                           supress_warning=True, fit_intercept=False)

    if use_l1_reg:
        param_grid = dict(group_reg=np.logspace(-5, 5, 5),
                          l1_reg=np.logspace(-5, 5, 5))
    else:
        param_grid = dict(group_reg=np.logspace(-5, 5, 5))

    clf = GridSearchCV(estimator=estimator, cv=cv, n_jobs=n_jobs,
                       verbose=joblib_verbose,
                       scoring='neg_mean_squared_error', param_grid=param_grid)
    clf.fit(X, y.reshape(-1, 1))

    return np.ravel(clf.best_estimator_.coef_)


def clustered_group_lasso_inference(X, y, n_clusters, linkage='ward',
                                    connectivity=None, fdr=0.1, offset=1,
                                    use_sklearn_backend=False,
                                    use_l1_reg=False, warm_start=True,
                                    n_iter=1e2, cov_estimator='ledoit_wolf',
                                    shrink=False, joblib_verbose=0,
                                    memory=None, verbose=False, n_jobs=1,
                                    seed=None):
    n_features = X.shape[1]
    mu, Sigma = _estimate_distribution(X, shrink, cov_estimator)
    X_tilde = gaussian_knockoff_generation(
        X, mu, Sigma, seed=seed, memory=memory)

    X_ko = np.column_stack([X, X_tilde])
    mem = check_memory(memory)

    gr_coef = mem.cache(
        cluster_group_lasso_cv, ignore=['n_jobs', 'joblib_verbose'])(
            X_ko, y, n_clusters, linkage=linkage, connectivity=connectivity,
            use_sklearn_backend=use_sklearn_backend,
            warm_start=warm_start, n_iter=n_iter,
            use_l1_reg=use_l1_reg, groups=None, do_clustering=True,
            n_jobs=n_jobs, joblib_verbose=joblib_verbose)

    gr_test_score = np.abs(gr_coef[:n_features]) - np.abs(gr_coef[n_features:])
    gr_threshold = _coef_diff_threshold(gr_test_score, fdr=fdr, offset=offset)
    gr_selected = np.where(gr_test_score > gr_threshold)[0]

    if verbose:
        return gr_selected, gr_test_score

    return gr_selected


def ensemble_clustered_group_lasso(X, y, n_clusters, linkage='ward',
                                   connectivity=None, fdr=0.1, offset=1,
                                   use_l1_reg=False, use_sklearn_backend=False,
                                   adaptive=False, gamma=0.5, n_jobs=1,
                                   cov_estimator='ledoit_wolf',
                                   joblib_verbose=0, fdr_control='bhq',
                                   warm_start=True, n_iter=200, verbose=False,
                                   memory=None, n_rand=25, random_state=None):

    if isinstance(random_state, int):
        rng = check_random_state(random_state)
        rands = rng.randint(1, np.iinfo(np.int32).max, n_rand)
    else:
        rands = np.arange(n_rand)

    n_jobs = n_jobs if n_jobs < n_rand else n_rand
    parallel = Parallel(n_jobs=n_jobs, verbose=joblib_verbose)

    temp = parallel(delayed(clustered_group_lasso_inference)(
        X, y, n_clusters, linkage=linkage, connectivity=connectivity,
        fdr=fdr, memory=memory, use_l1_reg=use_l1_reg,
        cov_estimator=cov_estimator, warm_start=warm_start, n_iter=n_iter,
        use_sklearn_backend=use_sklearn_backend,
        seed=rand, verbose=True) for rand in rands)

    test_scores = [temp[i][1] for i in range(n_rand)]
    pvals = parallel(delayed(_empirical_pval)(test_score, offset=offset)
                     for test_score in test_scores)

    aggregated_pvals = quantile_aggregation(np.array(pvals), gamma=gamma,
                                            adaptive=adaptive)
    threshold = fdr_threshold(aggregated_pvals, fdr=fdr, method=fdr_control)
    selected = np.where(aggregated_pvals <= threshold)[0]

    if verbose:
        return selected, test_scores, aggregated_pvals

    return selected


class GroupLassoRefit(GroupLasso):
    def __init__(self, groups=None, group_reg=0.001, l1_reg=0.001,
                 n_iter=1000, tol=1e-5, subsampling_scheme=None,
                 fit_intercept=False, frobenius_lipschitz=False,
                 random_state=None, warm_start=False):
        super(GroupLassoRefit, self).__init__(
            groups=groups,
            group_reg=group_reg,
            l1_reg=l1_reg,
            n_iter=n_iter,
            tol=tol,
            subsampling_scheme=subsampling_scheme,
            fit_intercept=fit_intercept,
            frobenius_lipschitz=frobenius_lipschitz,
            random_state=random_state,
            warm_start=warm_start,
        )

    def initial_fit(self, X, y):
        n_features = X.shape[1]
        self.groups = np.array([i for i in range(n_features)])
        super().fit(X, y.reshape(-1, 1))

    def fit(self, X, y):
        self.initial_fit(X, y)
        if np.sum(self.coef_ != 0) > 0:
            coef = self.coef_.ravel()
            lr = LinearRegression(
                fit_intercept=False,
                normalize=True)
            lr.fit(X[:, coef != 0], y)
            debiased_coef = np.zeros(X.shape[1])
            debiased_coef[coef != 0] = lr.coef_
            self.coef_ = debiased_coef.reshape(-1, 1)


class LassoRefit(Lasso):
    def __init__(self, alpha=1.0, fit_intercept=False,
                 normalize=False, precompute=False, copy_X=True,
                 max_iter=1e4, tol=1e-4, warm_start=False,
                 positive=False, random_state=None, selection='cyclic'):
        super(Lasso, self).__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            precompute=precompute,
            copy_X=copy_X,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            positive=positive,
            random_state=random_state,
            selection=selection
        )

    def initial_fit(self, X, y):
        super().fit(X, y)

    def fit(self, X, y):
        self.initial_fit(X, y)
        if np.sum(self.coef_ != 0) > 0:
            coef = self.coef_
            lr = LinearRegression(
                fit_intercept=self.fit_intercept,
                normalize=self.normalize)
            lr.fit(X[:, coef != 0], y)
            debiased_coef = np.zeros(X.shape[1])
            debiased_coef[coef != 0] = lr.coef_
            self.coef_ = debiased_coef
