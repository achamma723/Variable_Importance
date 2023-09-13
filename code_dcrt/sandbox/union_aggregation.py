import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_memory

from .gaussian_knockoff_fast import (_estimate_distribution,
                                     gaussian_knockoff_generation)
from .knockoffs import model_x_knockoff
from .stat_coef_diff import stat_coef_diff
from .utils import fdr_threshold, quantile_aggregation


def union_aggregation(
    X,
    y,
    centered=True,
    shrink=False,
    construct_method="equi",
    fdr=0.1,
    fdr_control="bhq",
    reshaping_function=None,
    offset=1,
    statistic="lasso_cv",
    cov_estimator="ledoit_wolf",
    joblib_verbose=0,
    n_bootstraps=10,
    n_jobs=1,
    weights=None,
    verbose=False,
    memory=None,
    random_state=None,
):
    # unnecessary to have n_jobs > number of bootstraps
    n_jobs = min(n_bootstraps, n_jobs)

    if centered:
        X = StandardScaler().fit_transform(X)

    mu, Sigma = _estimate_distribution(X, shrink=shrink, cov_estimator=cov_estimator)

    mem = check_memory(memory)
    stat_coef_diff_cached = mem.cache(
        stat_coef_diff, ignore=["n_jobs", "joblib_verbose"]
    )

    if isinstance(random_state, (int, np.int32, np.int64)):
        rng = check_random_state(random_state)
    elif random_state is None:
        rng = check_random_state(0)
    else:
        raise TypeError("Wrong type for random_state")

    seed_list = rng.randint(1, np.iinfo(np.int32).max, n_bootstraps)
    parallel = Parallel(n_jobs, verbose=joblib_verbose)
    X_tildes = parallel(
        delayed(gaussian_knockoff_generation)(
            X, mu, Sigma, method=construct_method, memory=memory, seed=seed
        )
        for seed in seed_list
    )

    ko_stats = parallel(
        delayed(stat_coef_diff_cached)(X, X_tildes[i], y, method=statistic)
        for i in range(n_bootstraps)
    )

    return True
