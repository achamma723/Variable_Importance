# -*- coding: utf-8 -*-
# Authors: Binh Nguyen <tuan-binh.nguyen@inria.fr> and Jerome-Alexis Chevalier
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_memory

from .gaussian_knockoff import (_estimate_distribution,
                                gaussian_knockoff_generation)
from .stat_coef_diff import _coef_diff_threshold, stat_coef_diff
from .utils import _empirical_pval, fdr_threshold, quantile_aggregation


def knockoff_aggregation(
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
    n_bootstraps=25,
    n_jobs=1,
    adaptive_aggregation=False,
    gamma=0.5,
    gamma_min=0.05,
    aggregation_method="empirical_pval",
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

    if n_bootstraps == 1:
        X_tilde = gaussian_knockoff_generation(
            X, mu, Sigma, method=construct_method, memory=memory, seed=random_state
        )
        ko_stat = stat_coef_diff_cached(X, X_tilde, y, method=statistic)
        pvals = _empirical_pval(ko_stat, offset)
        threshold = fdr_threshold(pvals, fdr=fdr, method=fdr_control)
        selected = np.where(pvals <= threshold)[0]

        if verbose:
            return selected, pvals

        return selected

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

    if aggregation_method == "empirical_pval":
        pvals = np.array(
            [_empirical_pval(ko_stats[i], offset) for i in range(n_bootstraps)]
        )

        aggregated_pval = quantile_aggregation(
            pvals, gamma=gamma, gamma_min=gamma_min, adaptive=adaptive_aggregation
        )

        threshold = fdr_threshold(
            aggregated_pval,
            fdr=fdr,
            method=fdr_control,
            reshaping_function=reshaping_function,
        )
        selected = np.where(aggregated_pval <= threshold)[0]

        if verbose:
            return selected, aggregated_pval, pvals

    # Based on Xie and Lederer (2019) <https://arxiv.org/abs/1907.03807>
    elif aggregation_method == "union_bound":
        if n_bootstraps > 10 or n_bootstraps < 5:
            warnings.warn(
                "The numbers of bootstraps should be in [5, 10]"
                " as indicated in the paper"
            )

        q = fdr / n_bootstraps
        thresholds = [
            _coef_diff_threshold(ko_stats[i], q, offset) for i in range(n_bootstraps)
        ]
        list_selected = [
            np.where(ko_stats[i] >= thresholds[i])[0] for i in range(n_bootstraps)
        ]
        # final selection is union of selected list
        selected = np.unique(np.concatenate(list_selected))
        if verbose:
            return selected, ko_stats, list_selected, thresholds

    else:
        raise ValueError(f"{aggregation_method} is not a valid aggregation scheme.")

    return selected
