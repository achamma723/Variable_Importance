# -*- coding: utf-8 -*-
# Authors: Binh Nguyen <tuan-binh.nguyen@inria.fr>
"""
Implementation of Model-X knockoffs inference procedure, introduced in
Candes et. al. (2016) " Panning for Gold: Model-X Knockoffs for
High-dimensional Controlled Variable Selection"
(https://arxiv.org/abs/1610.02351)
"""
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_memory

from .gaussian_knockoff import (_estimate_distribution,
                                gaussian_knockoff_generation)
from .orig_imp import orig_imp_fit
from .permfit_py import permfit
from .stat_coef_diff import _coef_diff_threshold, stat_coef_diff
from .stat_lambda_path import _lambda_threshold, stat_lambda_path
from .utils import _empirical_pval, fdr_threshold, quantile_aggregation


def model_x_knockoff(X, y, fdr=0.1, offset=1, method='equi',
                     statistics='lasso_cv', use_signed_max=False,
                     solver='liblinear', shrink=False, centered=True,
                     cov_estimator='ledoit_wolf', verbose=False, memory=None,
                     n_jobs=1, joblib_verbose=0, seed=None, nb_knockoffs=1,
                     save_file="Best_model", prob_type="regression"):
    """Model-X Knockoff inference procedure to control False Discoveries Rate,
    based on Candes et. al. (2016)

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        design matrix

    y : 1D ndarray (n_samples, )
        response vector

    fdr : float, optional
        desired controlled FDR level

    offset : int, 0 or 1, optional
        offset to calculate knockoff threshold, offset = 1 is
        equivalent to knockoff+

    method : str, optional
        knockoff construction methods, either equi for equi-correlated
        knockoff or sdp for optimization scheme

    statistics : str, optional
        method to calculate knockoff test score

    solver : str, optional
        solver for the optimization problem to calculate knockoff
        statistics

    shrink : bool, optional
        whether to shrink the empirical covariance matrix

    centered : bool, optional
        whether to standardize the data before doing the inference
        procedure

    cov_estimator : str, optional
        method of empirical covariance matrix estimation

    seed : int or None, optional
        random seed used to generate Gaussian knockoff variable

    nb_knockoffs : int, optional
        The number of knockoffs to generate when dealing with the
        "DNN" statistics

    Returns
    -------
    selected : 1D array, int
        vector of index of selected variables

    test_score : 1D array, (n_features, )
        vector of test statistic

    thres : float
        knockoff threshold

    X_tilde : 2D array, (n_samples, n_features)
        knockoff design matrix
    """
    memory = check_memory(memory)

    if centered:
        X = StandardScaler().fit_transform(X)

    mu, Sigma = _estimate_distribution(
        X, shrink=shrink, cov_estimator=cov_estimator)

    X_tilde = gaussian_knockoff_generation(X, mu, Sigma, memory=memory,
                                           method=method, seed=seed)

    if statistics == 'l1_regu_path':
        test_score = stat_lambda_path(X, X_tilde, y, memory=memory)
        thres = _lambda_threshold(test_score, fdr=fdr)
    elif statistics in ['lasso_cv', 'logistic_l1']:
        test_score, pred_score = memory.cache(
            stat_coef_diff,
            ignore=['n_jobs', 'joblib_verbose'])(
                X, X_tilde, y, method=statistics, n_jobs=n_jobs, solver=solver,
                use_signed_max=use_signed_max)
        thres = _coef_diff_threshold(test_score, fdr=fdr, offset=offset)
    elif statistics == 'bart':
        X_ko = StandardScaler().fit_transform(np.column_stack([X, X_tilde]))
        return X_ko, X_tilde
    elif statistics == 'deep':
        n, p = X.shape
        if isinstance(seed, (int, np.int32, np.int64)):
            rng = check_random_state(seed)
        elif seed is None:
            rng = check_random_state(0)
        else:
            raise TypeError('Wrong type for random_state')

        seed_list = rng.randint(1, np.iinfo(np.int32).max, nb_knockoffs)
        parallel = Parallel(n_jobs, verbose=joblib_verbose)
        X_tildes = np.array(parallel(delayed(gaussian_knockoff_generation)(
            X, mu, Sigma, method=method, memory=memory, seed=sd) for sd in seed_list))
        X_tmp = np.array([np.vstack([X_tildes[i, j, :] for i in range(
            X_tildes.shape[0])]) for j in range(X_tildes.shape[1])])
        X_r = np.zeros((n, nb_knockoffs+1, p))
        X_r[:, 0, :] = X
        X_r[:, 1:, :] = X_tmp

        # X_r[:, 1, :] = np.zeros((n, p))
        # X_tmp = X_tildes[0, :, :]
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(np.dot(np.transpose(X), X), interpolation="nearest")
        # axs[0, 0].set_title('X^T X')
        # axs[0, 1].imshow(np.dot(np.transpose(X), X_tmp), interpolation="nearest")
        # axs[0, 1].set_title('X^T X^tilda')
        # axs[1, 0].imshow(np.dot(np.transpose(X_tmp), X), interpolation="nearest")
        # axs[1, 0].set_title('X^tilda^T X^T')
        # axs[1, 1].imshow(np.dot(np.transpose(X_tmp), X_tmp), interpolation="nearest")
        # axs[1, 1].set_title('X^tilda^T X^tilda')
        # fig.savefig('correlation.jpg')
        # exit(0)
        imp_vals = permfit(X_r, y, prob_type=prob_type, k_fold=0, verbose=2,
                           n_ensemble=10, save_file=save_file)
        # imp_vals = orig_imp_fit(X_r, y)
        return imp_vals
    selected = np.where(test_score >= thres)[0]

    if verbose:
        return selected, test_score, pred_score, thres, X_tilde

    return selected


def knockoff_aggregation(X, y, centered=True, shrink=False,
                         construct_method='equi', fdr=0.1, fdr_control='bhq',
                         reshaping_function=None, offset=1,
                         statistic='lasso_cv', cov_estimator='ledoit_wolf',
                         joblib_verbose=0, n_bootstraps=25, n_jobs=1,
                         adaptive_aggregation=False, gamma=0.5, gamma_min=0.05,
                         aggregation_method='empirical_pval', verbose=False,
                         memory=None, random_state=None):
    """Aggregation of Multiple Knockoffs to control False Discoveries Rate,
    based on Nguyen et. al. (2020) <https://arxiv.org/abs/2002.09269>

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        design matrix

    y : 1D ndarray (n_samples, )
        response vector

    fdr : float, optional
        desired controlled FDR level

    fdr_control : str, optional
        method for controlling FDR

    offset : int, 0 or 1, optional
        offset to calculate knockoff threshold, offset = 1 is equivalent to
        knockoff+

    n_bootstraps : int, optional
        number of knockoff bootstraps

    construct_method : str, optional
        knockoff construction methods, either equi for equi-correlated knockoff
        or sdp for optimization scheme

    statistics : str, optional
        method to calculate knockoff test score

    adaptive_aggregation : bool, optional
        whether to use adaptive quantile aggregation scheme for bootstrapping 
        p-values, for more info see Meinhausen et al. (2009)
        <https://www.tandfonline.com/doi/abs/10.1198/jasa.2009.tm08647>

    shrink : bool, optional
        whether to shrink the empirical covariance matrix

    centered : bool, optional
        whether to standardize the data before doing the inference procedure

    cov_estimator : str, optional
        method of empirical covariance matrix estimation

    n_jobs : int, optional
        number of parallel jobs to run, increase this number will make the
        inference faster, but take more computational resource

    random_state : int or None, optional
        random seed used to generate Gaussian knockoff variable

    Returns
    -------
    selected : 1D array, int
        vector of index of selected variables

    aggregated_pval : 1D array, float
        vector of aggregated pvalues

    pvals : 2D array, float, (n_bootstraps, n_features)
        list of bootrapping pvalues

    """
    # unnecessary to have n_jobs > number of bootstraps
    n_jobs = min(n_bootstraps, n_jobs)

    if centered:
        X = StandardScaler().fit_transform(X)

    mu, Sigma = _estimate_distribution(
        X, shrink=shrink, cov_estimator=cov_estimator)

    mem = check_memory(memory)
    stat_coef_diff_cached = mem.cache(stat_coef_diff,
                                      ignore=['n_jobs', 'joblib_verbose'])

    if n_bootstraps == 1:
        X_tilde = gaussian_knockoff_generation(
            X, mu, Sigma, method=construct_method,
            memory=memory, seed=random_state)
        ko_stat = stat_coef_diff_cached(X, X_tilde, y, method=statistic)
        pvals = _empirical_pval(ko_stat, offset)
        threshold = fdr_threshold(pvals, fdr=fdr,
                                  method=fdr_control)
        selected = np.where(pvals <= threshold)[0]

        if verbose:
            return selected, pvals

        return selected

    if isinstance(random_state, (int, np.int32, np.int64)):
        rng = check_random_state(random_state)
    elif random_state is None:
        rng = check_random_state(0)
    else:
        raise TypeError('Wrong type for random_state')

    seed_list = rng.randint(1, np.iinfo(np.int32).max, n_bootstraps)
    parallel = Parallel(n_jobs, verbose=joblib_verbose)
    X_tildes = parallel(delayed(gaussian_knockoff_generation)(
        X, mu, Sigma, method=construct_method, memory=memory,
        seed=seed) for seed in seed_list)

    ko_stats = parallel(delayed(stat_coef_diff_cached)(
        X, X_tildes[i], y, method=statistic) for i in range(n_bootstraps))

    if aggregation_method == 'empirical_pval':
        pvals = np.array([_empirical_pval(ko_stats[i], offset)
                          for i in range(n_bootstraps)])

        aggregated_pval = quantile_aggregation(
            pvals, gamma=gamma, gamma_min=gamma_min,
            adaptive=adaptive_aggregation)

        threshold = fdr_threshold(aggregated_pval, fdr=fdr, method=fdr_control,
                                  reshaping_function=reshaping_function)
        selected = np.where(aggregated_pval <= threshold)[0]

        if verbose:
            return selected, aggregated_pval, pvals

    # Based on Xie and Lederer (2019) <https://arxiv.org/abs/1907.03807>
    elif aggregation_method == 'union_bound':
        if n_bootstraps > 10 or n_bootstraps < 5:
            warnings.warn(
                'The numbers of bootstraps should be in [5, 10]'
                ' as indicated in the paper')

        q = fdr / n_bootstraps
        thresholds = [_coef_diff_threshold(ko_stats[i], q, offset)
                      for i in range(n_bootstraps)]
        list_selected = [np.where(ko_stats[i] >= thresholds[i])[0]
                         for i in range(n_bootstraps)]
        # final selection is union of selected list
        selected = np.unique(np.concatenate(list_selected))
        if verbose:
            return selected, ko_stats, list_selected, thresholds

    else:
        raise ValueError(
            f'{aggregation_method} is not a valid aggregation scheme.')

    return selected
