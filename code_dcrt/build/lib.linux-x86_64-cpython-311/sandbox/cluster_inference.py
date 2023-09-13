# -*- coding: utf-8 -*-
"""Data clustering version of some inference methods to reduce dimension.  In
particular: Distillation Conditional Randomization Test (dCRT), Knockoffs and
Desparsified Lasso

"""
import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_memory

from .dcrt import dcrt_zero, dcrt_zero_aggregation
from .desparsified_lasso import desparsified_lasso
from .fcd_inference import _dl_fdr_threshold
from .gaussian_knockoff import _estimate_distribution
from .knockoff_aggregation import knockoff_aggregation
from .knockoffs import model_x_knockoff
from .stat_coef_diff import _coef_diff_threshold
from .stat_lambda_path import _lambda_threshold
from .utils import (_cluster_data, _vote_sign, broadcast_coef_sign,
                    broadcast_metric, fdr_threshold, quantile_aggregation)


def cluster_dcrt(X, y, n_clusters=None, fdr=0.1, train_size=None,
                 connectivity=None, estimated_coef=None, screening=True,
                 screening_threshold=1e-1, Sigma_X=None, cv=5, alpha=None,
                 max_iter=1e3, use_cv=False, loss='least_square', refit=False,
                 centered=True, fdr_control='bhq', statistic='residual',
                 solver='liblinear', verbose=False, n_jobs=1, memory=None,
                 random_state=None):

    _, n_features = X.shape

    if n_clusters is None:
        return dcrt_zero(
            X, y, fdr=fdr, estimated_coef=estimated_coef, Sigma_X=Sigma_X,
            max_iter=max_iter, use_cv=use_cv, refit=refit, screening=screening,
            screening_threshold=screening_threshold, statistic=statistic,
            centered=centered, n_jobs=n_jobs, loss=loss, solver=solver,
            verbose=verbose, alpha=alpha)

    X_reduced, group_labels = _cluster_data(
        X, n_clusters, train_size=train_size, connectivity=connectivity,
        rand=random_state, memory=memory)

    # Sigma_X now has to be recomputed for clustered design matrix
    if Sigma_X is not None:
        _, Sigma_X = _estimate_distribution(X_reduced)

    mem = check_memory(memory)
    dcrt_zero_cached = mem.cache(dcrt_zero, ignore=['n_jobs', 'verbose'])

    _, pvals_group, Ts_group = \
        dcrt_zero_cached(X_reduced, y, estimated_coef=estimated_coef,
                         max_iter=max_iter, use_cv=use_cv, refit=refit,
                         screening=screening, Sigma_X=Sigma_X,
                         screening_threshold=screening_threshold,
                         statistic=statistic, centered=centered, n_jobs=n_jobs,
                         loss=loss, solver=solver, verbose=True, alpha=alpha)

    pvals = broadcast_metric(pvals_group, group_labels, n_features)
    Ts = broadcast_metric(Ts_group, group_labels, n_features)
    threshold = fdr_threshold(pvals, fdr=fdr, method=fdr_control)
    selected = np.where(pvals <= threshold)[0]

    if verbose:
        return selected, pvals, Ts

    return selected


def cluster_dcrt_aggregation(X, y, n_clusters=None, n_bootstraps=1,
                             connectivity=None, train_size=0.8, fdr=0.1,
                             fdr_control='bhq', Sigma_X=None, cv=5,
                             use_cv=False, refit=True, screening=True,
                             screening_threshold=1e-1, loss='least_square',
                             statistic='residual', gamma=0.5, gamma_min=0.05,
                             alpha=None, adaptive_aggregation=True,
                             centered=True, n_jobs=1, memory=None,
                             dcrt_n_jobs=1, joblib_verbose=0, verbose=False,
                             random_state=None, solver='liblinear',
                             max_iter=1e3):
    """Aggregation of p-values output by clustered resampling-free d0-CRT
    """

    if n_clusters is None and n_bootstraps > 1:
        return dcrt_zero_aggregation(
            X, y, fdr=fdr, n_bootstraps=n_bootstraps, Sigma_X=Sigma_X,
            cv=cv, use_cv=use_cv, loss=loss, verbose=verbose,
            gamma=gamma, gamma_min=gamma_min, centered=centered,
            adaptive_aggregation=adaptive_aggregation,
            n_jobs=n_jobs, dcrt_n_jobs=dcrt_n_jobs,
            joblib_verbose=joblib_verbose, train_size=train_size,
            memory=memory, random_state=random_state, alpha=alpha)

    if n_bootstraps == 1:
        return cluster_dcrt(
            X, y, n_clusters=n_clusters, connectivity=connectivity,
            train_size=train_size, fdr=fdr, Sigma_X=Sigma_X, cv=cv,
            centered=centered, alpha=alpha, use_cv=use_cv,
            loss=loss, verbose=verbose, n_jobs=n_jobs,
            memory=memory, random_state=random_state)

    mem = check_memory(memory)
    dcrt_zero_cached = mem.cache(dcrt_zero, ignore=['n_jobs', 'verbose'])

    rgen = check_random_state(random_state)
    rands = rgen.randint(1, np.iinfo(np.int32).max, n_bootstraps)

    parallel = Parallel(n_jobs, verbose=joblib_verbose)
    clusters = parallel(delayed(_cluster_data)(
        X, n_clusters, connectivity=connectivity, train_size=train_size,
        memory=memory, rand=rand) for rand in rands)

    Xs_reduced = [clusters[i][0] for i in range(n_bootstraps)]
    cluster_labels = [clusters[i][1] for i in range(n_bootstraps)]

    _, n_features = X.shape

    temp = parallel(delayed(dcrt_zero_cached)(
        X_reduced, y, max_iter=max_iter, use_cv=use_cv, refit=refit,
        screening=screening, Sigma_X=Sigma_X, cv=cv,
        screening_threshold=screening_threshold, statistic=statistic,
        centered=centered, n_jobs=dcrt_n_jobs, loss=loss, solver=solver,
        verbose=True, alpha=alpha) for X_reduced in Xs_reduced)

    pvals_group = [temp[i][1] for i in range(n_bootstraps)]
    pvals = np.array(parallel(delayed(
        broadcast_metric)(pval, label, n_features)
        for pval, label in zip(pvals_group, cluster_labels)))

    aggregated_pval = quantile_aggregation(
        pvals, gamma=gamma, gamma_min=gamma_min, adaptive=adaptive_aggregation)

    threshold = fdr_threshold(aggregated_pval, fdr=fdr, method=fdr_control)
    selected = np.where(aggregated_pval <= threshold)[0]

    if verbose:
        return selected, aggregated_pval, np.array(pvals)

    return selected


def cluster_knockoff(X, y, n_clusters, linkage='ward', connectivity=None,
                     train_size=None, centered=True, shrink=True,
                     loss='least_square', fdr=0.1, offset=1,
                     statistic='lasso_cv', fdr_control='bhq', joblib_verbose=0,
                     cov_estimator='ledoit_wolf', n_bootstraps=1,
                     adaptive_aggregation=False, gamma=0.5, verbose=False,
                     n_jobs=1, memory=None, random_state=None):

    n_features = X.shape[1]

    X_reduced, cluster_labels = _cluster_data(
        X, n_clusters, linkage=linkage, memory=memory,
        connectivity=connectivity, train_size=train_size,
        rand=random_state)

    # If no knockoff aggregation is used
    if n_bootstraps == 1:
        result = model_x_knockoff(X_reduced, y, fdr=fdr, offset=offset,
                                  statistics=statistic,
                                  cov_estimator=cov_estimator, verbose=True,
                                  memory=memory, n_jobs=n_jobs,
                                  centered=centered, shrink=shrink,
                                  seed=random_state)

        coef_signs = broadcast_coef_sign(
            result[0], X_reduced, y, cluster_labels, n_features, loss)

        test_scores = broadcast_metric(result[1], cluster_labels, n_features)

        if statistic == 'l1_regu_path':
            thres = _lambda_threshold(test_scores, fdr=fdr)
        else:
            thres = _coef_diff_threshold(test_scores, fdr=fdr, offset=offset)

        selected = np.where(test_scores >= thres)[0]

        if verbose:
            return selected, test_scores, coef_signs

        return selected

    # Using knockoff aggregation
    result = knockoff_aggregation(
        X_reduced, y, centered=centered, shrink=shrink,
        fdr=fdr, offset=offset, statistic=statistic,
        verbose=True, cov_estimator=cov_estimator,
        n_bootstraps=n_bootstraps, gamma=gamma,
        joblib_verbose=joblib_verbose, fdr_control=fdr_control,
        adaptive_aggregation=adaptive_aggregation,
        random_state=random_state, n_jobs=n_jobs, memory=memory)

    test_scores = broadcast_metric(result[1], cluster_labels, n_features)
    threshold = fdr_threshold(test_scores, fdr=fdr, method=fdr_control)
    selected = np.where(test_scores <= threshold)[0]
    coef_signs = broadcast_coef_sign(
        result[0], X_reduced, y, cluster_labels, n_features, loss)

    if verbose:
        return selected, test_scores, coef_signs

    return selected


def cluster_knockoff_aggregation(X, y, n_clusters, linkage='ward',
                                 connectivity=None, n_rand=25, n_bootstraps=1,
                                 fdr=0.1, offset=1, statistic='lasso_cv',
                                 cov_estimator='ledoit_wolf', adaptive=False,
                                 cluster_adaptive=False, train_size=0.7,
                                 gamma=0.5, gamma_min=0.05, verbose=False,
                                 fdr_control='bhq', single_aggregation=True,
                                 n_jobs=1, joblib_verbose=0, memory=None,
                                 random_state=None):

    rgen = check_random_state(random_state)
    rands = rgen.randint(1, np.iinfo(np.int32).max, n_rand)

    n_features = X.shape[1]

    parallel = Parallel(n_jobs, verbose=joblib_verbose)
    clusters = parallel(delayed(_cluster_data)(
        X, n_clusters, linkage=linkage, memory=memory,
        connectivity=connectivity, train_size=train_size,
        rand=rand) for rand in rands)

    Xs_reduced = [clusters[i][0] for i in range(n_rand)]
    cluster_labels = [clusters[i][1] for i in range(n_rand)]

    # devide workers for nested loop in between AKO & ECKO
    if n_rand == 1:
        n_jobs_ecko = n_rand
        n_jobs_ako = n_jobs
    elif n_bootstraps == 1:
        n_jobs_ecko = n_jobs
        n_jobs_ako = n_bootstraps
    else:
        n_jobs_ecko = int(np.sqrt(n_jobs))
        n_jobs_ako = int(np.sqrt(n_jobs))

    results = Parallel(n_jobs_ecko)(delayed(
        knockoff_aggregation)(
            X, y, offset=offset,
            n_jobs=n_jobs_ako,
            adaptive_aggregation=cluster_adaptive,
            statistic=statistic,
            fdr_control=fdr_control,
            cov_estimator=cov_estimator,
            random_state=0,
            verbose=True,
            memory=memory,
            n_bootstraps=n_bootstraps)
        for (X, parcel_label) in clusters)

    if single_aggregation and n_bootstraps > 1:
        # Only do one aggregation instead of nested aggregation inside 1
        # cluster and then aggregated between clusters
        test_scores = parallel(
            delayed(broadcast_metric)(
                results[i][-1][j, :], cluster_labels[i], n_features)
            for i in range(n_rand) for j in range(n_bootstraps))

        coef_signs = parallel(
            delayed(broadcast_coef_sign)(
                selected=np.where(results[i][-1][j, :] <= fdr)[0],
                X_reduced=Xs_reduced[i], y=y,
                cluster_labels=cluster_labels[i],
                n_features=n_features)
            for i in range(n_rand) for j in range(n_bootstraps))

        test_scores = np.array(test_scores)
        coef_signs = np.array(coef_signs)

    else:
        # Doing nested aggregation (inter-cluster and intra-cluster)
        test_scores = np.array([broadcast_metric(
            results[i][1], cluster_labels[i], n_features)
            for i in range(n_rand)]
        )
        coef_signs = [
            broadcast_coef_sign(selected=results[i][0],
                                X_reduced=Xs_reduced[i], y=y,
                                n_features=n_features,
                                cluster_labels=cluster_labels[i])
            for i in range(n_rand)]

    aggregated_scores = quantile_aggregation(
        test_scores, gamma=gamma, gamma_min=gamma_min, adaptive=adaptive)
    threshold = fdr_threshold(aggregated_scores, fdr=fdr, method=fdr_control)
    selected = np.where(aggregated_scores <= threshold)[0]
    aggregated_signs = _vote_sign(np.array(coef_signs), selected)

    if verbose:
        return selected, aggregated_scores, aggregated_signs

    return selected


def cluster_dlasso(X, y, n_clusters, connectivity=None, fdr=None,
                   fdr_control='bhq', tol=1e-4, method="lasso", c=0.01,
                   train_size=None, memory=None, verbose=False, random_state=0):

    _, n_features = X.shape

    X_reduced, labels = _cluster_data(X, n_clusters, train_size=train_size,
                                      connectivity=connectivity, memory=memory,
                                      rand=random_state)

    _, zscore_group, pval_group = desparsified_lasso(
        X_reduced, y, tol=tol, centered=True,
        method=method, c=c)

    pval = broadcast_metric(pval_group, labels, n_features)
    zscore = broadcast_metric(zscore_group, labels, n_features)

    # if fdr level is specified, return the threshold variables
    if type(fdr) == float:
        if fdr_control == 'bhq':
            threshold = fdr_threshold(pval, fdr=fdr)
            selected = np.where(pval <= threshold)[0]
        elif fdr_control == 'jj18':
            threshold = _dl_fdr_threshold(zscore, fdr=fdr, one_sided=False)
            selected = np.where(np.abs(zscore) >= threshold)[0]

        if verbose:
            return selected, threshold, pval, zscore

        return selected

    return pval, zscore


def cluster_dlasso_aggregation(X, y, n_clusters, connectivity=None, fdr=None,
                               tol=1e-4, method="lasso", c=0.01, train_size=0.8,
                               memory=None, gamma=0.5, gamma_min=0.05,
                               adaptive_aggregation=True, n_bootstraps=25,
                               n_jobs=1, verbose=False, joblib_verbose=0,
                               random_state=0):

    _, n_features = X.shape

    rgen = check_random_state(random_state)
    rands = rgen.randint(1, np.iinfo(np.int32).max, n_bootstraps)

    parallel = Parallel(n_jobs, verbose=joblib_verbose)
    clusters = parallel(delayed(_cluster_data)(
        X, n_clusters, connectivity=connectivity, train_size=train_size,
        memory=memory, rand=rand) for rand in rands)

    Xs_reduced = [clusters[i][0] for i in range(n_bootstraps)]
    cluster_labels = [clusters[i][1] for i in range(n_bootstraps)]

    mem = check_memory(memory)
    dl_cached = mem.cache(desparsified_lasso)

    temp = parallel(delayed(dl_cached)(
        X_reduced, y, tol=tol, method=method, c=c)
        for X_reduced in Xs_reduced)

    pvals_group = [temp[i][2] for i in range(n_bootstraps)]
    pvals = np.array(parallel(delayed(
        broadcast_metric)(pval, label, n_features)
        for pval, label in zip(pvals_group, cluster_labels)))

    aggregated_pval = quantile_aggregation(pvals, gamma=gamma,
                                           gamma_min=gamma_min,
                                           adaptive=adaptive_aggregation)

    if type(fdr) == float:
        threshold = fdr_threshold(aggregated_pval, fdr=fdr)
        selected = np.where(aggregated_pval <= threshold)[0]

        if verbose:
            return selected, threshold, aggregated_pval

        return selected

    return aggregated_pval
