# -*- coding: utf-8 -*-
# Author: Binh Nguyen <tuan-binh.nguyen@inria.fr> & Jerome-Alexis Chevalier
import numpy as np
from scipy.stats import norm
from sklearn.cluster import FeatureAgglomeration
from sklearn.linear_model import (LassoCV, LassoLarsCV, LinearRegression,
                                  LogisticRegressionCV)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from .inflate_non_null import _inflate_non_null


def quantile_aggregation(pvals, gamma=0.5, gamma_min=0.05, adaptive=False):
    if adaptive:
        return _adaptive_quantile_aggregation(pvals, gamma_min)

    return _fixed_quantile_aggregation(pvals, gamma)


def fdr_threshold(pvals, fdr=0.1, method='bhq', reshaping_function=None):
    if method == 'bhq':
        return _bhq_threshold(pvals, fdr=fdr)
    elif method == 'bhy':
        return _bhy_threshold(
            pvals, fdr=fdr, reshaping_function=reshaping_function)
    elif method == 'adapt':
        return _adapt_threshold(pvals, fdr=fdr)
    else:
        raise ValueError(
            '{} is not support FDR control method'.format(method))


def cal_fdp_power(selected, non_zero_index, delta=0, n_features=None,
                  r_index=False):
    """ Calculate power and False Discovery Proportion

    Parameters
    ----------
    selected: list index (in R format) of selected non-null variables
    non_zero_index: true index of non-null variables
    r_index : True if the index is taken from rpy2 inference
    delta: FDR spatial tolerance (e.g. for cluster inference)

    Returns
    -------
    fdp: (delta-)False Discoveries Proportion
    power: percentage of correctly selected variables over total number of
        non-null variables

    """
    # selected is the index list in R and will be different from index of
    # python by 1 unit

    if selected.size == 0:
        return 0.0, 0.0
    
    n_positives = len(non_zero_index)

    if r_index:
        selected = selected - 1

    if delta > 0:
        non_zero_index = _inflate_non_null(non_zero_index, delta, n_features)

    true_positive = np.intersect1d(selected, non_zero_index)
    false_positive = np.setdiff1d(selected, true_positive)

    fdp = len(false_positive) / max(1, len(selected))
    power = min(len(true_positive), n_positives) / n_positives

    return fdp, power


def cal_dir_fdp_power(selected, sign_estimated, true_sign, non_zero_index,
                      r_index=True):
    """Calculate directional FDR and Power

    Parameters
    ----------
    selected : list index (in R format) of selected non-null variables
    sign_estimated :
    true_sign :
    non_zero_index :
    r_index : True if the index is taken from rpy2 inference

    Returns
    -------

    """
    # selected is the index list in R and will be different from index of
    # python by 1 unit
    if r_index:
        selected = selected - 1
    false_positive = np.sum(np.not_equal(sign_estimated[selected],
                                         true_sign[selected]))
    true_positive = np.sum(np.equal(sign_estimated[selected],
                                    true_sign[selected]))
    dir_fdp = false_positive / max(len(selected), 1)
    dir_power = true_positive / max(len(non_zero_index), 1)
    return dir_fdp, dir_power


def broadcast_metric(metrics, cluster_labels, n_features):

    broadcasted = [metrics[cluster_labels[i]] for i in range(n_features)]
    return np.array(broadcasted)


def broadcast_coef_sign(selected, X_reduced, y, cluster_labels, n_features,
                        loss='least_square'):

    coef_signs = np.zeros(n_features)

    if selected.size != 0:
        if loss == 'least_square':
            clf = LinearRegression(fit_intercept=False, normalize=True)
            clf.fit(X_reduced[:, selected], y)
            coef = clf.coef_
        elif loss == 'logistic':
            clf = LogisticRegressionCV(cv=3, tol=1e-6, max_iter=1e5)
            clf.fit(X_reduced[:, selected], y)
            coef = clf.coef_[0]

        coef_signs[selected] = np.sign(coef)

    assigned = broadcast_metric(coef_signs, cluster_labels, n_features)

    return np.array(assigned)


def zscore_logistic(data, labels):
    """Follow
    <https://stats.stackexchange.com/questions/89484/how-to-compute-the-standard-errors-of-a-logistic-regressions-coefficients>
    Covariance matrix of regression coefficient is (X.T * V * X)^(-1)

    Parameters
    ----------
    data
    labels

    Returns
    -------
    """

    nrows, ncols = data.shape
    data_with_intercept = np.hstack([np.ones((nrows, 1)), data])
    lr_estimator = LogisticRegressionCV(Cs=9, fit_intercept=False,
                                        tol=1e-08, cv=10,
                                        penalty='l1', solver='liblinear',
                                        max_iter=10000)
    lr_estimator.fit(data_with_intercept, labels)
    pred = lr_estimator.predict_proba(data_with_intercept)
    V = np.diag((pred[:, 0] * pred[:, 1]))
    coef_sigma = np.linalg.inv(
        np.dot(data_with_intercept.T, np.dot(V, data_with_intercept)))
    coef_std_error = np.sqrt(np.diag(coef_sigma))
    zscore = np.array([lr_estimator.coef_[0][i] / coef_std_error[i] for i in
                       range(ncols + 1)])
    return zscore[1:]


def zscore_ols(data, labels):
    """Calculate zscore of statistical selected variables only

    Parameters
    ----------
    data
    labels

    Returns
    -------

    """
    n_samples, _ = data.shape
    ols = LinearRegression(fit_intercept=False, normalize=True)
    ols.fit(data, labels)

    sigma_hat = np.sum((labels - data.dot(ols.coef_) - ols.intercept_ *
                        np.ones(n_samples)) ** 2) / n_samples
    inv_diag_data = np.diag(np.linalg.inv(data.T.dot(data)))
    se_beta = np.sqrt(sigma_hat * inv_diag_data)
    zscore = ols.coef_ / se_beta
    return zscore


def _reid(X, y, method="lars", tol=1e-6, max_iter=1e+3):
    """Estimation of noise standard deviation using Reid procedure

    Parameters
    -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary
        method : string, optional
            The method for the CV-lasso: "lars" or "lasso"
        tol : float, optional
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``.
        """

    X = np.asarray(X)
    n_samples, n_features = X.shape

    if int(max_iter / 5) <= n_features:
        max_iter = n_features * 5

    if method == "lars":
        clf_lars_cv = LassoLarsCV(max_iter=max_iter, normalize=False, cv=3)
        clf_lars_cv.fit(X, y)
        error = clf_lars_cv.predict(X) - y
        support = sum(clf_lars_cv.coef_ != 0)

    elif method == "lasso":
        clf_lasso_cv = LassoCV(tol=tol, max_iter=max_iter, cv=3)
        clf_lasso_cv.fit(X, y)
        error = clf_lasso_cv.predict(X) - y
        support = sum(clf_lasso_cv.coef_ != 0)

    sigma_hat = np.sqrt((1. / (n_samples - support))
                        * np.linalg.norm(error) ** 2)

    return sigma_hat


def pval_from_zscore(t_stat, distrib='Norm'):
    """p-values from z-scores

    Parameters
    -----------
        t_stat : float
            z-score values
    """
    n_features = t_stat.size

    if distrib == 'Norm':
        pval = 2 * norm.sf(np.abs(t_stat))

    pval_corr = np.minimum(1, pval * n_features)

    return pval, pval_corr


def zscore_from_sf(sf, distrib='Norm'):
    """z-scores from survival function values

    Parameters
    -----------
        sf : float
            Survival function values
    """
    if distrib == 'Norm':
        t_stat = norm.isf(sf)

    return t_stat


def pval_from_sf(sf, distrib='Norm'):
    """z-scores from survival function values

    Parameters
    -----------
        sf : float
            Survival function values
    """
    pval = pval_from_zscore(
        zscore_from_sf(sf, distrib=distrib), distrib=distrib)

    return pval


def _bhq_threshold(pvals, fdr=0.1):
    """Standard Benjamini-Hochberg for controlling False discovery rate
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    for i in range(n_features - 1, -1, -1):
        if pvals_sorted[i] <= fdr * (i + 1) / n_features:
            selected_index = i
            break
    if selected_index <= n_features:
        return pvals_sorted[selected_index]
    else:
        return -1.0


def _bhy_threshold(pvals, reshaping_function=None, fdr=0.1):
    """Benjamini-Hochberg-Yekutieli procedure for controlling FDR, with input
    shape function. Reference: Ramdas et al (2017)
    """
    n_features = len(pvals)
    pvals_sorted = np.sort(pvals)
    selected_index = 2 * n_features
    # Default value for reshaping function -- defined in
    # Benjamini & Yekutieli (2001)
    if reshaping_function is None:
        temp = np.arange(n_features)
        sum_inverse = np.sum(1 / (temp + 1))
        return _bhq_threshold(pvals, fdr / sum_inverse)
    else:
        for i in range(n_features - 1, -1, -1):
            if pvals_sorted[i] <= fdr * reshaping_function(i + 1) / n_features:
                selected_index = i
                break
        if selected_index <= n_features:
            return pvals_sorted[selected_index]
        else:
            return -1.0


def _adapt_threshold(pvals, fdr=0.1):
    """FDR controlling with AdaPT procedure (Lei & Fithian '18), in particular
    using the intercept only version, shown in Wang & Janson '20 section 3
    """
    pvals_sorted = pvals[np.argsort(-pvals)]

    for pv in pvals_sorted:
        false_pos = np.sum(pvals >= 1 - pv)
        selected = np.sum(pvals <= pv)
        if (1 + false_pos) / np.maximum(1, selected) <= fdr:
            return pv

    return -1.0


def _fixed_quantile_aggregation(pvals, gamma=0.5):
    """Quantile aggregation function based on Meinshausen et al (2008)

    Parameters
    ----------
    pvals : 2D ndarray (n_bootstrap, n_test)
        p-value (adjusted)

    gamma : float
        Percentile value used for aggregation.

    Returns
    -------
    1D ndarray (n_tests, )
        Vector of aggregated p-value
    """
    converted_score = (1 / gamma) * (
        np.percentile(pvals, q=100*gamma, axis=0))

    return np.minimum(1, converted_score)


def _adaptive_quantile_aggregation(pvals, gamma_min=0.05):
    """adaptive version of the quantile aggregation method, Meinshausen et al
    (2008)"""
    gammas = np.arange(gamma_min, 1.05, 0.05)
    list_Q = np.array([
        _fixed_quantile_aggregation(pvals, gamma) for gamma in gammas])

    return np.minimum(1, (1 - np.log(gamma_min)) * list_Q.min(0))


def _fdp_hat(t, score_vector, offset=1):
    false_dis = np.sum(score_vector <= -t)
    rejected = np.sum(score_vector >= t)
    res = (false_dis + offset) / np.maximum(1, rejected)

    return np.minimum(res, 1)


# def _inflate_non_null(non_zero_index, delta, n_features):
#     """Inflate non-null index to take into account of spatial tolerance delta
#     for FDR and power
#     """
#     if n_features is None and delta > 0:
#         raise ValueError(
#             'When delta is positive n_features should be inputted')
#     if n_features is None or delta == 0:
#         return non_zero_index

#     for s in non_zero_index:
#         for i in range(delta):
#             if s - i >= 0:
#                 non_zero_index = np.append(non_zero_index, s - i)
#             if s + i <= n_features - 1:
#                 non_zero_index = np.append(non_zero_index, s + i)

#     return np.unique(non_zero_index)


def _cluster_data(X, n_clusters, train_size=None, linkage='ward',
                  connectivity=None, memory=None, rand=None):

    cluster_object = FeatureAgglomeration(n_clusters, linkage=linkage,
                                          memory=memory,
                                          connectivity=connectivity)
    n_samples, _ = X.shape
    if train_size is None:
        X_reduced = cluster_object.fit_transform(X)
    else:
        train_index = resample(np.arange(n_samples),
                               n_samples=int(n_samples * train_size),
                               replace=False, random_state=rand)
        cluster_object.fit(X[train_index, :])
        X_reduced = cluster_object.transform(X)

    # Standardize
    X_reduced = StandardScaler().fit_transform(X_reduced)

    return X_reduced, cluster_object.labels_


def _empirical_pval(test_score, offset=1):

    pvals = []
    n_features = test_score.size

    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    test_score_inv = -test_score
    for i in range(n_features):
        if test_score[i] <= 0:
            pvals.append(1)
        else:
            pvals.append((offset +
                          np.sum(test_score_inv >= test_score[i])) / n_features)

    return np.array(pvals)


def _vote_sign(sign_matrix, selected):

    n_features = sign_matrix.shape[1]
    feature_signs = np.zeros(n_features)

    for i in range(n_features):
        if i in selected:
            extract = sign_matrix[:, i]
            feature_signs[i] = (_find_mode(extract[extract != 0]))
        else:
            continue

    return feature_signs


def _find_mode(array):
    vals, counts = np.unique(array, return_counts=True)
    return vals[counts.argmax()]


def _lambda_max(X, y, use_noise_estimate=True):
    """Calculation of lambda_max, the smallest value of regularization parameter in
    lasso program for non-zero coefficient
    """
    n_samples, _ = X.shape

    if not use_noise_estimate:
        return np.max(np.dot(X.T, y)) / n_samples

    norm_y = np.linalg.norm(y, ord=2)
    sigma_0 = (norm_y / np.sqrt(n_samples)) * 1e-3
    sig_star = max(sigma_0, norm_y / np.sqrt(n_samples))

    return np.max(np.abs(np.dot(X.T, y)) / (n_samples * sig_star))


def _logit(x):
    return np.exp(x) / (1 + np.exp(x))
