# -*- coding: utf-8 -*-
# Author: Binh Nguyen <tuan-binh.nguyen@inria.fr> & Jerome-Alexis Chevalier
"""
Adaptation of Debiased Lasso on controlling FDR (Javanmard & Javardi 2018)
"""
import numpy as np
from scipy import stats

from .desparsified_lasso import desparsified_lasso
from .utils import _reid, pval_from_zscore


def dl_fdr(X, y, fdr=0.1, one_sided=False, tol=1e-4, method="lasso", c=0.01,
           verbose=False):
    """FDR controlling with Debiased Lasso following Javanmard

    Parameters
    -----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data
        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary
        fdr : float, optional
            Target level for FDR control
        one_sided : bool, optional
            Whethere the inference have two-sided pvalue or not
        tol : float, optional
            The tolerance for the optimization: if the updates are
            smaller than ``tol``, the optimization code checks the
            dual gap for optimality and continues until it is smaller
            than ``tol``.
        method : string, optional
            The method for the nodewise lasso: "lasso", "lasso_cv" or
            "zhang_zhang"
        c : float, optional
            Only used if method="lasso". Then alpha = c * alpha_max.
        standardized : bool, optional
            Whether to center the data or not
        """

    _, zscore, pval = desparsified_lasso(X, y, tol=tol, method=method,
                                         c=c)
    threshold = _dl_fdr_threshold(zscore, fdr=fdr, one_sided=one_sided)
    selected = np.where(np.abs(zscore) >= threshold)[0]

    if verbose:
        return selected, threshold, pval, zscore

    return selected


def _dl_fdr_threshold(fcd_statistic, fdr=0.1, one_sided=False):
    """

    """
    n_features = len(fcd_statistic)
    t_p = np.sqrt(2 * np.log(n_features) - 2 * np.log(np.log(n_features)))
    abs_fcd_statistic = np.sort(np.abs(fcd_statistic))
    mesh = abs_fcd_statistic[abs_fcd_statistic <= t_p]
    # default threshold
    threshold = np.sqrt(2 * np.log(n_features))
    # adaptively search for threshold
    for i in range(len(mesh)):
        if one_sided:
            multiplicator = 1
        else:
            multiplicator = 2

        temp_result = \
            multiplicator * n_features * (stats.norm.sf(mesh[i]) /
                                          max(len(mesh) - i - 1, 1))
        if temp_result <= fdr:
            threshold = mesh[i]
            break
        i += 1
    # otherwise threshold
    return threshold
