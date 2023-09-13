# -*- coding: utf-8 -*-
# Authors: Binh Nguyen <tuan-binh.nguyen@inria.fr>

import numpy as np
from sklearn.linear_model import (ElasticNetCV, LassoCV, LogisticRegressionCV,
                                  RidgeCV)
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler


def stat_coef_diff(
    X,
    X_tilde,
    y,
    method="lasso_cv",
    cv=5,
    n_jobs=1,
    use_signed_max=False,
    n_lambdas=10,
    n_iter=1000,
    group_reg=1e-3,
    l1_reg=1e-3,
    joblib_verbose=0,
    return_coef=False,
    solver="liblinear",
    seed=None,
):
    """Calculate test statistic by doing estimation with Cross-validation on
    concatenated design matrix [X X_tilde] to find coefficients [beta
    beta_tilda]. The test statistic is then:

                        W_j =  abs(beta_j) - abs(beta_tilda_j)

    with j = 1, ..., n_features

    Parameters
    ----------
    X : 2D ndarray (n_samples, n_features)
        Original design matrix

    X_tilde : 2D ndarray (n_samples, n_features)
        Knockoff design matrix

    y : 1D ndarray (n_samples, )
        Response vector

    loss : str, optional
        if the response vector is continuous, the loss used should be
        'least_square', otherwise
        if the response vector is binary, it should be 'logistic'

    cv : int, optional
        number of cross-validation folds

    solver : str, optional
        solver used by sklearn function LogisticRegressionCV

    n_regu : int, optional
        number of regulation used in the regression problem

    return_coef : bool, optional
        return regression coefficient if set to True

    Returns
    -------
    test_score : 1D ndarray (n_features, )
        vector of test statistic

    coef: 1D ndarray (n_features * 2, )
        coefficients of the estimation problem
    """
    # the following import should be put inside to preven circular
    # dependencies, see
    # <https://stackoverflow.com/questions/9252543/importerror-cannot-import-name-x>
    from .experimentals import GroupLassoRefit, LassoRefit

    n_features = X.shape[1]
    X_ko = StandardScaler().fit_transform(np.column_stack([X, X_tilde]))
    X_train, X_test, y_train, y_test = train_test_split(
        X_ko, y, test_size=0.2, random_state=seed
    )
    lambda_max = np.max(np.dot(X_train.T, y_train)) / (2 * n_features)
    lambdas = np.linspace(lambda_max * np.exp(-n_lambdas), lambda_max, n_lambdas)

    estimator = {
        "lasso_cv": LassoCV(
            n_jobs=n_jobs, verbose=joblib_verbose, max_iter=int(1e4), cv=cv
        ),
        "ridge": RidgeCV(cv=cv),
        "logistic_l1": LogisticRegressionCV(
            penalty="l1",
            max_iter=2e2,
            tol=1e-6,
            solver=solver,
            cv=cv,
            random_state=0,
            n_jobs=n_jobs,
            scoring="roc_auc",
        ),
        "logistic_l2": LogisticRegressionCV(
            penalty="l2",
            max_iter=1e4,
            n_jobs=n_jobs,
            verbose=joblib_verbose,
            cv=cv,
            tol=1e-8,
        ),
        "enet": ElasticNetCV(
            cv=cv, max_iter=1e3, tol=1e-6, n_jobs=n_jobs, verbose=joblib_verbose
        ),
        "grlasso_refit": GridSearchCV(
            estimator=GroupLassoRefit(
                n_iter=n_iter, group_reg=group_reg, l1_reg=l1_reg
            ),
            cv=cv,
            n_jobs=n_jobs,
            verbose=joblib_verbose,
            scoring="neg_mean_squared_error",
            param_grid=dict(l1_reg=lambdas),
        ),
        "lasso_refit": GridSearchCV(
            estimator=LassoRefit(),
            cv=cv,
            n_jobs=n_jobs,
            verbose=joblib_verbose,
            scoring="neg_mean_squared_error",
            param_grid=dict(
                alpha=_alpha_grid(X_train, y_train, l1_ratio=1.0, fit_intercept=False)
            ),
        ),
    }

    try:
        clf = estimator[method]
    except KeyError:
        print(f"{method} is not a valid estimator")

    clf.fit(X_train, y_train)

    try:
        coef = np.ravel(clf.coef_)
    except AttributeError:
        coef = np.ravel(clf.best_estimator_.coef_)  # for GridSearchCV object

    if use_signed_max:
        test_score = np.sign(
            np.abs(coef[:n_features]) - np.abs(coef[n_features:])
        ) * np.maximum(np.abs(coef[:n_features]), np.abs(coef[n_features:]))
    else:
        test_score = np.abs(coef[:n_features]) - np.abs(coef[n_features:])

    if return_coef:
        return test_score, coef

    return test_score, clf.score(X_test, y_test)


def _coef_diff_threshold(test_score, fdr=0.1, offset=1):
    """Calculate the knockoff threshold based on the procedure stated in the
    article.

    Parameters
    ----------
    test_score : 1D ndarray, shape (n_features, )
        vector of test statistic

    fdr : float, optional
        desired controlled FDR level

    offset : int, 0 or 1, optional
        offset equals 1 is the knockoff+ procedure

    Returns
    -------
    thres : float or np.inf
        threshold level
    """
    if offset not in (0, 1):
        raise ValueError("'offset' must be either 0 or 1")

    t_mesh = np.sort(np.abs(test_score[test_score != 0]))
    for t in t_mesh:
        false_pos = np.sum(test_score <= -t)
        selected = np.sum(test_score >= t)
        if (offset + false_pos) / np.maximum(selected, 1) <= fdr:
            return t

    return np.inf
