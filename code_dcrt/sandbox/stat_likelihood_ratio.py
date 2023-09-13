import numpy as np
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler


def stat_likelihood_ratio(
    X,
    X_tilde,
    y,
    method="lasso_cv",
    n_jobs=1,
    cv=5,
    solver="liblinear",
    joblib_verbose=0,
):
    """Test score for model-X knockoff following Katsevich & Ramdas (2020)
    Prop.8"""

    _, n_features = X.shape
    X_ko = StandardScaler().fit_transform(np.column_stack([X, X_tilde]))

    estimator = {
        "lasso_cv": LassoCV(n_jobs=n_jobs, verbose=joblib_verbose, max_iter=1e4, cv=cv),
        "logistic_l1": LogisticRegressionCV(
            penalty="l1", max_iter=5e2, tol=1e-4, solver=solver, cv=cv, n_jobs=n_jobs
        ),
    }

    try:
        clf = estimator[method]
    except KeyError:
        print(f"{method} is not a valid estimator")

    clf.fit(X_ko, y)
    coef_X_true = np.ravel(clf.coef_)[:n_features]
    coef_X_ko = np.ravel(clf.coef_)[n_features:]

    # numerator of the ll
    ll_X_true = _cal_likelihood(X, coef_X_true, method=method)

    ll_ratio_score = []
    # denumerator and ll ratio
    for idx in range(n_features):
        temp = np.copy(X)
        temp[:, idx] = X_ko[:, idx]  # replace only j with its knockoff
        temp_coef = np.copy(coef_X_true)
        temp_coef[idx] = coef_X_ko[idx]
        ll_X_ko = _cal_likelihood(temp, temp_coef, method=method)
        ll_ratio_score.append(np.prod(ll_X_true / ll_X_ko))

    return np.array(ll_ratio_score)


def _cal_likelihood(x, beta, method="lasso_cv"):
    if method == "lasso_cv":
        return np.dot(x, beta)
    elif method == "logistic_l1":
        return 1 / (1 + np.exp(-np.dot(x, beta)))
    else:
        raise ValueError(f"{method} is not supported.")
