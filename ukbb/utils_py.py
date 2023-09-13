import sys

# sys.path.insert(1, '../../tuan_binh_nguyen/dev')
sys.path.insert(1, "..")
import time

import numpy as np
import pandas as pd
# import sandbox
# from permfit_python.permfit_py import permfit
from permfit_python.BBI import BlockBasedImportance
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid, train_test_split


def compute_d0crt(
    x,
    y,
    loss="least_square",
    statistic="residual",
    ntree=100,
    prob_type="regression",
    verbose=False,
    scaled_statistics=False,
    refit=False,
    n_jobs=20,
):
    d0crt_results = sandbox.dcrt_zero(
        x,
        y,
        loss="least_square",
        screening=False,
        statistic="randomforest",
        ntree=100,
        type_prob="regression",
        refit=False,
        verbose=True,
        n_jobs=n_jobs,
    )

    return pd.DataFrame(
        {
            "method": "d0crt",
            "variable": x.columns,
            "importance": d0crt_results[2],
            "p_value": d0crt_results[1],
            "score": d0crt_results[3],
        }
    )


def compute_RF(X, y, n_tree=100, k_fold=0, n_jobs=100, random_state=2022):
    var_labl = X.columns

    if k_fold >= 2:
        kf = KFold(n_splits=k_fold, random_state=random_state, shuffle=True)

    list_imp = []
    list_score = []
    for index_i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold: {index_i+1}")
        rf = RandomForestRegressor(n_estimators=n_tree, n_jobs=n_jobs)
        rf.fit(X.iloc[train_index, :], y.loc[train_index])
        list_imp.append(rf.feature_importances_)
        list_score.append(
            r2_score(y.loc[test_index], rf.predict(X.iloc[test_index, :]))
        )
    list_imp = np.mean(np.array(list_imp), axis=0)

    return pd.DataFrame(
        {
            "method": "RF",
            "variable": var_labl,
            "importance": list_imp,
            "p_value": np.nan,
            "score": sum(list_score) / len(list_score),
        }
    )


def compute_cpi_permfitDnn(
    x,
    y,
    conditional=False,
    n_jobs=10,
    k_fold=0,
    random_state=2022,
    list_nominal=None,
    groups=None,
    group_stacking=False,
    n_perm=100,
):
    rf = RandomForestRegressor(random_state=2023)
    dict_hyper = {"max_depth": [2, 5, 10, 20]}
    bbi_model = BlockBasedImportance(
        importance_estimator="Mod_RF",
        prob_type="regression",
        conditional=conditional,
        k_fold=k_fold,
        n_jobs=n_jobs,
        list_nominal=list_nominal,
        group_stacking=group_stacking,
        groups=groups,
        random_state=random_state,
        n_perm=n_perm,
        verbose=10,
    )
    bbi_model.fit(x, y)
    results = bbi_model.compute_importance()

    if len(groups) > 0:
        var_labl = list(groups)
    else:
        var_labl = x.columns

    # Specify the used method
    if conditional:
        method = "CPI-DNN"
    else:
        method = "Permfit-DNN"

    return pd.DataFrame(
        {
            "method": [method] * len(var_labl),
            "variable": var_labl,
            "importance": results["importance"][:, 0],
            "p_value": results["pval"][:, 0],
            "score": results["score"],
        }
    )


def compute_marginal(X, y, groups={}, random_state=2022):
    marginal_imp = []
    marginal_pval = []
    score = 0
    if len(groups) == 0:
        for el_ind, el in enumerate(X.columns):
            groups[el_ind] = [el]

    rng = np.random.RandomState(random_state)
    train_indices = rng.choice(X.shape[0], size=int(X.shape[0] * 0.8), replace=False)
    test_indices = np.array([i for i in range(X.shape[0]) if i not in train_indices])

    for i in groups.keys():
        reg = LinearRegression().fit(X.loc[train_indices, groups[i]], y[train_indices])
        result = f_regression(X.loc[train_indices, groups[i]], y[train_indices])
        marginal_imp.append(result[0][0])
        marginal_pval.append(result[1][0])
        score += reg.score(X.loc[test_indices, groups[i]], y[test_indices])
    return pd.DataFrame(
        {
            "method": "Marginal",
            "variable": groups.keys(),
            "importance": marginal_imp,
            "p_value": marginal_pval,
            "score": score / len(X.columns),
        }
    )


def loop_params(
    X,
    y,
    nominal_columns=[],
    grps={},
    param_grid={},
    n_jobs=1,
    k_fold=0,
    title_res="new_file.csv",
):
    count_res = 0
    for params in list(ParameterGrid(param_grid)):
        print(f"Params number:{count_res+1}")
        group_stacking = params["group_stacking"]
        if not params["group_based"]:
            group_stacking = False
            grps = {}

        start_time = time.monotonic()
        ## Marginal
        if params["method"] == "marginal":
            print("Applying Marginal")
            res = compute_marginal(X, y, groups=grps, random_state=2022)
            res["time"] = (time.monotonic() - start_time) / 60
            res = res.sort_values(by=["p_value"], ascending=True)

        ## PyPermfit-DNN
        if params["method"] == "permfit-dnn":
            print("Applying Permfit-DNN")
            res = compute_cpi_permfitDnn(
                X,
                y,
                conditional=False,
                k_fold=k_fold,
                random_state=2022,
                n_jobs=n_jobs,
                list_nominal=nominal_columns,
                group_stacking=group_stacking,
                groups=grps,
                n_perm=50,
            )
            res["time"] = (time.monotonic() - start_time) / 60
            res = res.sort_values(by=["p_value"], ascending=True)

        ## CPI-DNN
        if params["method"] == "cpi-dnn":
            print("Applying CPI-DNN")
            res = compute_cpi_permfitDnn(
                X,
                y,
                conditional=True,
                k_fold=k_fold,
                random_state=2022,
                n_jobs=n_jobs,
                list_nominal=nominal_columns,
                group_stacking=group_stacking,
                groups=grps,
                n_perm=50,
            )
            res["time"] = (time.monotonic() - start_time) / 60
            res = res.sort_values(by=["p_value"], ascending=True)

        ## d0crt method
        if params["method"] == "d0crt":
            print("Applying d0CRT")
            res = compute_d0crt(X, y, n_jobs=20)
            res["importance"] = np.abs(res["importance"])
            res["time"] = (time.monotonic() - start_time) / 60
            res = res.sort_values(by=["p_value"], ascending=True)

        ## RF method
        if params["method"] == "RF":
            print("Applying Random Forest")
            res = compute_RF(
                X, y, n_jobs=n_jobs, k_fold=k_fold, n_tree=2000, random_state=2022
            )

            res["importance"] = np.abs(res["importance"])
            res["time"] = (time.monotonic() - start_time) / 60
            res = res.sort_values(by=["importance"], ascending=False)

        print("mins: ", (time.monotonic() - start_time) / 60)

        res["group_based"] = params["group_based"]
        res["group_stacking"] = params["group_stacking"]
        if count_res == 0:
            final_res = res.copy()
            count_res += 1
        else:
            final_res = pd.concat([final_res, res])
            count_res += 1
    # print(final_res)
    final_res.to_csv(title_res, index=False)
