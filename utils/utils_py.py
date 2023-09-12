# from bartpy.sklearnmodel import SklearnModel
# from bartpy.features.featureselection import SelectSplitProportionThreshold
import numpy as np
# import sage
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import pytorch_lightning as pl
import sys
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(1, "../vimpy")
# import vimpy
from scipy.stats import ttest_1samp
from networks import *
from utils_lazy import *
from scipy import stats as st


def compute_bart_py(X, y):
    res = np.zeros((5, X.shape[1]))
    for i in range(5):
        model = SklearnModel(
            n_samples=1000,
            n_burn=250,
            n_trees=20,
            store_in_sample_predictions=False,
        )
        model_prop = SelectSplitProportionThreshold(model, 0)
        model_prop.fit(X, y)
        print(model_prop.feature_proportions)
        res[i] = model_prop.feature_proportions
    return np.mean(res, axis=0)


def compute_sage(X, y, ntree=100, seed=2021, prob_type="regression"):
    y = np.array(y)

    if prob_type == "classification":
        clf_rf = RandomForestClassifier(n_estimators=ntree)

    if prob_type == "regression":
        clf_rf = RandomForestRegressor(n_estimators=ntree)
    rng = np.random.RandomState(seed)
    train_ind = rng.choice(
        X.shape[0], int(X.shape[0] * 0.8), replace=False
    )
    test_ind = np.array(
        [i for i in range(X.shape[0]) if i not in train_ind]
    )

    clf_rf.fit(X.iloc[train_ind, :], y[train_ind])
    imputer = sage.MarginalImputer(clf_rf, X.iloc[test_ind, :])

    if prob_type == "classification":
        estimator = sage.PermutationEstimator(imputer, "cross entropy")

    if prob_type == "regression":
        estimator = sage.PermutationEstimator(imputer, "mse")

    sage_values = estimator(X.iloc[test_ind, :].values, y[test_ind])
    return sage_values.values


def compute_lazy(X, y):
    dict_vals = {"imp_vals": [], "lb_list": [], "ub_list": []}
    level = 0.05
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33
    )
    trainset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).view(-1, 1),
    )
    train_loader = DataLoader(trainset, batch_size=32)
    full_nn = NN4vi(X_train.shape[1], [100], 1)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # early_stopping = EarlyStopping('val_loss', min_delta=1e-2)
    trainer = pl.Trainer(max_epochs=100, enable_checkpointing=False)
    trainer.fit(full_nn, train_loader)
    full_pred_test = full_nn(X_test)
    for indx in range(X_train.shape[1]):
        X_test_change = dropout(X_test, indx)
        X_train_change = dropout(X_train, indx)
        lazy_pred_train, lazy_pred_test, cv_res = lazy_train_cv(
            full_nn, X_train_change, X_test_change, y_train, [100]
        )
        est = (
            nn.MSELoss()(lazy_pred_test, y_test).item()
            - nn.MSELoss()(full_pred_test, y_test).item()
        )

        # Confidence intervals
        eps_j = (
            ((y_test - lazy_pred_test) ** 2)
            .detach()
            .numpy()
            .reshape(1, -1)
        )
        eps_full = (
            ((y_test - full_pred_test) ** 2)
            .detach()
            .numpy()
            .reshape(1, -1)
        )
        se = np.sqrt(np.var(eps_j - eps_full) / y_test.shape[0])
        z = st.norm.ppf((1 - level / 2))
        lb = est - z * se
        ub = est + z * se

        dict_vals["imp_vals"].append(est)
        dict_vals["lb_list"].append(lb)
        dict_vals["ub_list"].append(ub)

    return dict_vals


def compute_vimp_dnn(X, y, prob_type="regression", seed=2022):
    dict_vals = {"val_imp": [], "p_value": []}

    if prob_type == "regression":
        measure_type = "r_squared"
    else:
        measure_type = "auc"
    for indx in range(X.shape[1]):
        print(f"Processing: {indx+1}")
        # dnn_model = CPI_DNN(prob_type=prob_type, k_fold=0, random_state=seed)
        ntrees = np.arange(100, 500, 1000)
        max_depth = np.array([2, 5, 10, 20])
        param_grid = [{"n_estimators": ntrees, "max_depth": max_depth}]
        cv_full = GridSearchCV(
            RandomForestRegressor(random_state=seed),
            param_grid=param_grid,
            cv=2,
        )
        vimp = vimpy.vim(
            y=y, x=X, s=indx, pred_func=cv_full, measure_type=measure_type
        )
        # vimp = vimpy.cv_vim(y=y, x=X, s=indx, V=2, pred_func=rf, measure_type=measure_type)
        ## get the point estimate of variable importance
        vimp.get_point_est()
        ## get the influence function estimate
        vimp.get_influence_function()
        ## get a standard error
        vimp.get_se()
        ## do a hypothesis test, compute p-value
        vimp.hypothesis_test(alpha=0.05, delta=0.002)
        dict_vals["val_imp"].append(vimp.vimp_)
        dict_vals["p_value"].append(vimp.p_value_)
        # print(vimp.p_value_)
        # print(vimp.hyp_test_)

    return dict_vals


def compute_loco(X, y, ntree=100, seed=2021, prob_type="regression"):
    y = np.array(y)
    dict_vals = {"val_imp": [], "p_value": []}
    list_delta = []
    if prob_type == "classification":
        clf_rf = RandomForestClassifier(
            n_estimators=ntree, random_state=seed
        )
    if prob_type == "regression":
        clf_rf = RandomForestRegressor(
            n_estimators=ntree, random_state=seed
        )

    rng = np.random.RandomState(seed)
    train_ind = rng.choice(
        X.shape[0], int(X.shape[0] * 0.8), replace=False
    )
    test_ind = np.array(
        [i for i in range(X.shape[0]) if i not in train_ind]
    )

    # Full Model
    clf_rf.fit(X.iloc[train_ind, :], y[train_ind])

    if prob_type == "regression":
        loss = np.square(y[test_ind] - clf_rf.predict(X.iloc[test_ind, :]))
    else:
        y_test = (
            OneHotEncoder(handle_unknown="ignore")
            .fit_transform(y[test_ind].reshape(-1, 1))
            .toarray()
        )

        loss = -np.sum(
            y_test * np.log(clf_rf.predict_proba(X.iloc[test_ind, :])),
            axis=1
        )
 
    # Retrain model

    for col in range(X.shape[1]):
        X_minus_idx = np.delete(np.copy(X), col, -1)
        clf_rf.fit(X_minus_idx[train_ind, :], y[train_ind])

        if prob_type == "regression":
            loss0 = np.square(
                y[test_ind] - clf_rf.predict(X_minus_idx[test_ind, :])
            )
        else:
            loss0 = np.sum(
                y_test
                * np.log(clf_rf.predict_proba(X_minus_idx[test_ind, :])),
                axis=1
            )
        delta = loss0 - loss

        t_statistic, p_value = ttest_1samp(delta, 0, alternative="greater")
        if np.isnan(t_statistic):
            t_statistic = 0
        if np.isnan(p_value):
            p_value = 1
        dict_vals["val_imp"].append(t_statistic)
        dict_vals["p_value"].append(p_value)

    return dict_vals

