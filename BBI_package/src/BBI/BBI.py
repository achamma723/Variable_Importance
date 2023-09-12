import itertools
import warnings
from copy import copy

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (log_loss, mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from .compute_importance import (joblib_compute_conditional,
                                 joblib_compute_permutation)
from .Dnn_learner import DNN_learner
from .utils import convert_predict_proba, create_X_y


class BlockBasedImportance(BaseEstimator, TransformerMixin):
    """This class implements the Block Based Importance (BBI),
       it consists of the learner block (first block)
       and the importance block (second block).
    Parameters
    ----------
    estimator: scikit-learn compatible estimator, default=None
        The provided estimator for the prediction task (First block).
        The default estimator is the DNN learner. Other options are (1) RF
        for Random Forest.
    importance_estimator: {scikit-learn compatible estimator or string},
                          default=None
        The provided estimator for the importance task (Second block).
        Using "Mod_RF" will apply the modified version of the Random Forest as
        the importance predictor.
    do_hyper: bool, default=True
        Tuning the hyperparameters of the provided estimator.
    dict_hyper: dict, default=None
        The dictionary of hyperparameters to tune.
    prob_type: str, default='regression'
        A classification or a regression problem.
    bootstrap: bool, default=True
        Application of bootstrap sampling for the training set.
    split_perc: float, default=0.8
        The training/validation cut for the provided data.
    conditional: bool, default=True
        The permutation or the conditional sampling approach.
    list_nominal: dict, default=None
        The dictionary of binary, nominal and ordinal variables.
    Perm: bool, default=False
        The use of permutations or random sampling with CPI-DNN.
    n_perm: int, default=50
        The number of permutations/random sampling for each column.
    n_jobs: int, default=1
        The number of workers for parallel processing.
    verbose: int, default=0
        If verbose > 0, the fitted iterations will be printed.
    groups: dict, default=None
        The knowledge-driven/data-driven grouping of the variables if provided.
    group_stacking: bool, default=False
        Apply the stacking-based method for the provided groups.
    prop_out_subLayers: int, default=0.
        If group_stacking is set to True, proportion of outputs for
        the linear sub-layers per group.
    index_i: int, default=None
        The index of the current processed iteration.
    random_state: int, default=2023
        Fixing the seeds of the random generator.
    com_imp: boolean, default=True
        Compute or not the importance scores.
    Attributes
    ----------
    ToDO
    """

    def __init__(
        self,
        estimator=None,
        importance_estimator=None,
        do_hyper=True,
        dict_hyper=None,
        prob_type="regression",
        bootstrap=True,
        split_perc=0.8,
        conditional=True,
        list_nominal=None,
        Perm=False,
        n_perm=50,
        n_jobs=1,
        verbose=0,
        groups=None,
        group_stacking=False,
        k_fold=2,
        prop_out_subLayers=0,
        index_i=None,
        random_state=2023,
        com_imp=True,
    ):
        self.estimator = estimator
        self.importance_estimator = importance_estimator
        self.do_hyper = do_hyper
        self.dict_hyper = dict_hyper
        self.prob_type = prob_type
        self.bootstrap = bootstrap
        self.split_perc = split_perc
        self.conditional = conditional
        self.list_nominal = list_nominal
        self.Perm = Perm
        self.n_perm = n_perm
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.groups = groups
        self.group_stacking = group_stacking
        self.k_fold = k_fold
        self.prop_out_subLayers = prop_out_subLayers
        self.index_i = index_i
        self.random_state = random_state
        self.X_test = [None] * max(self.k_fold, 1)
        self.y_test = [None] * max(self.k_fold, 1)
        self.org_pred = [None] * max(self.k_fold, 1)
        self.pred_scores = [None] * max(self.k_fold, 1)
        self.X_nominal = [None] * max(self.k_fold, 1)
        self.type = None
        self.list_estimators = [None] * max(self.k_fold, 1)
        self.X_proc = [None] * max(self.k_fold, 1)
        self.scaler_x = [None] * max(self.k_fold, 1)
        self.scaler_y = [None] * max(self.k_fold, 1)
        self.com_imp = com_imp

    def fit(self, X, y=None):
        """Build the provided estimator with the training set (X, y)
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        # Fixing the random generator's seed
        self.rng = np.random.RandomState(self.random_state)

        # Convert list_nominal to a dictionary if initialized
        # as an empty string
        if not isinstance(self.list_nominal, dict):
            self.list_nominal = {
                "nominal": [],
                "ordinal": [],
                "binary": [],
            }

        if "binary" not in self.list_nominal:
            self.list_nominal["binary"] = []
        if "ordinal" not in self.list_nominal:
            self.list_nominal["ordinal"] = []
        if "nominal" not in self.list_nominal:
            self.list_nominal["nominal"] = []

        # Move the ordinal columns with 2 values to the binary part
        if self.list_nominal["ordinal"] != []:
            for ord_col in self.list_nominal["ordinal"]:
                if len(np.unique(X[ord_col])) < 3:
                    self.list_nominal["binary"].append(ord_col)
                    self.list_nominal["ordinal"].remove(ord_col)

        # Convert X to pandas dataframe
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if (self.groups is None) or (not bool(self.groups)):
            # Initialize the list_cols variable with each feature
            # in a seperate list (default case)
            self.groups = [[col] for col in X.columns]
            if self.group_stacking:
                # Remove the group_stacking flag not
                # to complex the DNN architecture
                self.group_stacking = False
                warnings.warn(
                    "The groups are not provided to apply the stacking"
                    " approach, back to single variables case."
                )

        # Convert dictionary of groups to a list of lists
        if isinstance(self.groups, dict):
            self.groups = list(self.groups.values())

        self.list_cols = self.groups.copy()
        self.list_cat_tot = list(
            itertools.chain.from_iterable(self.list_nominal.values())
        )
        X_nominal_org = X.loc[:, self.list_cat_tot]

        # One-hot encoding of nominal variables
        tmp_list = []
        self.dict_nom = {}
        # A dictionary to save the encoders of the nominal variables
        self.dict_enc = {}
        if len(self.list_nominal["nominal"]) > 0:
            for col_encode in self.list_nominal["nominal"]:
                enc = OneHotEncoder(handle_unknown="ignore")
                enc.fit(X[[col_encode]])
                labeled_cols = [
                    enc.feature_names_in_[0] + "_" + str(enc.categories_[0][j])
                    for j in range(len(enc.categories_[0]))
                ]
                hot_cols = pd.DataFrame(
                    enc.transform(X[[col_encode]]).toarray(),
                    dtype="int32",
                    columns=labeled_cols,
                )
                X = X.drop(columns=[col_encode])
                X = pd.concat([X, hot_cols], axis=1)
                self.dict_enc[col_encode] = enc

        # Create a dictionary for categorical variables with their indices
        for col_cat in self.list_cat_tot:
            current_list = [
                col
                for col in range(len(X.columns))
                if X.columns[col].split("_")[0] == col_cat
            ]
            if len(current_list) > 0:
                self.dict_nom[col_cat] = current_list
                # A list to store the labels of the categorical variables
                tmp_list.extend(current_list)

        # Create a dictionary for the continuous variables that will be scaled
        self.dict_cont = {}
        for ind_col, col_cont in enumerate(X.columns):
            if ind_col not in tmp_list:
                self.dict_cont[col_cont] = [ind_col]
        self.list_cont = [el[0] for el in self.dict_cont.values()]
        X = X.to_numpy()

        if len(y.shape) != 2:
            y = np.array(y).reshape(-1, 1)

        if self.prob_type == "classification":
            self.loss = log_loss
        else:
            self.loss = mean_squared_error

        # Replace groups' variables by the indices in the design matrix
        self.list_grps = []
        self.inp_dim = None
        if self.group_stacking:
            for grp in self.groups:
                current_grp = []
                for i in grp:
                    if i in self.dict_nom.keys():
                        current_grp += self.dict_nom[i]
                    else:
                        current_grp += self.dict_cont[i]
                self.list_grps.append(current_grp)

            if self.estimator is not None:
                # Force the output to 1 neurone per group
                # in standard stacking case
                self.prop_out_subLayers = 0
                list_alphas = np.geomspace(1e-3, 1e5, num=100)
                X_prvs = X.copy()
                X = np.zeros((X.shape[0], len(self.list_grps)))
                for grp_ind, grp in enumerate(self.list_grps):
                    if len(grp) > 1:
                        clf_stack = RidgeCV(alphas=list_alphas, cv=10).fit(X_prvs, y)
                        X[:, grp_ind] = clf_stack.predict(X_prvs).ravel()
                    else:
                        X[:, grp_ind] = X_prvs[:, grp].ravel()
                self.list_cont = list(np.arange(len(self.list_grps)))

            self.inp_dim = [
                max(1, int(self.prop_out_subLayers * len(grp)))
                for grp in self.list_grps
            ]
            self.inp_dim.insert(0, 0)
            self.inp_dim = np.cumsum(self.inp_dim)
            self.list_cols = [
                list(np.arange(self.inp_dim[grp_ind], self.inp_dim[grp_ind + 1]))
                for grp_ind in range(len(self.list_grps))
            ]

        # Initialize the first estimator (block learner)
        if self.estimator is None:
            self.estimator = DNN_learner(
                prob_type=self.prob_type,
                encode=True,
                do_hyper=False,
                list_cont=self.list_cont,
                list_grps=self.list_grps,
                group_stacking=self.group_stacking,
                n_jobs=self.n_jobs,
                inp_dim=self.inp_dim,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            self.type = "DNN"
            # Initializing the dictionary for tuning the hyperparameters
            if self.dict_hyper is None:
                self.dict_hyper = {
                    "lr": [1e-4, 1e-3, 1e-2],
                    "l1_weight": [0, 1e-4, 1e-2],
                    "l2_weight": [0, 1e-4, 1e-2],
                }

        elif self.estimator == "RF":
            if self.prob_type == "regression":
                self.estimator = RandomForestRegressor(random_state=2023)
            else:
                self.estimator = RandomForestClassifier(random_state=2023)
            self.dict_hyper = {"max_depth": [2, 5, 10, 20]}

        if self.k_fold != 0:
            # Implementing k-fold cross validation as the default behavior
            kf = KFold(
                n_splits=self.k_fold,
                random_state=self.random_state,
                shuffle=True,
            )
            for ind_fold, (train_index, test_index) in enumerate(kf.split(X)):
                print(f"Processing: {ind_fold+1}")
                X_fold = X.copy()
                y_fold = y.copy()

                self.X_nominal[ind_fold] = X_nominal_org.iloc[test_index, :]

                X_train, X_test = (
                    X_fold[train_index, :],
                    X_fold[test_index, :],
                )
                y_train, y_test = y_fold[train_index], y_fold[test_index]

                self.X_test[ind_fold] = X_test.copy()
                self.y_test[ind_fold] = y_test.copy()

                # Find the list of optimal sub-models to be used in the
                # following steps (Default estimator)
                if self.do_hyper:
                    self.__tuning_hyper(X_train, y_train, ind_fold)
                if self.type == "DNN":
                    self.estimator.fit(X_train, y_train)
                self.list_estimators[ind_fold] = copy(self.estimator)

        else:
            # Hyperparameter tuning
            if self.do_hyper:
                self.__tuning_hyper(X, y, 0)

            if self.type == "DNN":
                self.estimator.fit(X, y)
            self.list_estimators[0] = copy(self.estimator)

        self.is_fitted = True
        return self

    def __tuning_hyper(self, X, y, ind_fold=None):
        """ """
        (
            X_train_scaled,
            y_train_scaled,
            X_valid_scaled,
            y_valid_scaled,
            X_scaled,
            __,
            scaler_x,
            scaler_y,
            ___,
        ) = create_X_y(
            X,
            y,
            bootstrap=self.bootstrap,
            split_perc=self.split_perc,
            prob_type=self.prob_type,
            list_cont=self.list_cont,
            random_state=self.random_state,
        )
        list_hyper = list(itertools.product(*list(self.dict_hyper.values())))
        list_loss = []

        if self.type == "DNN":
            list_loss = self.estimator.hyper_tuning(
                X_train_scaled,
                y_train_scaled,
                X_valid_scaled,
                y_valid_scaled,
                list_hyper,
                random_state=self.random_state,
            )
        else:
            for ind_el, el in enumerate(list_hyper):
                curr_params = dict(
                    (k, v) for v, k in zip(el, list(self.dict_hyper.keys()))
                )
                list_hyper[ind_el] = curr_params
                self.estimator.set_params(**curr_params)
                if self.prob_type == "regression":
                    y_train_curr = y_train_scaled * scaler_y.scale_ + scaler_y.mean_
                    y_valid_curr = y_valid_scaled * scaler_y.scale_ + scaler_y.mean_
                    func = lambda x: self.estimator.predict(x)
                else:
                    y_train_curr = y_train_scaled.copy()
                    y_valid_curr = y_valid_scaled.copy()
                    func = lambda x: self.estimator.predict_proba(x)
                self.estimator.fit(X_train_scaled, y_train_curr)

                list_loss.append(self.loss(y_valid_curr, func(X_valid_scaled)))

        ind_min = np.argmin(list_loss)
        best_hyper = list_hyper[ind_min]
        if not isinstance(best_hyper, dict):
            best_hyper = dict(zip(self.dict_hyper.keys(), best_hyper))

        self.estimator.set_params(**best_hyper)
        self.estimator.fit(X_scaled, y)

        # If not a DNN learner case, need to save the scalers
        self.scaler_x[ind_fold] = scaler_x
        self.scaler_y[ind_fold] = scaler_y

    def predict(self, X=None, encoding=True):
        """Predict regression target for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        if not isinstance(X, list):
            list_X = [X.copy() for el in range(max(self.k_fold, 1))]
            mean_pred = True
        else:
            list_X = X.copy()
            mean_pred = False

        for ind_fold, curr_X in enumerate(list_X):
            # Prepare the test set for the prediction
            if encoding:
                X_tmp = self.__encode_input(curr_X)
            else:
                X_tmp = curr_X.copy()

            if self.type != "DNN":
                if not isinstance(curr_X, np.ndarray):
                    X_tmp = np.array(X_tmp)
                X_tmp[:, self.list_cont] = self.scaler_x[ind_fold].transform(
                    X_tmp[:, self.list_cont]
                )
                self.X_proc[ind_fold] = [X_tmp.copy()]

            self.org_pred[ind_fold] = self.list_estimators[ind_fold].predict(X_tmp)

            # Convert to the (n_samples x n_outputs) format
            if len(self.org_pred[ind_fold].shape) != 2:
                self.org_pred[ind_fold] = self.org_pred[ind_fold].reshape(-1, 1)

            if self.type == "DNN":
                self.X_proc[ind_fold] = np.array(
                    self.list_estimators[ind_fold].X_test.copy()
                ).swapaxes(0, 1)

        if mean_pred:
            return np.mean(np.array(self.org_pred), axis=0)

    def predict_proba(self, X=None, encoding=True):
        """Predict class probabilities for X. ToDo
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        if not isinstance(X, list):
            list_X = [X.copy() for el in range(max(self.k_fold, 1))]
            mean_pred = True
        else:
            list_X = X.copy()
            mean_pred = False

        for ind_fold, curr_X in enumerate(list_X):
            # Prepare the test set for the prediction
            if encoding:
                X_tmp = self.__encode_input(curr_X)
            else:
                X_tmp = curr_X.copy()

            if self.type != "DNN":
                if not isinstance(curr_X, np.ndarray):
                    X_tmp = np.array(X_tmp)
                X_tmp[:, self.list_cont] = self.scaler_x[ind_fold].transform(
                    X_tmp[:, self.list_cont]
                )
                self.X_proc[ind_fold] = [X_tmp.copy()]

            self.org_pred[ind_fold] = self.list_estimators[ind_fold].predict_proba(
                X_tmp
            )

            if self.type == "DNN":
                self.X_proc[ind_fold] = np.array(
                    self.list_estimators[ind_fold].X_test.copy()
                ).swapaxes(0, 1)
            else:
                self.org_pred[ind_fold] = convert_predict_proba(self.org_pred[ind_fold])

        if mean_pred:
            return np.mean(np.array(self.org_pred), axis=0)

    def __encode_input(self, X):
        # Check is fit had been called
        check_is_fitted(self, ["is_fitted"])

        # One-hot encoding for the test set
        if len(self.list_nominal["nominal"]) > 0:
            for col_encode in self.list_nominal["nominal"]:
                enc = self.dict_enc[col_encode]
                labeled_cols = [
                    enc.feature_names_in_[0] + "_" + str(enc.categories_[0][j])
                    for j in range(len(enc.categories_[0]))
                ]
                hot_cols = pd.DataFrame(
                    enc.transform(X[[col_encode]]).toarray(),
                    dtype="int32",
                    columns=labeled_cols,
                )
                X = X.drop(columns=[col_encode])
                X = pd.concat([X, hot_cols], axis=1)

        return X

    def compute_importance(self, X=None, y=None):
        """This function is used to compute the importance scores and
        the statistical guarantees for the different variables/groups
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        y : {array-like}, shape (n_samples,)
            The output samples.
        Returns
        -------
        results : dict
            The dictionary of importance scores, p-values and the corresponding
            score.
        """
        # Check is fit had been called
        check_is_fitted(self, ["is_fitted"])
        encoding = True

        if self.k_fold != 0:
            X = self.X_test.copy()
            y = self.y_test.copy()
            encoding = False
        else:
            # Convert X to pandas dataframe
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
            self.X_nominal[0] = X.loc[:, self.list_cat_tot]
            X = [X.copy() for el in range(max(self.k_fold, 1))]
            if self.prob_type == "classification":
                pass
            else:
                if len(y.shape) != 2:
                    y = y.reshape(-1, 1)
            y = [y.copy() for el in range(max(self.k_fold, 1))]

        # Compute original predictions
        if self.prob_type == "regression":
            output_dim = y[0].shape[1]
            self.predict(X, encoding=encoding)
        else:
            output_dim = 1
            self.predict_proba(X, encoding=encoding)

        list_seeds_imp = self.rng.randint(1e5, size=self.n_perm)
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        score_imp_l = []
        score_cur_l = []
        # n_features x n_permutations x n_samples
        for ind_fold, estimator in enumerate(self.list_estimators):
            if self.com_imp:
                if not self.conditional:
                    self.pred_scores[ind_fold], score_cur = list(
                        zip(
                            *parallel(
                                delayed(joblib_compute_permutation)(
                                    self.list_cols[p_col],
                                    perm,
                                    estimator,
                                    self.type,
                                    self.X_proc[ind_fold],
                                    y[ind_fold],
                                    self.prob_type,
                                    self.org_pred[ind_fold],
                                    dict_cont=self.dict_cont,
                                    dict_nom=self.dict_nom,
                                    proc_col=p_col,
                                    index_i=ind_fold + 1,
                                    group_stacking=self.group_stacking,
                                    random_state=list_seeds_imp[perm],
                                )
                                for p_col in range(len(self.list_cols))
                                for perm in range(self.n_perm)
                            )
                        )
                    )
                    self.pred_scores[ind_fold] = np.array(
                        self.pred_scores[ind_fold]
                    ).reshape(
                        (
                            len(self.list_cols),
                            self.n_perm,
                            y[ind_fold].shape[0],
                            output_dim,
                        )
                    )
                else:
                    self.pred_scores[ind_fold], score_cur = list(
                        zip(
                            *parallel(
                                delayed(joblib_compute_conditional)(
                                    self.list_cols[p_col],
                                    self.n_perm,
                                    estimator,
                                    self.type,
                                    self.importance_estimator,
                                    self.X_proc[ind_fold],
                                    y[ind_fold],
                                    self.prob_type,
                                    self.org_pred[ind_fold],
                                    seed=self.random_state,
                                    dict_cont=self.dict_cont,
                                    dict_nom=self.dict_nom,
                                    X_nominal=self.X_nominal[ind_fold],
                                    list_nominal=self.list_nominal,
                                    encoder=self.dict_enc,
                                    proc_col=p_col,
                                    index_i=ind_fold + 1,
                                    group_stacking=self.group_stacking,
                                    list_seeds=list_seeds_imp,
                                    Perm=self.Perm,
                                    output_dim=output_dim,
                                )
                                for p_col in range(len(self.list_cols))
                            )
                        )
                    )
                score_imp_l.append(score_cur[0])
                # Compute the mean over the number of permutations/resampling
                self.pred_scores[ind_fold] = np.mean(self.pred_scores[ind_fold], axis=1)
            else:
                score_cur_l.append(
                    (
                        mean_absolute_error(y[ind_fold], self.org_pred[ind_fold]),
                        r2_score(y[ind_fold], self.org_pred[ind_fold]),
                    )
                )
        if len(score_cur_l) > 0:
            return np.mean(score_cur_l, axis=0)

        weights = np.array([el.shape[1] for el in self.pred_scores])
        # Compute the mean of each fold over the number of observations
        pred_mean = np.array([np.mean(el.copy(), axis=1) for el in self.pred_scores])
        results = {}
        # Weighted average
        results["importance"] = np.average(pred_mean, axis=0, weights=weights)
        # Compute the standard deviation of each fold
        # over the number of observations
        pred_std = np.array(
            [
                np.mean(
                    (el - results["importance"][:, np.newaxis]) ** 2,
                    axis=1,
                )
                for el in self.pred_scores
            ]
        )
        results["std"] = np.sqrt(
            np.average(pred_std, axis=0, weights=weights) / (np.sum(weights) - 1)
        )
        results["pval"] = norm.sf(results["importance"] / results["std"])
        results["pval"][np.isnan(results["pval"])] = 1
        if self.prob_type == "regression":
            results["score_MAE"] = np.mean(np.array(score_imp_l), axis=0)[0]
            results["score_R2"] = np.mean(np.array(score_imp_l), axis=0)[1]
        else:
            results["score_AUC"] = np.mean(np.array(score_imp_l), axis=0)

        return results
