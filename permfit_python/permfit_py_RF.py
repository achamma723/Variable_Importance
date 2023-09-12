import os
import itertools
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, roc_auc_score, mean_squared_error, log_loss
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from .deep_model_standard import create_X_y
from .compute_perm_conditional_standard import joblib_compute_perm_conditional

def permfit_RF(X_train, y_train, train_index=None, test_index=None, prob_type='regression', k_fold=0,
              split_perc=0.2, random_state=2023, n_ensemble=10, batch_size=1024, n_perm=50,
              res=True, verbose=0, conditional=False, n_jobs=1, backend = "loky",
              list_nominal = {}, max_depth=2, index_i=None, groups=[], group_stacking=False,
              n_out_subLayers=1, noImp=False):
    '''
    X_train: The matrix of predictors to train/validate
    y_train: The response vector to train/validate
    train_index: if given, the indices of the train matrix
    test_index: if given, the indices of the test matrix
    prob_type: A classification or regression probem
    k_fold: The number of folds for the cross validaton
    split_perc: if k_fold==0, The percentage of the split to train/validate portions
    random_state: Fixing the seeds of the random generator
    n_ensemble: The number of DNNs to be fit
    batch_size: The number of samples per batch for test prediction
    n_perm: The number of permutations for each column
    res: If True, it will return the dictionary of results
    verbose: If > 1, the progress bar will be printed
    conditional: The permutation or the conditional sampling approach
    list_nominal: The list of categorical variables if exists
    max_depth: The maximum depth of the RF model for the variable of interest prediction
    index_i: The index of the current processed iteration
    groups: The knowledge-driven grouping of the variables if provided
    group_stacking: Apply the stacking-based method for the groups
    n_out_subLayers: If group_stacking True, number of outputs for the linear sub-layers
    '''
    # Fixing the random generator's seed
    rng = np.random.RandomState(random_state)

    # Convert dictionary of groups to a list of lists
    if type(groups) is dict:
        groups = list(groups.values())

    results = {}
    n, p = X_train.shape
    if list_nominal == "":
        list_nominal = {'nominal': [], 'ordinal': [], 'binary': []}
    list_cols = list(X_train.columns)
    list_cat_tot = list(itertools.chain.from_iterable(list_nominal.values()))
    # No cross validation, splitting the samples into train/validate portions
    if k_fold == 0:
        X_nominal = X_train[list(set(list_cat_tot) & set(list_cols))]
        # One-hot encoding of Nominal variables
        tmp_list = []
        dict_nom = {}
        enc_dict = {}
        if len(list_nominal['nominal']) > 0:
            for col_encode in list_nominal['nominal']:#list(set(list_nominal['nominal']) & set(list_cols)):
                enc = OneHotEncoder(handle_unknown='ignore')
                enc.fit(X_train[[col_encode]])
                labeled_cols = [enc.feature_names_in_[0] + '_' + str(enc.categories_[0][j])
                    for j in range(len(enc.categories_[0]))]
                hot_cols = pd.DataFrame(enc.transform(X_train[[col_encode]]).toarray(),
                dtype='int32', columns=labeled_cols)
                X_train = X_train.drop(columns=[col_encode])
                X_train = pd.concat([X_train, hot_cols], axis=1)
                enc_dict[col_encode] = enc
            # Create a dictionary for the one-hot encoded variables with the indices of the corresponding categories
            for col_nom in list_cat_tot:
                current_list = [col for col in range(len(X_train.columns)) if X_train.columns[col].split('_')[0] == col_nom]
                if len(current_list) > 0:
                    dict_nom[col_nom] = current_list
                    tmp_list.extend(current_list)

        # Retrieve the list of continuous variables that will be scaled
        dict_cont = {}
        for col_cont in range(len(X_train.columns)):
            if col_cont not in tmp_list:
                dict_cont[X_train.columns[col_cont]] = [col_cont]
        X_train = X_train.to_numpy()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if prob_type == 'classification':
            # Converting target values to corresponding integer values
            ind_unique = np.unique(y_train)
            dict_target = dict(zip(ind_unique, range(len(ind_unique))))
            y_train = np.array([dict_target[el]
                                for el in list(y_train)]).reshape(-1, 1)
            score = roc_auc_score
            loss = log_loss
            rf = RandomForestClassifier(random_state=random_state)
        else:
            y_train = np.array(y_train).reshape(-1, 1)
            score = r2_score
            loss = mean_squared_error
            rf = RandomForestRegressor(random_state=random_state)
        if train_index is None:
            train_index = rng.choice(X_train.shape[0], size=int(X_train.shape[0]*(1-split_perc)), replace=False)
            test_index = np.array([ind for ind in range(n) if ind not in train_index])
        X_train, X_test = X_train[train_index, :], X_train[test_index, :]
        y_train, y_test = y_train[train_index], y_train[test_index]

        if isinstance(X_nominal, pd.DataFrame):
            X_nominal = X_nominal.iloc[test_index, :]

        # Replace group labels by the corresponding index in the design matrix
        list_grps = []
        if group_stacking:
            for grp in groups:
                current_grp = []
                for i in grp:
                    if i in dict_nom.keys():
                        current_grp += dict_nom[i]
                    else:
                        current_grp += [dict_cont[i]]
                list_grps.append(current_grp)

        parallel = Parallel(n_jobs=min(n_jobs, 10), verbose=verbose, backend=backend)

        list_cont = [el[0] for el in dict_cont.values()]
        scaler_x, scaler_y = StandardScaler(), StandardScaler()

        X_train_scaled, y_train_scaled, X_valid_scaled, y_valid_scaled, X_scaled, __, \
        scaler_x, scaler_y, ___ = create_X_y(X_train, y_train, bootstrap=True, split_perc=split_perc,
                                           prob_type=prob_type, list_cont=list_cont,
                                           random_state=random_state)

        list_cols = [[val] for val in list_cols]

        list_maxDepth = [2, 5, 10, 20]
        list_loss = []
        for max_depth in list_maxDepth:
            rf.set_params(max_depth=max_depth)
            if prob_type == 'regression':
                y_train_curr = y_train_scaled*scaler_y.scale_ + scaler_y.mean_
                y_valid_curr = y_valid_scaled*scaler_y.scale_ + scaler_y.mean_
            else:
                y_train_curr = y_train_scaled.copy()
                y_valid_curr = y_valid_scaled.copy()
            rf.fit(X_train_scaled, y_train_curr)
            if prob_type == "regression":
                pred_cur = rf.predict(X_valid_scaled)
            else:
                pred_cur = rf.predict_proba(X_valid_scaled)[:, 1]
            list_loss.append(loss(y_valid_curr, pred_cur))

        rf.set_params(max_depth=list_maxDepth[np.argmin(np.array(list_loss))])
        # Refitting the random forest model with the best max_depth parameter
        rf.fit(X_scaled, y_train)

        X_test_scaled = X_test.copy()
        X_test_scaled[:, list_cont] = scaler_x.transform(X_test[:, list_cont])

        if prob_type == 'regression':
            org_pred = rf.predict(X_test_scaled)
        else:
            org_pred = rf.predict_proba(X_test_scaled)[:, 1]

        # Check whether to include the indices or the values of the groups into the group-based part
        if len(groups) != 0:
            if not group_stacking:
                list_cols = groups
            else:
                list_cols = [[val] for val in range(len(list_grps))]

        rng = np.random.RandomState(2023)
        list_seeds_imp = rng.randint(1e5, size=n_perm)
        if noImp:
            return (X_test, y_test, org_pred)

        parallel = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)
        p2_score = np.array(parallel(delayed(joblib_compute_perm_conditional)(list_cols[p_col], n_perm, [[rf, scaler_x, scaler_y]], [X_test_scaled], y_test, prob_type,
                                                                    org_pred, seed=random_state, dict_cont=dict_cont, dict_nom=dict_nom,
                                                                    X_nominal=X_nominal, list_nominal=list_nominal, encoder=enc_dict, proc_col=p_col,
                                                                    index_i=index_i, group_stacking=group_stacking, list_seeds=list_seeds_imp)
                                    for p_col in range(len(list_cols))))

        if not res:
            return p2_score, org_pred, score(y_test, org_pred)
        # p2_score (nb_features (p) or nb_groups x nb_permutations x length_ytest)
        results['importance'] = np.mean(np.mean(p2_score, axis=2), axis=1)
        results['std'] = np.std(np.mean(p2_score, axis=1),
                                axis=1) / np.sqrt(len(y_test)-1)
        results['pval'] = norm.sf(results['importance'] / results['std'])
        results['pval'][np.isnan(results['pval'])] = 1
        results['score'] = score(y_test, org_pred)
        if index_i != None:
            print(f"Done processing iteration/fold: {index_i}")
        return results
    else:
        valid_ind = []
        score_l = []

        kf = KFold(n_splits=k_fold, random_state=random_state, shuffle=True)
        # if len(groups) > 0:
        #     p2_score = np.empty((len(groups), n_perm, n))
        # else:
        #     p2_score = np.empty((p, n_perm, n))
        p2_score = [None] * max(1, k_fold)
        for index_i, (train_index, test_index) in enumerate(kf.split(X_train)):
            print(f"Fold: {index_i+1}")
            valid_ind.append((train_index, test_index))
            p2_score[index_i], org_pred, score_val = permfit_RF(X_train, y_train, train_index, test_index, prob_type,
                                                                      k_fold=0, split_perc=split_perc, random_state=random_state,
                                                                      n_ensemble=n_ensemble, batch_size=batch_size, n_perm=n_perm,
                                                                      res=False, conditional=conditional, n_jobs=n_jobs,
                                                                      list_nominal=list_nominal, max_depth=max_depth, index_i=index_i+1,
                                                                      groups=groups, group_stacking=group_stacking, n_out_subLayers=n_out_subLayers)
            print(f"Done Fold: {index_i+1}")
            score_l.append(score_val)
        p2_score = np.mean(np.array(p2_score), axis=0)
        results = {}
        results['importance'] = np.mean(np.mean(p2_score, axis=2), axis=1)
        results['std'] = np.std(np.mean(p2_score, axis=1),
                                axis=1) / np.sqrt(p2_score.shape[2]-1)
        results['pval'] = norm.sf(results['importance'] / results['std'])
        results['pval'][np.isnan(results['pval'])] = 1
        results['score'] = np.mean(score_l)
        # results['validation_ind'] = valid_ind
        return results
