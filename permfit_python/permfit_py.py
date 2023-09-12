import itertools
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import torch
from .deep_model_standard import ensemble_dnnet
from .compute_perm_conditional_standard import (pred_avg, joblib_compute_perm,
    joblib_compute_perm_conditional)


def permfit(X_train, y_train, train_index=None, test_index=None, prob_type='regression', k_fold=0,
            split_perc=0.2, random_state=2023, n_ensemble=10, batch_size=1024, n_perm=50,
            res=True, verbose=0, conditional=False, n_jobs=1, backend = "loky",
            list_nominal = {}, max_depth=2, index_i=None, groups=[], group_stacking=False,
            n_out_subLayers=1, noImp=False, noPerm=False):
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
    noImp: If True, returns the results of the learner block without computing the importance
    noPerm: If True, a random sampling with replacement is applied on the residuals (CPI-DNN)
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
            for col_encode in list(set(list_nominal['nominal']) & set(list_cols)):
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
        list_cont = [el[0] for el in dict_cont.values()]

        X_train = X_train.to_numpy()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if prob_type == 'classification':
            # Converting target values to corresponding integer values
            ind_unique = np.unique(y_train)
            dict_target = dict(zip(ind_unique, range(len(ind_unique))))
            y_train = np.array([dict_target[el]
                                for el in list(y_train)]).reshape(-1, 1)
            score = roc_auc_score
        else:
            y_train = np.array(y_train).reshape(-1, 1)
            score = r2_score
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
                        current_grp += dict_cont[i]
                list_grps.append(current_grp)

        parallel = Parallel(n_jobs=min(n_jobs, 10), verbose=verbose, backend=backend)

        # Find the list of optimal sub-models to be used in the following steps
        optimal_list = ensemble_dnnet(X_train, y_train, n_ensemble=n_ensemble,
                                      prob_type=prob_type, verbose=verbose,
                                      random_state=random_state, parallel=parallel,
                                      list_cont=list_cont, list_grps=list_grps,
                                      group_stacking=group_stacking,
                                      n_out_subLayers=n_out_subLayers)

        list_cols = [[val] for val in list_cols]

        # The input will be either the original input or the result of the provided sub-linear layers
        # in a stacking way for the different groups
        X_test_n = [None] * len(optimal_list)
        for mod in range(len(optimal_list)):
            X_test_scaled = X_test.copy()
            if group_stacking:
                X_test_scaled[:, list_cont] = optimal_list[mod][1][0].transform(X_test_scaled[:, list_cont])
                X_test_n_curr = np.zeros((X_test_scaled.shape[0], len(list_grps) * n_out_subLayers))
                for grp_ind in range(len(list_grps)):
                    X_test_n_curr[:, list(np.arange(n_out_subLayers*grp_ind, (grp_ind+1)*n_out_subLayers))] = \
                    X_test_scaled[:, list_grps[grp_ind]].dot(optimal_list[mod][0][3][grp_ind]) \
                    + optimal_list[mod][0][4][grp_ind]
                X_test_n[mod] = X_test_n_curr
            else:
                X_test_n[mod] = X_test_scaled

        # Check whether to include the indices or the values of the groups into the group-based part
        if len(groups) != 0:
            if not group_stacking:
                list_cols = groups
            else:
                list_cols = [[val] for val in range(len(list_grps))]

        org_pred = pred_avg(optimal_list, X_test_n, y_test, prob_type,
                            list_cont, group_stacking=group_stacking)

        list_seeds_imp = rng.randint(1e5, size=n_perm)
        if noImp:
            return (X_test, y_test, org_pred)
        parallel = Parallel(n_jobs=n_jobs, verbose=verbose, backend=backend)
        if not conditional:
            p2_score = np.array(parallel(delayed(joblib_compute_perm)(list_cols[p_col], perm, optimal_list, X_test_n, y_test, prob_type,
                                                                      org_pred, seed=random_state, dict_cont=dict_cont, dict_nom=dict_nom,
                                                                      proc_col=p_col, index_i=index_i, group_stacking=group_stacking,
                                                                      random_state=list_seeds_imp[perm])
                                        for p_col in range(len(list_cols)) for perm in range(n_perm))).reshape((len(list_cols), n_perm, len(y_test)))
        else:
            # print("Apply Conditional!")
            p2_score = np.array(parallel(delayed(joblib_compute_perm_conditional)(list_cols[p_col], n_perm, optimal_list, X_test_n, y_test, prob_type,
                                                                     org_pred, seed=random_state, dict_cont=dict_cont, dict_nom=dict_nom,
                                                                     X_nominal=X_nominal, list_nominal=list_nominal, encoder=enc_dict, proc_col=p_col,
                                                                     index_i=index_i, group_stacking=group_stacking, list_seeds=list_seeds_imp, noPerm=noPerm)
                                        for p_col in range(len(list_cols))))
            results['RF_depth'] = max_depth
        if not res:
            return p2_score, org_pred, score(y_test, org_pred)
        # p2_score (nb_features (p) or nb_groups x nb_permutations x length_ytest)
        results['importance'] = np.mean(np.mean(p2_score, axis=2), axis=1)
        results['std'] = np.std(np.mean(p2_score, axis=1),
                                axis=1) / np.sqrt(len(y_test)-1)
        results['pval'] = norm.sf(results['importance'] / results['std'])
        results['score'] = score(y_test, org_pred)
        if index_i != None:
            print(f"Done processing iteration/fold: {index_i}")
        return results
    else:
        valid_ind = []
        score_l = []
        kf = KFold(n_splits=k_fold, random_state=random_state, shuffle=True)
        if len(groups) > 0:
            p2_score = np.empty((len(groups), n_perm, n))
        else:
            p2_score = np.empty((p, n_perm, n))

        for index_i, (train_index, test_index) in enumerate(kf.split(X_train)):
            print(f"Fold: {index_i+1}")
            valid_ind.append((train_index, test_index))
            p2_score[:, :, test_index], org_pred, score_val = permfit(X_train, y_train, train_index, test_index, prob_type,
                                                                      k_fold=0, split_perc=split_perc, random_state=random_state,
                                                                      n_ensemble=n_ensemble, batch_size=batch_size, n_perm=n_perm,
                                                                      res=False, conditional=conditional, n_jobs=n_jobs,
                                                                      list_nominal=list_nominal, max_depth=max_depth, index_i=index_i+1,
                                                                      groups=groups, group_stacking=group_stacking, n_out_subLayers=n_out_subLayers,
                                                                      noPerm=noPerm)
            print(f"Done Fold: {index_i+1}")
            score_l.append(score_val)

        results = {}
        results['importance'] = np.mean(np.mean(p2_score, axis=2), axis=1)
        results['std'] = np.std(np.mean(p2_score, axis=1),
                                axis=1) / np.sqrt(p2_score.shape[2]-1)
        results['pval'] = norm.sf(results['importance'] / results['std'])
        results['score'] = np.mean(score_l)
        results['validation_ind'] = valid_ind
        return results

