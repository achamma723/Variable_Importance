import numpy as np
from .utils import relu, sigmoid
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

def pred_avg(optimal_list, X_test_list, y_test, prob_type, list_cont=[],
             group_stacking=False):
    org_pred = np.zeros(len(y_test))
    n_layer = len(optimal_list[0][0][0]) - 1
    counter_mod = 0
    for mod in optimal_list:
        for j in range(n_layer):
            X_test_scaled = X_test_list[counter_mod].copy()
            if not group_stacking:
                if len(list_cont) > 0:
                    X_test_scaled[:, list_cont] = mod[1][0].transform(X_test_scaled[:, list_cont])
            if j == 0:
                pred = relu(X_test_scaled.dot(mod[0][0][j]) + mod[0][1][j])
            else:
                pred = relu(pred.dot(mod[0][0][j]) + mod[0][1][j])
        
        pred = (pred.dot(mod[0][0][n_layer]) + mod[0][1][n_layer])[:, 0]
        # Original Predictions
        if prob_type == 'regression':
            org_pred += pred * mod[1][1].scale_ + mod[1][1].mean_
        else:
            org_pred += sigmoid(pred)
        counter_mod += 1
    org_pred /= len(optimal_list)
    return org_pred


def conf_binary(y, y_min, y_max):
    y_new = np.empty(len(y))
    for el in range(len(y)):
        if y[el] < y_min:
            y_new[el] = y_min
        elif y[el] > y_max:
            y_new[el] = y_max
        else:
            y_new[el] = y[el]
    return y_new


def joblib_compute_perm_conditional(p_col, n_sample, optimal_list, X_test_list,
                                    y_test, prob_type, org_pred, seed=None,
                                    dict_cont={}, dict_nom={}, X_nominal=None, list_nominal={},
                                    encoder={}, proc_col=None, index_i=None, group_stacking=False,
                                    list_seeds=None, noPerm=False):
    list_cont = [el[0] for el in dict_cont.values()]
    rng = np.random.RandomState(seed)
    # if index_i != None:
    #     print(f"Iteration/Fold:{index_i}, Processing col:{proc_col}")
    # else:
    #     print(f"Processing col:{proc_col}")
    res_ar = np.empty((n_sample, len(y_test)))
    y_test_new = np.ravel(y_test)

    # A list of copied items to avoid any overlapping in the process
    current_X_test_list = [X_test_el.copy() for X_test_el in X_test_list]
    # Without the stacking part, the same element is repeated in the list
    if not group_stacking:
        current_X_test_list = [current_X_test_list[0]]
    Res_col = [None] * len(optimal_list)
    X_col_pred = {'regr': [None] * len(optimal_list), 'class': None}
    counter_test = 0

    grp_nom = [item for item in p_col if item in dict_nom.keys()]
    grp_cont = [item for item in p_col if item not in grp_nom]
    p_col_n = {"regr": [], "class": []}
    if not group_stacking:
        for val in p_col:
            if val in grp_nom:
                p_col_n['class'] += dict_nom[val]
            if val in grp_cont:
                p_col_n['regr'] += dict_cont[val]
    else:
        p_col_n['regr'] = p_col

    # Dictionary of booleans checking for the encountered type of group
    var_type = {"regr": False, "class": False}

    X_col_new = {"regr": None, "class": None}
    output = {"regr": None, "class": None}
    rf_models = {"regr": RandomForestRegressor(max_depth=5, random_state=seed),
                 "class": RandomForestClassifier(max_depth=5, random_state=seed)}

    # The case of pure nominal groups
    if len(grp_cont) == 0:
        var_type["class"] = True
    # Check the case for heterogenous groups (nominal + continuous)
    elif (len(grp_nom)>0) and (len(grp_cont)>0):
        var_type["regr"] = True
        var_type["class"] = True
    # The case of pure continuous groups
    else:
        var_type["regr"] = True

    for X_test_comp in current_X_test_list:
    # Check for homogeneous vs heterogeneous groups
        X_test_minus_idx = np.delete(np.copy(X_test_comp), p_col_n['regr'] + p_col_n['class'], 1)
        if var_type["regr"]:
            output['regr'] = X_test_comp[:, p_col_n['regr']]
            rf_models['regr'].fit(X_test_minus_idx, output['regr'])
            X_col_pred['regr'][counter_test] = rf_models['regr'].predict(X_test_minus_idx)
            X_col_pred['regr'][counter_test] = X_col_pred['regr'][counter_test].reshape(-1, output['regr'].shape[1])
            Res_col[counter_test] = output['regr'] - X_col_pred['regr'][counter_test]
        counter_test += 1

    # The loop doesn't include the classification part because the input across the different DNN sub-models
    # is the same without the stacking part (where extra sub-linear layers are used), therefore identical inputs
    # won't need looping classification process. This is not the case with the regression part
    if var_type["class"]:
        list_int = [i for i, v in X_nominal.dtypes.items() if v == np.int]
        output['class'] = X_nominal[grp_nom].astype(str)
        rf_models['class'].fit(X_test_minus_idx, output['class'])
        X_col_pred['class'] = rf_models['class'].predict_proba(X_test_minus_idx)

        if not isinstance(X_col_pred['class'], list):
            tmp_list = []
            tmp_list.append(X_col_pred['class'])
            X_col_pred['class'] = tmp_list

        counter = 0
        # X_col_pred['class'] is a list of arrays representing the probability of getting each value at the 
        # corresponding variable (resp. to each observation)
        list_seed_cat = rng.randint(1e5, size=len(X_col_pred['class']))
        for cat_prob_ind in range(len(X_col_pred['class'])):
            rng_cat = np.random.RandomState(list_seed_cat[cat_prob_ind])
            current_X_col_new = np.array([[rng_cat.choice(np.unique(X_nominal[[grp_nom[cat_prob_ind]]]),
                size=1, p=X_col_pred['class'][cat_prob_ind][i])[0] for i in range(X_col_pred['class'][cat_prob_ind].shape[0])]]).T
            if grp_nom[cat_prob_ind] in list_nominal['nominal']:
                current_X_col_new = encoder[grp_nom[cat_prob_ind]].transform(current_X_col_new).toarray().astype('int32')
            if counter == 0:
                X_col_new['class'] = current_X_col_new
            else:
                X_col_new['class'] = np.concatenate((X_col_new['class'], current_X_col_new), axis=1)
            counter += 1

    for sample in range(n_sample):
        # Same shuffled indices across the sub-models items
        indices = np.arange(len(y_test))
        rng = np.random.RandomState(list_seeds[sample])
        if noPerm:
            indices = rng.choice(indices, size=len(indices))
        else:
            rng.shuffle(indices)

        counter_test = 0
        for X_test_comp in current_X_test_list:
            if var_type["regr"]:
                X_test_comp[:, p_col_n['regr']] = X_col_pred['regr'][counter_test] + Res_col[counter_test][indices, :]

            if var_type["class"]:
                # Check if to remove the zero columns from the one-hot encoding
                X_test_comp[:, p_col_n['class']] = X_col_new['class']
            
            counter_test += 1
        # return current_X_test_list
        if not(isinstance(optimal_list[0][0], RandomForestRegressor) or isinstance(optimal_list[0][0], RandomForestClassifier)):
            if not group_stacking:
                current_X_test_pred = current_X_test_list * len(optimal_list)
            else:
                current_X_test_pred = current_X_test_list.copy()

            pred_i = pred_avg(optimal_list, current_X_test_pred, y_test,
                              prob_type, list_cont=list_cont,
                              group_stacking=group_stacking)
        else:
            if prob_type == 'regression':
                pred_i = optimal_list[0][0].predict(current_X_test_list[0])
                # pred_i = pred_i * optimal_list[0][2].scale_ + optimal_list[0][2].mean_
            else:
                pred_i = optimal_list[0][0].predict_proba(current_X_test_list[0])[:, 1]

        if prob_type == 'regression':
            res_ar[sample, :] = (
                y_test_new - pred_i) ** 2 - (y_test_new - org_pred) ** 2
        else:
            y_max = 1-1e-10
            y_min = 1e-10
            pred_i = conf_binary(pred_i, y_min, y_max)
            org_pred = conf_binary(org_pred, y_min, y_max)
            res_ar[sample, :] = -y_test_new*np.log(pred_i) - (1-y_test_new)*np.log(1-pred_i) \
                + y_test_new*np.log(org_pred) + \
                (1-y_test_new)*np.log(1-org_pred)
    return res_ar


def joblib_compute_perm(p_col, perm, optimal_list, X_test_list, y_test, prob_type,
                        org_pred, seed=None, dict_cont={}, dict_nom={}, proc_col=None,
                        index_i=None, group_stacking=False, random_state=None):
    rng = np.random.RandomState(random_state)
    list_cont = [el[0] for el in dict_cont.values()]
    y_test_new = np.ravel(y_test)
    # if index_i != None:
    #     print(f"Iteration/Fold:{index_i}, Processing col:{proc_col+1}, Permutation:{perm+1}")
    # else:
    #     print(f"Processing col:{proc_col+1}, Permutation:{perm+1}")

    current_X_test_list = [X_test_el.copy() for X_test_el in X_test_list]
    # Without the stacking part, the same element is repeated in the list
    if not group_stacking:
        current_X_test_list = [current_X_test_list[0]]
    indices = np.arange(X_test_list[0].shape[0])
    rng.shuffle(indices)

    if not group_stacking:
        p_col_new = []
        for val in p_col:
            if val in dict_nom.keys():
                p_col_new += dict_nom[val]
            else:
                p_col_new += dict_cont[val]
    else:
        p_col_new = p_col

    for X_test_comp in current_X_test_list:
        X_test_comp[:, p_col_new] = X_test_comp[:, p_col_new][indices, :]

    if not group_stacking:
        current_X_test_list = current_X_test_list * len(optimal_list)

    pred_i = pred_avg(optimal_list, current_X_test_list, y_test, prob_type,
                      list_cont=list_cont, group_stacking=group_stacking)
    if prob_type == 'regression':
        res = (y_test_new - pred_i) ** 2 - (y_test_new - org_pred) ** 2
    else:
        y_max = 1-1e-10
        y_min = 1e-10
        pred_i = conf_binary(pred_i, y_min, y_max)
        org_pred = conf_binary(org_pred, y_min, y_max)
        res = -y_test_new*np.log(pred_i) - (1-y_test_new)*np.log(1-pred_i) \
            + y_test_new*np.log(org_pred) + \
            (1-y_test_new)*np.log(1-org_pred)
    return res
