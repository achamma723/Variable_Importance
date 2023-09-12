import numpy as np
import math
from joblib import delayed
from .utils import create_X_y, sigmoid, relu, relu_
from sklearn.metrics import mean_squared_error, log_loss


def dnn_net(X_train, y_train, X_valid, y_valid, n_hidden, 
            noStack=False, list_grps=[], n_out_subLayers=1, n_batch=50,
            n_epoch=200, prob_type="regression", beta1=0.9, beta2=0.999,
            lr=1e-3, epsilon=1e-8, l1_weight=1e-4, l2_weight=0,
            random_state=None):
    n_layer = len(n_hidden)
    n, p = X_train.shape
    n_subLayers = len(list_grps)
    input_dim = (n_subLayers * n_out_subLayers) if not noStack else p
    rng = np.random.RandomState(random_state)

    loss = [None] * n_epoch
    best_loss = math.inf
    best_weight = []
    best_bias = []
    best_weight_stack = []
    best_bias_stack = []

    weight_stack = [None]*(n_subLayers)
    bias_stack = [None]*(n_subLayers)

    weight = [None]*(n_layer+1)
    bias = [None]*(n_layer+1)
    a = [None]*(n_layer+1)
    h = [None]*(n_layer+1)
    d_a = [None]*(n_layer+1)
    d_h = [None]*(n_layer+1)
    d_w = [None]*(n_layer+1)
    dw = [None]*(n_layer+1)
    db = [None]*(n_layer+1)

    # ADAM
    mt_ind = 0
    mt_w_stack = [None]*(n_subLayers)
    vt_w_stack = [None]*(n_subLayers)
    mt_b_stack = [None]*(n_subLayers)
    vt_b_stack = [None]*(n_subLayers)

    mt_w = [None]*(n_layer+1)
    vt_w = [None]*(n_layer+1)
    mt_b = [None]*(n_layer+1)
    vt_b = [None]*(n_layer+1)

    if not noStack:
        # Initialization of the stacking sub-layers
        # 1 sub-layer corresponds to 1 group
        for sub_layer in range(n_subLayers):
            weight_stack[sub_layer] = rng.uniform(-1, 1, (len(list_grps[sub_layer]), n_out_subLayers)) * 0.1
            bias_stack[sub_layer] = rng.uniform(-1, 1, (1, n_out_subLayers)) * 0.05
            mt_w_stack[sub_layer] = np.zeros((len(list_grps[sub_layer]), n_out_subLayers))
            mt_b_stack[sub_layer] = np.zeros((1, n_out_subLayers))
            vt_w_stack[sub_layer] = np.zeros((len(list_grps[sub_layer]), n_out_subLayers))
            vt_b_stack[sub_layer] = np.zeros((1, n_out_subLayers))
    
    # Initialization of the remaining layers
    for i in range(n_layer+1):
        if i == 0:
            weight[i] = rng.uniform(-1, 1, (input_dim, n_hidden[i])) * 0.1
            bias[i] = rng.uniform(-1, 1, (1, n_hidden[i])) * 0.05
            mt_w[i] = np.zeros((input_dim, n_hidden[i]))
            mt_b[i] = np.zeros((1, n_hidden[i]))
            vt_w[i] = np.zeros((input_dim, n_hidden[i]))
            vt_b[i] = np.zeros((1, n_hidden[i]))
        elif i == n_layer:
            weight[i] = rng.uniform(-1, 1, (n_hidden[i-1], 1)) * 0.1
            bias[i] = rng.uniform(-1, 1, (1, 1)) * 0.05
            mt_w[i] = np.zeros((n_hidden[i-1], 1))
            mt_b[i] = np.zeros((1, 1))
            vt_w[i] = np.zeros((n_hidden[i-1], 1))
            vt_b[i] = np.zeros((1, 1))
        else:
            weight[i] = rng.uniform(-1, 1, (n_hidden[i-1], n_hidden[i])) * 0.1
            bias[i] = rng.uniform(-1, 1, (1, n_hidden[i])) * 0.05
            mt_w[i] = np.zeros((n_hidden[i-1], n_hidden[i]))
            mt_b[i] = np.zeros((1, n_hidden[i]))
            vt_w[i] = np.zeros((n_hidden[i-1], n_hidden[i]))
            vt_b[i] = np.zeros((1, n_hidden[i]))

    # Getting the batches ready
    n_round = int(np.ceil(n/n_batch))
    i_bgn = np.empty(n_round, dtype=np.int32)
    i_end = np.empty(n_round, dtype=np.int32)

    for s in range(n_round):
        i_bgn[s] = s*n_batch
        i_end[s] = min((s+1)*n_batch, n)

    for k in range(n_epoch):
        # Shuffling the instances
        new_order = rng.choice(n, n, replace=False)
        X_train_n = X_train[new_order, :]
        y_train_n = y_train[new_order]

        for i in range(n_round):
            # Going through the different batches
            xi = X_train_n[i_bgn[i]:i_end[i], :]
            yi = y_train_n[i_bgn[i]:i_end[i]]
            
            if noStack:
                xi_n = xi.copy()
            else:
                xi_n = np.zeros((xi.shape[0], n_subLayers * n_out_subLayers))
                # Linear sub-layers without activation (Stacking groups)
                for grp_ind in range(len(list_grps)):
                    xi_n[:, list(np.arange(n_out_subLayers*grp_ind,
                                           (grp_ind+1)*n_out_subLayers))] = \
                        xi[:, list_grps[grp_ind]].dot(weight_stack[grp_ind]) \
                            + bias_stack[grp_ind]

            for j in range(n_layer):
                if j==0:
                    a[j] = xi_n.dot(weight[j]) + bias[j]
                else:
                    a[j] = h[j-1].dot(weight[j]) + bias[j]
                h[j] = relu(a[j])

            y_pi = h[n_layer-1].dot(weight[n_layer]) + bias[n_layer]

            if prob_type=="classification":
                y_pi = sigmoid(y_pi)

            # Backward propagation
            d_a[n_layer] = -(yi - y_pi) / len(yi)
            d_w[n_layer] = h[n_layer-1].T.dot(d_a[n_layer])
            bias_grad = d_a[n_layer].T.dot(np.ones((d_a[n_layer].shape[0], 1)))

            mt_ind += 1
            mt_w[n_layer] = mt_w[n_layer] * beta1 + (1-beta1) * d_w[n_layer]
            mt_b[n_layer] = mt_b[n_layer] * beta1 + (1-beta1) * bias_grad
            vt_w[n_layer] = vt_w[n_layer] * beta2 + (1-beta2) * d_w[n_layer]**2
            vt_b[n_layer] = vt_b[n_layer] * beta2 + (1-beta2) * bias_grad**2
            dw[n_layer] = lr * mt_w[n_layer] / (1-beta1**mt_ind) / (np.sqrt(vt_w[n_layer] / (1-beta2**mt_ind)) + epsilon)
            db[n_layer] = lr * mt_b[n_layer] / (1-beta1**mt_ind) / (np.sqrt(vt_b[n_layer] / (1-beta2**mt_ind)) + epsilon)
            weight[n_layer] = weight[n_layer] - dw[n_layer] - l1_weight * ((weight[n_layer] > 0)*1 \
                                                                           - (weight[n_layer] < 0)*1) \
                                                                            - l2_weight * weight[n_layer]
            bias[n_layer] = bias[n_layer] - db[n_layer]

            for j in range((n_layer-1), -1, -1):
                d_h[j] = d_a[j+1].dot(weight[j+1].T)
                d_a[j] = d_h[j] * relu_(a[j])
                if(j>0):
                    d_w[j] = h[j-1].T.dot(d_a[j])
                else:
                    d_w[j] = xi_n.T.dot(d_a[j])
                bias_grad = np.ones((d_a[j].shape[0], 1)).T.dot(d_a[j])
                mt_w[j] = mt_w[j] * beta1 + (1-beta1) * d_w[j]
                mt_b[j] = mt_b[j] * beta1 + (1-beta1) * bias_grad
                vt_w[j] = vt_w[j] * beta2 + (1-beta2) * d_w[j]**2
                vt_b[j] = vt_b[j] * beta2 + (1-beta2) * bias_grad**2
                dw[j] = lr * mt_w[j] / (1-beta1**mt_ind) / (np.sqrt(vt_w[j] / (1-beta2**mt_ind)) + epsilon)
                db[j] = lr * mt_b[j] / (1-beta1**mt_ind) / (np.sqrt(vt_b[j] / (1-beta2**mt_ind)) + epsilon)

                weight[j] = weight[j] - dw[j] - l1_weight * ((weight[j] > 0)*1 - (weight[j] < 0)*1) - l2_weight * weight[j]
                bias[j] = bias[j] - db[j]
            
            if not noStack:
                d_h_stack = d_a[0].dot(weight[0].T)
                # The derivative of the identity function is 1
                d_a_stack = d_h_stack * 1

                bias_grad = np.ones((d_a_stack.shape[0], 1)).T.dot(d_a_stack)
                # Including groups stacking in backward propagation
                for grp_ind in range(len(list_grps)):
                    d_w_stack = xi[:, list_grps[grp_ind]].T.dot(d_a_stack[:, [grp_ind]])

                    mt_w_stack[grp_ind] = mt_w_stack[grp_ind] * beta1 + (1-beta1) * d_w_stack
                    mt_b_stack[grp_ind] = mt_b_stack[grp_ind] * beta1 + (1-beta1) * bias_grad[:, [grp_ind]]
                    vt_w_stack[grp_ind] = vt_w_stack[grp_ind] * beta2 + (1-beta2) * d_w_stack**2
                    vt_b_stack[grp_ind] = vt_b_stack[grp_ind] * beta2 + (1-beta2) * bias_grad[:, [grp_ind]]**2

                    dw_stack = lr * mt_w_stack[grp_ind] / (1-beta1**mt_ind) / (np.sqrt(vt_w_stack[grp_ind] \
                                                                                       / (1-beta2**mt_ind)) + epsilon)
                    db_stack = lr * mt_b_stack[grp_ind] / (1-beta1**mt_ind) / (np.sqrt(vt_b_stack[grp_ind] \
                                                                                       / (1-beta2**mt_ind)) + epsilon)

                    weight_stack[grp_ind] = weight_stack[grp_ind] - dw_stack - l1_weight * ((weight_stack[grp_ind] > 0)*1 - \
                    (weight_stack[grp_ind] < 0)*1) - l2_weight * weight_stack[grp_ind]
                    bias_stack[grp_ind] = bias_stack[grp_ind] - db_stack

        if noStack:
            X_valid_n = X_valid.copy()
        else:
            X_valid_n = np.zeros((X_valid.shape[0], n_subLayers * n_out_subLayers))
            for grp_ind in range(len(list_grps)):
                X_valid_n[:, list(np.arange(n_out_subLayers*grp_ind, (grp_ind+1)*n_out_subLayers))] = \
                    X_valid[:, list_grps[grp_ind]].dot(weight_stack[grp_ind]) + bias_stack[grp_ind]            

        for j in range(n_layer):
            if j == 0:
                pred = relu(X_valid_n.dot(weight[j]) + bias[j])
            else:
                pred = relu(pred.dot(weight[j]) + bias[j])

        pred = (pred.dot(weight[n_layer]) + bias[n_layer])[:, 0]

        if(prob_type == "classification"):
            pred = sigmoid(pred)
            loss[k] = log_loss(y_valid, pred)
        else:
            loss[k] = mean_squared_error(y_valid, pred)
        
        if(loss[k] < best_loss):
            best_loss = loss[k]
            best_weight = weight.copy()
            best_bias = bias.copy()
            best_weight_stack = weight_stack.copy()
            best_bias_stack = bias_stack.copy()

    return [best_weight, best_bias, best_loss, best_weight_stack, best_bias_stack]


def ensemble_dnnet(X, y, n_ensemble=10, verbose=0, bootstrap=True, split_perc=0.8,
                   batch_size=32, batch_size_val=128, n_epoch=200, min_keep=10, prob_type='regression',
                   n_hidden=np.array([50, 40, 30, 20]), random_state=None, parallel=None,
                   list_cont=[], list_grps=[], group_stacking=False, n_out_subLayers=1):
    '''
    X: The matrix of predictors
    y: The response vector
    n_ensemble: The number of sub-DNNs to be fit
    verbose: If > 0, the fitted iterations will be printed
    bootstrap: If True, a bootstrap sampling is used
    split_perc: If bootstrap==False, a training/validation cut for the data will be used
    batch_size: The number of samples per batch for training
    batch_size_val: The number of samples per batch for validation
    n_epoch: The number of epochs for the DNN learner
    min_keep: The minimal number of DNNs to be kept
    prob_type: A classification or regression problem
    n_hidden: The number of neurones per layer for the DNN learner
    n_jobs: The number of workers for parallel processing
    '''
    n, p = X.shape
    n_layer = len(n_hidden)
    min_keep = max(min(min_keep, n_ensemble), 1)
    rng = np.random.RandomState(random_state)
    list_seeds = rng.randint(1e5, size=(n_ensemble+1))

    # Fine-tuning the hyperparameters
    list_hyper = []
    lr = [1e-2, 1e-3, 1e-4]
    l1 = [0, 1e-2, 1e-4]
    l2 = [0, 1e-2, 1e-4]
    X_train_scaled, y_train_scaled, X_valid_scaled, y_valid_scaled, _, __, ___, ____, \
    _____ = create_X_y(X, y, bootstrap=bootstrap, split_perc=split_perc,
                       prob_type=prob_type, list_cont=list_cont, random_state=list_seeds[-1])
    # Check for the stacking method flag
    noStack = False if group_stacking else True

    for el_lr in lr:
        for el_l1 in l1:
            for el_l2 in l2:
                list_hyper.append((el_lr, el_l1, el_l2))
    res_hyper = list(zip(*parallel(delayed(dnn_net)(X_train_scaled, y_train_scaled, X_valid_scaled,
                                                    y_valid_scaled, n_hidden=n_hidden, n_batch=batch_size,
                                                    n_epoch=n_epoch, prob_type=prob_type, lr=el[0], l1_weight=el[1],
                                                    l2_weight=el[2], list_grps=list_grps, n_out_subLayers=n_out_subLayers,
                                                    random_state=list_seeds[-1], noStack=noStack) for el in list_hyper)))[2]

    ind_min = np.argmin(res_hyper)
    best_hyper = [list_hyper[ind_min][0], list_hyper[ind_min][1], list_hyper[ind_min][2]] # Lr, L1 weight, L2 weight

    res_ens = list(zip(*parallel(delayed(joblib_ensemble_dnnet)(X, y, n, bootstrap, split_perc, n_hidden,
                        n_layer, batch_size, n_epoch, prob_type, verbose, list_cont=list_cont, lr=best_hyper[0],
                        l1_weight=best_hyper[1], l2_weight=best_hyper[2], list_grps=list_grps, noStack=noStack,
                        n_out_subLayers=n_out_subLayers, random_state=list_seeds[i]) for i in range(n_ensemble))))

    pred_m = np.array(res_ens[3])
    loss = np.array(res_ens[4])

    if n_ensemble == 1:
        return [(res_ens[0][0], (res_ens[1][0], res_ens[2][0]))]
    # Keeping the optimal subset of DNNs
    sorted_loss = loss.copy()
    sorted_loss.sort()
    new_loss = np.empty(n_ensemble-1)
    for i in range(n_ensemble-1):
        current_pred = np.mean(pred_m[loss >= sorted_loss[i], :], axis=0)
        if prob_type == 'regression':
            new_loss[i] = mean_squared_error(y, current_pred)
        else:
            new_loss[i] = log_loss(y, current_pred)
    keep_dnn = loss >= sorted_loss[np.argmin(
        new_loss[:(n_ensemble - min_keep + 1)])]
    return [(res_ens[0][i], (res_ens[1][i], res_ens[2][i])) for i in range(n_ensemble) if keep_dnn[i] == True]


def joblib_ensemble_dnnet(X, y, n, bootstrap, split_perc, n_hidden, n_layer,
                          batch_size, n_epoch, prob_type, verbose, list_cont=[],
                          lr=1e-3, l1_weight=1e-2, l2_weight=0, list_grps=[],
                          noStack=False, n_out_subLayers=1, random_state=None):
    pred_v = np.empty(n)
    # Sampling and Train/Validate splitting
    X_train_scaled, y_train_scaled, X_valid_scaled, y_valid_scaled, X_scaled, y_valid, \
    scaler_x, scaler_y, valid_ind = create_X_y(X, y, bootstrap=bootstrap, split_perc=split_perc,
                                               prob_type=prob_type, list_cont=list_cont,
                                               random_state=random_state)

    current_model = dnn_net(X_train_scaled, y_train_scaled, X_valid_scaled, y_valid_scaled, 
        n_hidden=n_hidden, n_batch=batch_size, n_epoch=n_epoch, prob_type=prob_type, lr=lr,
        l1_weight=l1_weight, l2_weight=l2_weight, list_grps=list_grps, n_out_subLayers=n_out_subLayers,
        random_state=random_state, noStack=noStack)
    if noStack:
        X_scaled_n = X_scaled.copy()
    else:
        X_scaled_n = np.zeros((X_scaled.shape[0], len(list_grps) * n_out_subLayers))
        for grp_ind in range(len(list_grps)):
            X_scaled_n[:, list(np.arange(n_out_subLayers*grp_ind, (grp_ind+1)*n_out_subLayers))] = \
                X_scaled[:, list_grps[grp_ind]].dot(current_model[3][grp_ind]) + current_model[4][grp_ind]

    for j in range(n_layer):
        if j == 0:
            pred = relu(X_scaled_n.dot(current_model[0][j]) + current_model[1][j])
        else:
            pred = relu(pred.dot(current_model[0][j]) + current_model[1][j])
    
    pred = (pred.dot(current_model[0][n_layer]) + current_model[1][n_layer])[:, 0]

    if prob_type == 'regression':
        pred_v = pred * scaler_y.scale_ + scaler_y.mean_
        loss = np.std(y_valid) ** 2 - mean_squared_error(y_valid, pred_v[valid_ind])
    else:
        pred_v = sigmoid(pred)
        loss = log_loss(y_valid, np.ones(len(y_valid))*np.mean(
            y_valid)) - log_loss(y_valid, pred_v[valid_ind])
    
    return (current_model, scaler_x, scaler_y, pred_v, loss)
