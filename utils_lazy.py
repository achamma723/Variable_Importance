import torch
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error as mse
from scipy import stats as st
import time
from IPython.utils import io
from sklearn.ensemble import RandomForestRegressor
import tqdm
from networks import *

"""
VI helpers
"""
def calculate_cvg(est, se, true_vi, level=.05):
    z = st.norm.ppf((1-level/2))
    lb = est - z*se
    ub = est + z*se
    if true_vi >= lb and true_vi <= ub:
        return 1
    else:
        return 0

def dropout(X, grp):
    X = np.array(X)
    N = X.shape[0]
    X_change = np.copy(X)
    if type(grp)==int:
        X_change[:, grp] = np.ones(N) * np.mean(X[:, grp])
    else:
        for j in grp:
            X_change[:, j] = np.ones(N) * np.mean(X[:, j])
    X_change = torch.tensor(X_change, dtype=torch.float32)
    return X_change

def retrain(p, hidden_layers, j, X_train_change, y_train, X_test_change, tol=1e-3):
    retrain_nn = NN4vi(p, hidden_layers, 1)
    #tb_logger = pl.loggers.TensorBoardLogger('logs/{}'.format(exp_name), name='retrain_{}'.format(j))
    early_stopping = EarlyStopping('val_loss', min_delta=tol)

    trainset = torch.utils.data.TensorDataset(torch.tensor(X_train_change, dtype=torch.float32),
                                              torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    train_loader = DataLoader(trainset, batch_size=256)

    trainer = pl.Trainer(callbacks=[early_stopping], max_epochs=100)
    with io.capture_output() as captured: trainer.fit(retrain_nn, train_loader, train_loader)

    retrain_pred_train = retrain_nn(X_train_change)
    retrain_pred_test = retrain_nn(X_test_change)
    return retrain_pred_train, retrain_pred_test

def fake_retrain(p, full_nn, hidden_layers, j, X_train_change, y_train, X_test_change, tol=1e-5, max_epochs=10):
    retrain_nn = NN4vi(p, hidden_layers, 1)
    retrain_nn.load_state_dict(full_nn.state_dict()) # take trained model
    #tb_logger = pl.loggers.TensorBoardLogger('logs/{}'.format(exp_name), name='retrain_{}'.format(j))
    early_stopping = EarlyStopping('val_loss', min_delta=tol)

    trainset = torch.utils.data.TensorDataset(torch.tensor(X_train_change, dtype=torch.float32),
                                              torch.tensor(y_train, dtype=torch.float32).view(-1, 1))
    train_loader = DataLoader(trainset, batch_size=256)

    trainer = pl.Trainer(callbacks=[early_stopping], max_epochs=max_epochs)
    with io.capture_output() as captured: trainer.fit(retrain_nn, train_loader, train_loader)

    retrain_pred_train = retrain_nn(X_train_change)
    retrain_pred_test = retrain_nn(X_test_change)
    return retrain_pred_train, retrain_pred_test


"""
lazy training
"""

def flat_tensors(T_list: list):
    """
    Flatten a list of tensors to a vector, and store the original shapes of the tensors for future recovery
    output: Tuple[tensor, list]
    """
    info = [t.shape for t in T_list]
    res = torch.cat([t.reshape(-1) for t in T_list])
    return res, info

def recover_tensors(T: torch.Tensor, info: list):
    """
    recover the parameter tensors in order to feed into a neural network
    output: a list of tensors
    """
    i = 0
    res = []
    for s in info:
        len_s = np.prod(s)
        res.append(T[i:i+len_s].reshape(s))
        i += len_s
    return res

def extract_grad(X, full_nn):
    """
    extract gradients from trained network
    output: n x (# network params) matrix
    """
    grads = []
    n = X.shape[0]
    params_full = tuple(full_nn.parameters())
    flat_params, shape_info = flat_tensors(params_full)
    for i in range(n):
        # calculate the first order gradient wrt all parameters
        if len(X.shape) > 2:
            yi = full_nn(X[[i]])
        else:
            yi = full_nn(X[i])
        this_grad = torch.autograd.grad(yi, params_full, create_graph=True)
        flat_this_grad, _ = flat_tensors(this_grad)
        grads.append(flat_this_grad)
    grads = np.array([grad.detach().numpy() for grad in grads])
    return grads, flat_params, shape_info


def lazy_predict(grads, flat_params, full_nn, hidden_layers, shape_info, X_train, y_train, X_test, lam):
    _, p = X_train.shape
    dr_pred_train = full_nn(X_train)
    lazy = Ridge(alpha=lam).fit(grads, y_train - dr_pred_train.detach().numpy())
    delta = lazy.coef_
    lazy_retrain_params = torch.FloatTensor(delta) + flat_params
    lazy_retrain_Tlist = recover_tensors(lazy_retrain_params.reshape(-1), shape_info)
    lazy_retrain_nn = NN4vi(p, hidden_layers, 1)
    # hidden_layers probably doesn't need to be an argument here - get it from the structure
    for k, param in enumerate(lazy_retrain_nn.parameters()):
        param.data = lazy_retrain_Tlist[k]
    lazy_pred_train = lazy_retrain_nn(X_train)
    lazy_pred_test = lazy_retrain_nn(X_test)
    return lazy_pred_train, lazy_pred_test


def lazy_train_cv(full_nn, X_train_change, X_test_change, y_train, hidden_layers,
                  lam_path=np.logspace(-3, 3, 20), file=False):
    kf = KFold(n_splits=3, shuffle=True)
    errors = []
    grads, flat_params, shape_info = extract_grad(X_train_change, full_nn)
    for lam in lam_path:
        for train, test in kf.split(X_train_change):
            dr_pred_train = full_nn(X_train_change[train])
            grads_train = grads[train]
            lazy_pred_train, lazy_pred_test = lazy_predict(grads_train, flat_params, full_nn, hidden_layers, shape_info,
                                                           X_train_change[train], y_train[train], X_train_change[test],
                                                           lam)
            errors.append([lam, nn.MSELoss()(lazy_pred_test, y_train[test]).item()])

    errors = pd.DataFrame(errors, columns=['lam', 'mse'])
    lam = errors.groupby(['lam']).mse.mean().sort_values().index[0]
    lazy_pred_train, lazy_pred_test = lazy_predict(grads, flat_params, full_nn, hidden_layers, shape_info,
                                                   X_train_change, y_train, X_test_change, lam)
    return lazy_pred_train, lazy_pred_test, errors


"""
Experiment wrapped for faster simulations
"""

def vi_experiment_wrapper(X, y, network_width,  ix=None, exp_iter=1, lambda_path=np.logspace(0, 2, 10),
                          lam='cv', lazy_init='train', do_retrain=True, include_linear=False, include_rf=False,
                          early_stop = False, max_epochs=100):
    n, p = X.shape
    if ix is None:
        ix = np.arange(p)
    hidden_layers = [network_width]
    tol = 1e-3
    results = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = exp_iter)
    trainset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                              torch.tensor(y_train, dtype=torch.float32).view(-1,1))
    train_loader = DataLoader(trainset, batch_size=256)

    full_nn = NN4vi(p, hidden_layers, 1)
    early_stopping = EarlyStopping('val_loss', min_delta=tol)
    trainer = pl.Trainer(callbacks=[early_stopping])
    t0 = time.time()
    with io.capture_output() as captured: trainer.fit(full_nn, train_loader, train_loader)
    full_time = time.time() - t0
    full_pred_test = full_nn(X_test)
    results.append(['all', 'full model', full_time, 0,
                    nn.MSELoss()(full_nn(X_train), y_train).item(),
                    nn.MSELoss()(full_pred_test, y_test).item()])

    if include_linear:
        lm = LinearRegression()
        lm.fit(X_train.detach().numpy(), y_train.detach().numpy())

    if include_rf:
        rf = RandomForestRegressor()
        rf.fit(X_train.detach().numpy(), y_train.detach().numpy())

    for j in ix:
        varr = 'X' + str(j + 1)
        # DROPOUT
        X_test_change = dropout(X_test, j)
        X_train_change = dropout(X_train, j)
        dr_pred_train = full_nn(X_train_change)
        dr_pred_test = full_nn(X_test_change)
        dr_train_loss = nn.MSELoss()(dr_pred_train, y_train).item()
        dr_test_loss = nn.MSELoss()(dr_pred_test, y_test).item()
        dr_vi = nn.MSELoss()(dr_pred_test, y_test).item() - nn.MSELoss()(full_pred_test, y_test).item()

        # variance
        eps_j = ((y_test - dr_pred_test) ** 2).detach().numpy().reshape(1, -1)
        eps_full = ((y_test - full_pred_test) ** 2).detach().numpy().reshape(1, -1)
        se = np.sqrt(np.var(eps_j - eps_full)/y_test.shape[0])


        results.append([varr, 'dropout', 0, dr_vi, dr_train_loss, dr_test_loss, se])

        # LAZY
        t0 = time.time()

        if lam == 'cv':
            lazy_pred_train, lazy_pred_test, cv_res = lazy_train_cv(full_nn, X_train_change, X_test_change, y_train,
                                                                 hidden_layers, lam_path = lambda_path)
        else:
            grads, flat_params, shape_info = extract_grad(X_train_change, full_nn)
            lazy_pred_train, lazy_pred_test = lazy_predict(grads, flat_params, full_nn, hidden_layers, shape_info,
                                                           X_train_change, y_train, X_test_change, lam)
        lazy_time = time.time() - t0
        lazy_train_loss = nn.MSELoss()(lazy_pred_train, y_train).item()
        lazy_test_loss = nn.MSELoss()(lazy_pred_test, y_test).item()
        lazy_vi = nn.MSELoss()(lazy_pred_test, y_test).item() - nn.MSELoss()(full_pred_test, y_test).item()

        # variance
        eps_j = ((y_test - lazy_pred_test) ** 2).detach().numpy().reshape(1, -1)
        eps_full = ((y_test - full_pred_test) ** 2).detach().numpy().reshape(1, -1)
        se = np.sqrt(np.var(eps_j - eps_full)/y_test.shape[0])

        results.append([varr, 'lazy', lazy_time, lazy_vi, lazy_train_loss, lazy_test_loss, se])

        # LAZY
        if lazy_init == 'random':
            t0 = time.time()
            random_nn = NN4vi(p, hidden_layers, 1)
            params_full = tuple(full_nn.parameters())
            flat_params, shape_info = flat_tensors(params_full)
            lazy_retrain_Tlist = recover_tensors(flat_params.reshape(-1), shape_info)
            # hidden_layers probably doesn't need to be an argument here - get it from the structure
            for k, param in enumerate(random_nn.parameters()):
                param.data = lazy_retrain_Tlist[k] + np.random.normal(size = lazy_retrain_Tlist[k].shape)


            lazy_pred_train, lazy_pred_test, cv_res = lazy_train_cv(random_nn, X_train_change, X_test_change, y_train,
                                                                 hidden_layers, lam_path = lambda_path)
            lazy_time = time.time() - t0
            lazy_train_loss = nn.MSELoss()(lazy_pred_train, y_train).item()
            lazy_test_loss = nn.MSELoss()(lazy_pred_test, y_test).item()
            lazy_vi = nn.MSELoss()(lazy_pred_test, y_test).item() - nn.MSELoss()(full_pred_test, y_test).item()
            results.append([varr, 'lazy_random', lazy_time, lazy_vi, lazy_train_loss, lazy_test_loss])

        # RETRAIN
        if do_retrain:
            t0 = time.time()
            retrain_pred_train, retrain_pred_test = retrain(p, hidden_layers, j, X_train_change, y_train, X_test_change, tol=tol)
            retrain_time = time.time() - t0
            vi_retrain = nn.MSELoss()(retrain_pred_test, y_test).item() - nn.MSELoss()(y_test, full_pred_test).item()
            loss_rt_test = nn.MSELoss()(retrain_pred_test, y_test).item()
            loss_rt_train = nn.MSELoss()(retrain_pred_train, y_train).item()

            # variance
            eps_j = ((y_test - retrain_pred_test) ** 2).detach().numpy().reshape(1, -1)
            eps_full = ((y_test - full_pred_test) ** 2).detach().numpy().reshape(1, -1)
            se = np.sqrt(np.var(eps_j - eps_full)/y_test.shape[0])

            results.append([varr, 'retrain', retrain_time, vi_retrain, loss_rt_train, loss_rt_test, se])

        # Early stopping
        if early_stop:
            t0 = time.time()
            retrain_pred_train, retrain_pred_test = fake_retrain(p, full_nn, hidden_layers, j, X_train_change, y_train,
                                                                 X_test_change, tol=tol, max_epochs=max_epochs)
            retrain_time = time.time() - t0
            vi_retrain = nn.MSELoss()(retrain_pred_test, y_test).item() - nn.MSELoss()(y_test, full_pred_test).item()
            loss_rt_test = nn.MSELoss()(retrain_pred_test, y_test).item()
            loss_rt_train = nn.MSELoss()(retrain_pred_train, y_train).item()

            # variance
            eps_j = ((y_test - retrain_pred_test) ** 2).detach().numpy().reshape(1, -1)
            eps_full = ((y_test - full_pred_test) ** 2).detach().numpy().reshape(1, -1)
            se = np.sqrt(np.var(eps_j - eps_full)/y_test.shape[0])

            results.append([varr, 'early_stopping', retrain_time, vi_retrain, loss_rt_train, loss_rt_test, se])

        # LINEAR RETRAIN
        if include_linear:
            t0 = time.time()
            lmj = LinearRegression()
            lmj.fit(X_train_change.detach().numpy(), y_train.detach().numpy())
            #retrain_pred_train, retrain_pred_test = retrain(p, hidden_layers, j, X_train_change, y_train, X_test_change, tol=tol)
            lin_time = time.time() - t0
            vi_linear = mse(lmj.predict(X_test_change.detach().numpy()), y_test.detach().numpy()) - mse(lm.predict(X_test.detach().numpy()), y_test.detach().numpy())
            loss_rt_test = mse(lmj.predict(X_test_change.detach().numpy()), y_test.detach().numpy())
            loss_rt_train = mse(lmj.predict(X_train_change.detach().numpy()), y_train.detach().numpy())
            results.append([varr, 'ols', lin_time, vi_linear, loss_rt_train, loss_rt_test])

        # LINEAR RETRAIN
        if include_rf:
            t0 = time.time()
            rfj = RandomForestRegressor()
            rfj.fit(X_train_change.detach().numpy(), y_train.detach().numpy())
            #retrain_pred_train, retrain_pred_test = retrain(p, hidden_layers, j, X_train_change, y_train, X_test_change, tol=tol)
            lin_time = time.time() - t0
            vi_linear = mse(rfj.predict(X_test_change.detach().numpy()), y_test.detach().numpy()) - mse(rf.predict(X_test.detach().numpy()), y_test.detach().numpy())
            loss_rt_test = mse(rfj.predict(X_test_change.detach().numpy()), y_test.detach().numpy())
            loss_rt_train = mse(rfj.predict(X_train_change.detach().numpy()), y_train.detach().numpy())
            results.append([varr, 'rf', lin_time, vi_linear, loss_rt_train, loss_rt_test])

    df = pd.DataFrame(results, columns=['variable', 'method', 'time', 'vi', 'train_loss', 'test_loss', 'se'])
    return df

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

def boxplot_2d(x,y, ax, co='g', whis=1.5, method=''):
    xlimits = [np.percentile(x, q) for q in (25, 50, 75)]
    ylimits = [np.percentile(y, q) for q in (25, 50, 75)]

    ##the box
    box = Rectangle(
        (xlimits[0],ylimits[0]),
        (xlimits[2]-xlimits[0]),
        (ylimits[2]-ylimits[0]),
        ec = co,
        color = co,
        zorder=0
    )
    ax.add_patch(box)

    ##the x median
    vline = Line2D(
        [xlimits[1],xlimits[1]],[ylimits[0],ylimits[2]],
        color=co,
        zorder=1
    )
    ax.add_line(vline)

    ##the y median
    hline = Line2D(
        [xlimits[0],xlimits[2]],[ylimits[1],ylimits[1]],
        color=co,
        zorder=1
    )
    ax.add_line(hline)

    ##the central point
    ax.plot([xlimits[1]],[ylimits[1]], color=co, marker='o', label=method)

    ##the x-whisker
    ##defined as in matplotlib boxplot:
    ##As a float, determines the reach of the whiskers to the beyond the
    ##first and third quartiles. In other words, where IQR is the
    ##interquartile range (Q3-Q1), the upper whisker will extend to
    ##last datum less than Q3 + whis*IQR). Similarly, the lower whisker
    ####will extend to the first datum greater than Q1 - whis*IQR. Beyond
    ##the whiskers, data are considered outliers and are plotted as
    ##individual points. Set this to an unreasonably high value to force
    ##the whiskers to show the min and max values. Alternatively, set this
    ##to an ascending sequence of percentile (e.g., [5, 95]) to set the
    ##whiskers at specific percentiles of the data. Finally, whis can
    ##be the string 'range' to force the whiskers to the min and max of
    ##the data.
    iqr = xlimits[2]-xlimits[0]

    ##left
    left = np.min(x[x > xlimits[0]-whis*iqr])
    whisker_line = Line2D(
        [left, xlimits[0]], [ylimits[1],ylimits[1]],
        color = co,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [left, left], [ylimits[0],ylimits[2]],
        color = co,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##right
    right = np.max(x[x < xlimits[2]+whis*iqr])
    whisker_line = Line2D(
        [right, xlimits[2]], [ylimits[1],ylimits[1]],
        color = co,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [right, right], [ylimits[0],ylimits[2]],
        color = co,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##the y-whisker
    iqr = ylimits[2]-ylimits[0]

    ##bottom
    bottom = np.min(y[y > ylimits[0]-whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [bottom, ylimits[0]],
        color = co,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [bottom, bottom],
        color = co,
        zorder = 1
    )
    ax.add_line(whisker_bar)

    ##top
    top = np.max(y[y < ylimits[2]+whis*iqr])
    whisker_line = Line2D(
        [xlimits[1],xlimits[1]], [top, ylimits[2]],
        color = co,
        zorder = 1
    )
    ax.add_line(whisker_line)
    whisker_bar = Line2D(
        [xlimits[0],xlimits[2]], [top, top],
        color = co,
        zorder = 1
    )