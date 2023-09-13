import csv
import math
import os
import pickle
import random
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras import layers, metrics, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


def _index_predictions(predictions, labels):
    """
    Indexes predictions, a [batch_size, num_classes]-shaped tensor,
    by labels, a [batch_size]-shaped tensor that indicates which
    class each sample should be indexed by.

    Args:
        predictions: A [batch_size, num_classes]-shaped tensor. The input to a model.
        labels: A [batch_size, num_classes]-shaped tensor.
                The tensor used to index predictions, in one-hot encoding form.
    Returns:
        A tensor of shape [batch_size] representing the predictions indexed by the labels.
    """
    current_batch_size = tf.shape(predictions)[0]
    sample_indices = tf.range(current_batch_size)
    sparse_labels = tf.argmax(labels, axis=-1)
    indices_tensor = tf.stack(
        [sample_indices, tf.cast(sparse_labels, tf.int32)], axis=1
    )
    predictions_indexed = tf.gather_nd(predictions, indices_tensor)
    return predictions_indexed


class My_Callback2(tf.keras.callbacks.Callback):
    def __init__(self, X, y, DNN, Feat_Import, epoch_param_current):
        self.pVal = X.shape[1]
        self.epoch_param_current = epoch_param_current
        self.X = X
        self.y = y
        self.DNN = DNN
        self.res_list = []

    def on_epoch_end(self, epoch, logs={}):
        if (epoch + 1) in self.epoch_param_current:
            x3D_allT = tf.convert_to_tensor(self.X, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(x3D_allT)
                predictions = self.DNN(x3D_allT)
                predictions_indexed = _index_predictions(predictions, self.y)
            gradients = tape.gradient(predictions_indexed, x3D_allT)
            # print(gradients)
            temp_g = gradients.numpy()
            avg_all = np.zeros((self.pVal, self.X.shape[2]))
            for row_ind in range(temp_g.shape[1]):
                for col_ind in range(temp_g.shape[2]):
                    avg_all[row_ind, col_ind] = np.mean(temp_g[:, row_ind, col_ind])
            # print(avg_all)
            # exit(0)
            self.res_list.append(np.hstack((avg_all[:, 0], avg_all[:, 1])))


def train_DNN(model, X, y, myCallback, num_epochs, batch_size):
    model.fit(
        X,
        y,
        epochs=num_epochs,
        batch_size=batch_size,
        verbose=1,
        validation_split=0,
        callbacks=[myCallback],
    )
    return model


def getModelLocalEq(
    pVal, Num_knock, coeff1, lr, FILTER=8, KERNEL=25, STRIDE=25, bias=True
):
    input = Input(name="Input1", shape=(pVal, Num_knock + 1))
    local1 = LocallyConnected1D(
        1,
        1,
        use_bias=True,
        kernel_initializer=Constant(value=0.1),
        activation="elu",
        kernel_regularizer=tf.keras.regularizers.l1(coeff1),
    )(input)
    Dropout1 = Dropout(0.1)(local1)
    local2 = LocallyConnected1D(
        FILTER,
        KERNEL,
        strides=STRIDE,
        use_bias=True,
        kernel_initializer="glorot_normal",
        activation="elu",
    )(Dropout1)
    flat = Flatten()(local2)
    dense1 = Dense(
        50, activation="elu", use_bias=True, kernel_initializer="glorot_normal"
    )(flat)
    out_ = Dense(
        1, activation="linear", use_bias=True, kernel_initializer="glorot_normal"
    )(dense1)
    opt = tf.optimizers.Adam(learning_rate=lr)
    model = Model(inputs=input, outputs=out_)
    model.compile(
        loss="mse", optimizer=opt, metrics=[tf.keras.metrics.MeanSquaredError()]
    )
    return model


def orig_imp_fit(
    X,
    y,
    num_epochs=100,
    num_param=2,
    cutoff_point=50,
    batch_size=1024,
    FILTER=8,
    KERNEL=25,
    STRIDE=25,
    Num_knock=1,
    bias=True,
    num_folds=2,
    seed=457,
):
    np.random.seed(seed)
    kf = KFold(n_splits=num_folds, random_state=869, shuffle=True)
    X = np.swapaxes(X, 1, 2)
    # Standard Scaling
    for lay in range(X.shape[2]):
        scaler = StandardScaler()
        X[:, :, lay] = scaler.fit_transform(X[:, :, lay])
    y = np.array(y).reshape(-1, 1)

    # The csv file of parameter values
    Param_Space = pd.read_csv(os.path.dirname(__file__) + "/Param_SKAT2.csv", header=0)

    ######## Zero-Padding #########
    # In order to have enough components for the Stride movement across the matrix
    Orig_dim = X.shape[1]
    res = Orig_dim % STRIDE
    if res != 0:
        new_row = X.shape[0]
        new_col = STRIDE - res
        x_new = np.zeros((new_row, new_col, Num_knock + 1))
        X = np.hstack((X, x_new))
    ###############################

    pVal = X.shape[1]
    AVG_Val_Loss = np.zeros((num_param, num_epochs))
    AVG_Val_mse = np.zeros((num_param, num_epochs))

    for param_counter in range(0, num_param):  # Loop for different parameter set
        lr = Param_Space["Learning_Rate"].iat[param_counter]
        CoeffS = Param_Space["L1_Norm"].iat[param_counter]
        fold_no = 0
        Val_loss_All = []
        Val_mse_All = []

        for train_index, val_index in kf.split(X):  # Loop over the different 5 folds
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            ## learning
            DNN = getModelLocalEq(pVal, Num_knock, CoeffS, lr)
            history = DNN.fit(
                X_train,
                y_train,
                epochs=num_epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                verbose=0,
            )
            Val_loss = history.history[list(history.history.keys())[2]]
            Val_mse = history.history[list(history.history.keys())[3]]
            Val_mse_All.append(Val_mse)
            Val_loss_All.append(Val_loss)
            fold_no = fold_no + 1
        AVG_Val_Loss[param_counter] = np.mean(np.array(Val_loss_All), axis=0)
        AVG_Val_mse[param_counter] = np.mean(np.array(Val_mse_All), axis=0)
    #############

    CoeffS = np.array(Param_Space["L1_Norm"][:num_param])
    temp_coeff = np.tile(CoeffS, (num_epochs, 1)).T  # (num_param, num_epochs)

    Lr = np.array(Param_Space["Learning_Rate"][:num_param])
    temp_Lr = np.tile(Lr, (num_epochs, 1)).T  # (num_param, num_epochs)

    vect_val = AVG_Val_Loss.ravel()  # Validation Loss (num_param * num_epochs, )
    vect_coef = temp_coeff.ravel()  # L1-norm coefficients (num_param * num_epochs, )
    vect_lr = temp_Lr.ravel()  # Learning rate (num_param * num_epochs, )

    # Sort the validation losses after the aggregation of the 5-fold cross validation
    temp_sort_ind = np.argsort(vect_val)
    maximum_loss = np.max(vect_val)
    minimum_loss = np.min(vect_val)

    W_FIs = np.zeros((1, num_param * num_epochs))
    # Weight Feature Importance -> Min-Max scaling while considering the least to be better (MSE error)
    for i in range(num_param * num_epochs):
        W_FIs[0, i] = (maximum_loss - vect_val[i]) / (maximum_loss - minimum_loss)
    W_optimal = W_FIs[0, temp_sort_ind]

    truncated_sort_ind = temp_sort_ind[:cutoff_point]
    vect_opt_lr = vect_lr[temp_sort_ind][:cutoff_point]
    vect_opt_coef = vect_coef[temp_sort_ind][:cutoff_point]

    epoch_trunc = np.zeros(cutoff_point)
    res = (truncated_sort_ind + 1) % num_epochs

    for i in range(0, cutoff_point):
        if res[i] != 0:
            epoch_trunc[i] = res[i]
        else:
            epoch_trunc[i] = num_epochs

    For_FI_callback = np.column_stack(
        (vect_opt_lr, vect_opt_coef, epoch_trunc, W_optimal[:cutoff_point])
    )

    Feat_Import = []
    for i in range(0, num_param):
        lr = Param_Space["Learning_Rate"].iat[i]
        CoeffS = Param_Space["L1_Norm"].iat[i]
        temp = For_FI_callback[
            (For_FI_callback[:, 0] == lr) & (For_FI_callback[:, 1] == CoeffS)
        ]
        sorted_filtered_param = temp[temp[:, 2].argsort()]
        epoch_param_current = sorted_filtered_param[:, 2]
        if len(epoch_param_current) != 0:
            DNN = getModelLocalEq(pVal, Num_knock, CoeffS, lr)
            myCallback = My_Callback2(X, y, DNN, Feat_Import, epoch_param_current)
            trained_DNN = train_DNN(DNN, X, y, myCallback, num_epochs, batch_size)
            Feat_Import += myCallback.res_list

    Temp = np.array(np.transpose(Feat_Import)).dot(np.transpose(For_FI_callback[:, 3]))

    FI_final = Temp / np.sum(W_FIs)
    res = abs(FI_final[: int(len(FI_final) / 2)]) - abs(
        FI_final[int(len(FI_final) / 2) :]
    )
    print(res)
    return res
