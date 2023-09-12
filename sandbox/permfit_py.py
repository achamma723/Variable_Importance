import os
import torch
from sklearn.datasets import make_regression
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import KFold
from .deep_model import _ensemble_dnnet
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def permfit(X_train, y_train, X_test=None, y_test=None, prob_type='regression', k_fold=0,
            split_perc=0.2, random_state=2021, n_ensemble=10, batch_size=1024, n_perm=100,
            res=True, save_file="Best_model_knock", verbose=1):
    """ Run the PermFit implementation

        this function 
    Args:
        X_train (numpy.array): The matrix of predictors to train/validate
        y_train (numpy.array): The response vector to train/validate
        X_test (numpy.array, optional): If given, The matrix of predictors
        to test. Defaults to None.
        y_test (numpy.array, optional): If given, The response vector to
        test. Defaults to None.
        prob_type (str, optional): A classification or regression
        probem. Defaults to 'regression'.
        k_fold (int, optional): The number of folds for the cross
        validaton. Defaults to 0.
        split_perc (float, optional): If k_fold==0, The percentage of
        the split to train/validate portions. Defaults to 0.2.
        random_state (int, optional): Fixing the seed of the random
        generator. Defaults to 2021.
        n_ensemble (int, optional): The number of DNNs to be fit.
        Defaults to 1.
        batch_size (int, optional): The number of samples per batch
        for test prediction. Defaults to 1024.
        n_perm (int, optional): The number of permutations for each
        column. Defaults to 100.
        res (bool, optional): If True, it will return the dictionary
        of results. Defaults to True.
        save_file (str, optional): Save file for Best pytorch model.
        Defaults to "Best_model".
        verbose (int, optional): If > 1, the progress bar will be
        printed. Defaults to 1.

    Returns:
        results (dict): [description]
    """
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()

    n, _, p = X_train.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if prob_type == 'classification':
        # Converting target values to corresponding integer values
        ind_unique = np.unique(y_train)
        dict_target = dict(zip(ind_unique, range(len(ind_unique))))
        y_train = np.array([dict_target[el]
                            for el in list(y_train)]).reshape(-1, 1)
    else:
        y_train = np.array(y_train).reshape(-1, 1)
    # No cross validation, dividing the samples into train/validate portions
    if k_fold == 0:
        if X_test is None:
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                                test_size=split_perc)
        else:
            y_test = np.array(y_test).reshape(-1, 1)
        result_gradients = _ensemble_dnnet(X_train, y_train, X_test, y_test,
                                           n_ensemble=n_ensemble, prob_type=prob_type,
                                           save_file=save_file, verbose=verbose)

        os.remove(save_file + '.pth')
        print("Done Gradients List!")

        return result_gradients

    else:
        valid_ind = []
        kf = KFold(n_splits=2, random_state=random_state, shuffle=True)
        p2_score = np.empty((n_perm, n, p))
        i = 0
        for train_index, test_index in kf.split(X):
            print(f"Fold: {i+1}")
            i += 1
            valid_ind.append((train_index, test_index))
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            current_score = permfit(X_train, y_train, X_test, y_test, prob_type,
                                    k_fold=0, split_perc=split_perc, random_state=random_state,
                                    n_ensemble=n_ensemble, batch_size=batch_size, n_perm=n_perm,
                                    res=False)
            p2_score[:, test_index, :] = current_score

        results = {}
        results['importance'] = np.mean(np.mean(p2_score, axis=0), axis=0)
        results['std'] = np.std(np.mean(p2_score, axis=0),
                                axis=0) / np.sqrt(len(y_test)-1)
        results['pval'] = 1 - norm.cdf(results['importance'] / results['std'])
        results['validation_ind'] = valid_ind
        return results
