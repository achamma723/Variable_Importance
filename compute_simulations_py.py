import argparse
import pickle
import time

import numpy as np
import pandas as pd
from BBI import BlockBasedImportance
from joblib import Parallel, delayed
from scipy.linalg import cholesky
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder

parser = argparse.ArgumentParser()
# Number of samples
parser.add_argument("--n", type=int, default=1000)
# Number of variables
parser.add_argument("--p", type=int, default=50)
# Number of significant variables
parser.add_argument("--nsig", type=int, default=20)
# Number of blocks
parser.add_argument("--nblocks", type=int, default=10)
# Intra Correlation
parser.add_argument("--intra", type=float, default=0.8)
# CPI or PI
parser.add_argument("--conditional", type=int, default=1)
# Starting iteration
parser.add_argument("--f", type=int, default=1)
# Stepsize
parser.add_argument("--s", type=int, default=100)
# Number of jobs
parser.add_argument("--njobs", type=int, default=1)
args, _ = parser.parse_known_args()


def generate_cor_blocks(p, inter_cor, intra_cor, n_blocks):
    vars_per_grp = int(p / n_blocks)
    cor_mat = np.zeros((p, p))
    cor_mat.fill(inter_cor)
    for i in range(n_blocks):
        cor_mat[
            (i * vars_per_grp) : ((i + 1) * vars_per_grp),
            (i * vars_per_grp) : ((i + 1) * vars_per_grp),
        ] = intra_cor
    np.fill_diagonal(cor_mat, 1)
    return cor_mat


def cov2corr(A):
    """
    covariance matrix to correlation matrix.
    """
    d = np.sqrt(A.diagonal())
    A = ((A.T / d).T) / d
    return A


def compute_simulations(
    seed,
    filename=None,
    resample=False,
    n_signal=20,
    list_nominal=None,
    snr=4,
    conditional=False,
    n=1000,
    p=50,
    inter_cor=0.8,
    intra_cor=0,
    n_blocks=10,
    prob_type="regression",
):
    start = time.time()
    rng = np.random.RandomState(seed)

    if filename is not None:
        data = pd.read_csv(f"{filename}.csv")
        with open("groups_UKBB.pickle", "rb") as file:
            groups = pickle.load(file)
        # Remove the connectivity group
        data = data.drop(columns=groups["connectivity"])

    else:
        cor_mat = generate_cor_blocks(p, inter_cor, intra_cor, n_blocks)
        x = norm.rvs(size=(p, n), random_state=seed)
        c = cholesky(cor_mat, lower=True)
        data = pd.DataFrame(np.dot(c, x).T, columns=[str(i) for i in np.arange(p)])

    data_enc = data.copy()

    # Randomly draw n_signal predictors which are defined as signal predictors
    indices_var = list(
        rng.choice(range(data_enc.shape[1]), size=n_signal, replace=False)
    )

    # Reorder data matrix so that first n_signal predictors are the signal predictors
    ## List of remaining indices
    indices_rem = [ind for ind in range(data_enc.shape[1]) if ind not in indices_var]
    total_indices = indices_var + indices_rem
    # Before including the non-linear effects
    data_enc = data_enc.iloc[:, total_indices]
    data_enc_a = data_enc.iloc[:, np.arange(n_signal)]

    enc_dict = {}
    total_labels_enc = []
    if len(list_nominal["nominal"]) > 0:
        for col_encode in list_nominal["nominal"]:
            if col_encode in data_enc_a.columns:
                enc = OneHotEncoder(handle_unknown="ignore")
                enc.fit(data_enc_a[[col_encode]])
                labeled_cols = [
                    enc.feature_names_in_[0] + "_" + str(enc.categories_[0][j])
                    for j in range(len(enc.categories_[0]))
                ]
                total_labels_enc += labeled_cols
                hot_cols = pd.DataFrame(
                    enc.transform(data_enc_a[[col_encode]]).toarray(),
                    dtype="int32",
                    columns=labeled_cols,
                )
                data_enc_a = data_enc_a.drop(columns=[col_encode])
                data_enc_a = pd.concat([data_enc_a, hot_cols], axis=1)
                enc_dict[col_encode] = enc

    count_pairs = 0
    tmp_comb = data_enc_a.shape[1]

    # Determine beta coefficients
    effectset = [-0.5, -1, -2, -3, 0.5, 1, 2, 3]
    beta = rng.choice(effectset, size=(tmp_comb + count_pairs), replace=True)

    # Generate response
    ## The product of the signal predictors with the beta coefficients
    prod_signal = np.dot(data_enc_a, beta)

    if prob_type != "classification":
        sigma_noise = np.linalg.norm(prod_signal, ord=2) / (
            snr * np.sqrt(data_enc_a.shape[0])
        )
        y = prod_signal + sigma_noise * rng.normal(size=prod_signal.shape[0])
    else:
        from scipy.special import expit

        y = rng.binomial(1, size=data_enc_a.shape[0], p=expit(prod_signal)).astype(str)
        while y.tolist().count("0") < 0.1 * data_enc_a.shape[0]:
            y = rng.binomial(1, size=data_enc_a.shape[0], p=expit(prod_signal)).astype(
                str
            )

    bbi_model = BlockBasedImportance(
        estimator=None,
        do_hyper=True,
        importance_estimator="Mod_RF",
        dict_hyper=None,
        conditional=conditional,
        n_perm=100,
        n_jobs=1,
        prob_type=prob_type,
        k_fold=2,
        list_nominal=list_nominal,
    )
    bbi_model.fit(data_enc, y)
    res = bbi_model.compute_importance()

    elapsed = time.time() - start
    method = "Permfit-DNN" if not conditional else "CPI-DNN"

    f_res = {}
    f_res["method"] = [method] * len(list(data.columns))
    f_res["score"] = res["score_R2"]
    f_res["elapsed"] = [elapsed] * len(list(data.columns))
    f_res["correlation"] = [intra_cor] * len(list(data.columns))
    f_res["n_samples"] = [data.shape[0]] * len(list(data.columns))
    f_res["prob_data"] = ["regression_combine"] * len(list(data.columns))
    f_res["iteration"] = seed

    f_res = pd.DataFrame(f_res)

    # Add importance, std & p-val
    df_imp = pd.DataFrame(res["importance"], columns=["importance"])
    df_pval = pd.DataFrame(res["pval"], columns=["p_value"])

    f_res = pd.concat([f_res, df_imp, df_pval], axis=1)
    return f_res


filename = None
list_nominal = {"nominal": [], "ordinal": [], "binary": []}

# UKBB case
# filename = "data/ukbb_data_age_no_hot_encoding"
# list_nominal["nominal"] = list(
#     pd.read_csv(filename + "_nominal_columns.csv")["x"]
# )
# list_nominal["ordinal"] = list(
#     pd.read_csv(filename + "_ordinal_columns.csv")["x"]
# )
# list_nominal["binary"] = list(
#     pd.read_csv(filename + "_binary_columns.csv")["x"]
# )

n = args.n
p = args.p
conditional = True if args.conditional == 1 else False
f_d = args.f
l_d = args.f + args.s

print(f"Range: {f_d} to {l_d-1}")

parallel = Parallel(n_jobs=args.njobs, verbose=1)
final_res = parallel(
    delayed(compute_simulations)(
        seed=seed,
        filename=filename,
        resample=False,
        n_signal=args.nsig,
        list_nominal=list_nominal,
        snr=4,
        conditional=conditional,
        n=n,
        p=p,
        inter_cor=0,
        intra_cor=args.intra,
        n_blocks=args.nblocks,
        prob_type="regression",
    )
    for seed in np.arange(args.f, l_d)
)

final_res = pd.concat(final_res).reset_index(drop=True)
if filename is not None:
    final_res.to_csv(
        f"results/results_csv/simulation_results_blocks_100_UKBB_single_{args.f}::{l_d-1}_{'cpi' if conditional else 'permfit'}.csv",
        index=False,
    )
else:
    final_res.to_csv(
        f"results/results_csv/simulation_results_blocks_100_n_{n}_p_{p}_{args.f}::{l_d-1}_{'cpi' if conditional else 'permfit'}.csv",
        index=False,
    )
