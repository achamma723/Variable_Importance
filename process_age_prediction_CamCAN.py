import numpy as np
import pandas as pd
from age_prediction_by_source import get_data
from BBI import BlockBasedImportance


def cov2corr(A):
    """
    covariance matrix to correlation matrix.
    """
    d = np.sqrt(A.diagonal())
    A = ((A.T / d).T) / d
    return A


list_inputs = ["FREQ"]
lbl_inputs = "_".join(list_inputs)

# Input data
X, y, groups = get_data(list_inputs)

# Configuration
conditional = True
k_fold_bbi = 10
n_jobs = 100
list_nominal = None
group_stacking = True
n_perm = 100
random_state = 2023
groups = None

# Model initialization
bbi_model = BlockBasedImportance(
    importance_estimator="Mod_RF",
    prob_type="regression",
    conditional=conditional,
    k_fold=k_fold_bbi,
    n_jobs=n_jobs,
    list_nominal=list_nominal,
    group_stacking=group_stacking,
    groups=groups,
    random_state=random_state,
    n_perm=n_perm,
    verbose=10,
    com_imp=True,
)

bbi_model.fit(X, y)
results = bbi_model.compute_importance()

if groups is None:
    list_var = list(X.columns)
else:
    list_var = list(groups)

res = pd.DataFrame(
    {
        "method": ["CPI-DNN"] * len(list_var),
        "variable": list_var,
        "importance": results["importance"][:, 0],
        "p_value": results["pval"][:, 0],
        "score_R2": results["score_R2"],
        "score_MAE": results["score_MAE"],
    }
)

res.to_csv(
    f"Result_{'grps' if groups is not None else 'single'}_{lbl_inputs}_all_imp_outer_{k_fold_bbi}_inner.csv",
    index=False,
)
