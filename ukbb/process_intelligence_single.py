import pandas as pd
from utils_py import loop_params

# Subset of variables to use
list_cols = [
    "25056-2.0",
    "25057-2.0",
    "25060-2.0",
    "25059-2.0",
    "25061-2.0",
    "25058-2.0",
    "25009-2.0",
    "25025-2.0",
    "25898-2.0",
    "25901-2.0",
    "25904-2.0",
    "25907-2.0",
    "25910-2.0",
    "25913-2.0",
    "25916-2.0",
    "25919-2.0",
    "25396-2.0",
    "25393-2.0",
    "25392-2.0",
    "25394-2.0",
    "25397-2.0",
    "25395-2.0",
    "25156-2.0",
    "25153-2.0",
    "25152-2.0",
    "25154-2.0",
    "25157-2.0",
    "25108-2.0",
    "25105-2.0",
    "25104-2.0",
    "25106-2.0",
    "25109-2.0",
    "25107-2.0",
    "25300-2.0",
    "25297-2.0",
    "25296-2.0",
    "25298-2.0",
    "25301-2.0",
    "25299-2.0",
    "25252-2.0",
    "25251-2.0",
    "25253-2.0",
    "25250-2.0",
    "25248-2.0",
    "25249-2.0",
    "25203-2.0",
    "25205-2.0",
    "25202-2.0",
    "25200-2.0",
    "25201-2.0",
    "25204-2.0",
    "25443-2.0",
    "25445-2.0",
    "25442-2.0",
    "25440-2.0",
    "25441-2.0",
    "25444-2.0",
    "25347-2.0",
    "25349-2.0",
    "25346-2.0",
    "25344-2.0",
    "25345-2.0",
    "25348-2.0",
    "25892-2.0",
    "25001-2.0",
    "25005-2.0",
    "31-0.0",
    "21022-0.0",
    "20016-2.0",
    "20127-0.0",
    "1677-2.0",
    "1787-2.0",
    "6138-2.0",
    "709-2.0",
    "738-2.0",
    "2040-2.0",
    "4526-2.0",
    "4537-2.0",
    "4548-2.0",
    "6142-2.0",
    "4559-2.0",
    "4570-2.0",
    "4581-2.0",
    "670-2.0",
    "699-2.0",
    "767-2.0",
    "806-2.0",
    "816-2.0",
    "3426-2.0",
    "6160-2.0",
    "4631-2.0",
    "6156-2.0",
    "6145-2.0",
    "4598-2.0",
    "4609-2.0",
    "4620-2.0",
]

col_pred = "20016-2.0"
df = pd.read_csv("ukbb_data_intelligence_no_hot_encoding.csv", usecols=list_cols)

nominal_columns = list(
    pd.read_csv("ukbb_data_age_no_hot_encoding_nominal_columns.csv")["x"]
)
ordinal_columns = list(
    pd.read_csv("ukbb_data_age_no_hot_encoding_ordinal_columns.csv")["x"]
)
binary_columns = list(
    pd.read_csv("ukbb_data_age_no_hot_encoding_binary_columns.csv")["x"]
)

nominal_columns = [i for i in nominal_columns if i in list(df)]
ordinal_columns = [i for i in ordinal_columns if i in list(df)]
binary_columns = [i for i in binary_columns if i in list(df)]

list_cat = {
    "nominal": nominal_columns,
    "ordinal": ordinal_columns,
    "binary": binary_columns,
}

X = df.loc[:, df.columns != col_pred]
y = df[col_pred]


# Parameters configuration
param_grid = {"method": ["cpi-dnn"], "group_based": [False], "group_stack": [False]}
grps_merged = {}

# DNN settings
n_jobs = 100
k_fold = 2

loop_params(
    X,
    y,
    list_cat,
    grps=grps_merged,
    param_grid=param_grid,
    n_jobs=n_jobs,
    k_fold=k_fold,
    title_res="Results_variables/Result_intelligence_cpi_single.csv",
)
