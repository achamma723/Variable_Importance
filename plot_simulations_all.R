suppressMessages({
  source("utils/plot_methods_all.R")
  library("tools")
  library("data.table")
  library("ggpubr")
  library("scales")
})

file_path <- paste0("results/")
filename <- "simulation_results_blocks_100_allMethods_pred_imp_final"
nb_relevant <- 20
N_CPU <- 100

list_func <- c(
  "Marg",
  "Knockoff_bart",
  "Knockoff_lasso",
  "Shap",
  "SAGE",
  "MDI",
  "d0CRT",
  "BART",
  "Knockoff_deep",
  # "Knockoff_deepall_mlp",
  # "Knockoff_deepall_single_mlp",
  # "Knockoff_deepall_mlp_new",
  # "Knockoff_deep_single_mlp_test",
  # "Knockoff_deep_mlp_new",
  # "Permfit-DNN_5",
  # "CPI-DNN_5",
  "Permfit-DNN",
  "CPI-DNN",
  # "CPI-DNN_Mod",
  # "CPI-DNN_Stack",
  # "CPI-DNN_noStack",
  # "gpfi",
  # "gopfi",
  # "dgi",
  # "goi",
  "lazyvi",
  # "vimp",
  "CPI-RF",
  "Strobl",
  "cpi_knockoff",
  "loco"
  # "Knockoff_path",
  #    "Knockoff_lasso",
  #    "Ale",
  #    "HoldOut_nr",
  #    "Altmann",
  #    "d0CRT_scaled",
  #    "Bart_py",
  #    "GRF_regression",
  #    "GRF_quantile",
)

run_plot_auc <- TRUE
run_plot_type1error <- TRUE
run_plot_power <- TRUE
run_time <- TRUE
run_plot_pred <- FALSE
run_plot_fdr <- FALSE
run_plot_combine <- FALSE


if (run_plot_auc) {
  plot_method(paste0(filename, ".csv"),
              "AUC_blocks_100_allMethods_pred_imp_final",
              compute_auc,
              nb_relevant = nb_relevant,
              cor_coef = 0.8,
              title = "AUC",
              list_func = list_func,
              mediane_bool = TRUE
  )
}


if (run_plot_type1error) {
  plot_method(paste0(filename, ".csv"),
              "type1error_blocks_100_allMethods_pred_imp_final",
              compute_pval,
              nb_relevant = nb_relevant,
              upper_bound = 0.05,
              cor_coef = 0.8,
              title = "Type I Error",
              list_func = list_func
  )
}


if (run_plot_power) {
  plot_method(paste0(filename, ".csv"),
              "power_blocks_100_allMethods_pred_imp_final",
              compute_power,
              nb_relevant = nb_relevant,
              upper_bound = 0.05,
              cor_coef = 0.8,
              title = "Power",
              list_func = list_func
  )
}


if (run_time) {
  plot_time(paste0(filename, ".csv"),
            "time_bars_blocks_100_allMethods_pred_imp_final",
            list_func = list_func,
            N_CPU = N_CPU
  )
}


if (run_plot_pred) {
  plot_method(paste0(filename, ".csv"),
              "pred_blocks_100_allMethods_pred_imp",
              compute_pred,
              nb_relevant = nb_relevant,
              upper_bound = 0.2,
              cor_coef = 0.8,
              title = "Prediction scores",
              list_func = list_func
  )
}


if (run_plot_combine) {
  plot_method(paste0(filename, ".csv"),
              "combine_blocks_100_dnn_dnn_py_perm_100--1000",
              nb_relevant = nb_relevant,
              cor_coef = 0.8,
              title = "AUC",
              list_func = list_func,
              mediane_bool = TRUE
  )
}


if (run_plot_fdr) {
  plot_method(paste0(filename, ".csv"),
              "fdr_blocks_10_knockoffDeep_single_orig_imp_n",
              compute_fdr,
              nb_relevant = nb_relevant,
              upper_bound = 0.2,
              cor_coef = 0.8,
              title = "FDR Control",
              list_func = list_func
  )
}
