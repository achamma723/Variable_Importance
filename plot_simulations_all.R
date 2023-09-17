suppressMessages({
  source("utils/plot_methods_all.R")
  library("tools")
  library("data.table")
  library("ggpubr")
  library("scales")
})

file_path <- paste0(getwd(), "/results/results_csv/")
filename <- paste0(file_path, "simulation_results_blocks_100_allMethods_pred_final")
nb_relevant <- 20
N_CPU <- 100

list_func <- c(
  "Marg",
  "d0CRT",
  "Permfit-DNN",
  "CPI-DNN",
  "CPI-RF",
  "lazyvi",
  "cpi_knockoff",
  "loco",
  # "LOCO-DNN",
  "Strobl"
  # "Shap",
  # "SAGE",
  # "MDI",
  # "BART"
  # "Knockoff_bart",
  # "Knockoff_lasso",
  # "Knockoff_deep",
)

run_plot_auc <- TRUE
run_plot_type1error <- TRUE
run_plot_power <- TRUE
run_time <- TRUE
run_plot_combine <- FALSE

run_all_methods <- FALSE
with_pval <- TRUE

filename_lbl_auc <- strsplit(filename, "_results_")[[1]][2]
filename_lbl <- strsplit(filename, "_results_")[[1]][2]
if (run_all_methods){
  if (with_pval==TRUE){
    filename_lbl_auc <- paste0(filename_lbl_auc, '_withPval')
  }else{
  filename_lbl_auc <- paste0(filename_lbl_auc, '_withoutPval')
  }
}


if (run_plot_auc) {
  plot_method(paste0(filename, ".csv"),
              paste0("AUC_", filename_lbl_auc),
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
              paste0("type1error_", filename_lbl),
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
              paste0("power_", filename_lbl),
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
            paste0("time_bars_", filename_lbl),
            list_func = list_func,
            N_CPU = N_CPU
  )
}


if (run_plot_combine) {
  plot_method(paste0(filename, ".csv"),
              paste0("combine_", filename_lbl),
              nb_relevant = nb_relevant,
              cor_coef = 0.8,
              title = "AUC",
              list_func = list_func,
              mediane_bool = TRUE
  )
}
