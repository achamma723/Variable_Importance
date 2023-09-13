# load relevant R package
require(data.table)
suppressMessages({
  library("ROCR")
  library("ggplot2")
  library("dplyr")
  library("stringr")
})


compute_auc <- function(obj,
                        nb_relevant = 20,
                        ...) {
  if((is.na(obj$p_value[[1]])) | (obj$method == "lazyvi")) {

  imp_vals <- as.numeric(obj$importance)
  if (obj$method %in% c("Marg", "Ale", "Shap", "SAGE"))
    imp_vals <- abs(imp_vals)
  }
  else {
    imp_vals <- as.numeric(-obj$p_value)
  }
  return(performance(
    prediction(
      imp_vals,
      rep(c(1, 0), c(nb_relevant, length(imp_vals) - nb_relevant))
    ),
    "auc"
  )@y.values[[1]])
}


compute_pval <- function(obj,
                         nb_relevant = 20,
                         upper_bound = 0.05) {
  if (length(obj$p_value[!is.na(obj$p_value)]) == 0) {
    return(NULL)
  }
  if (obj$method == "lazyvi") {
    return(mean(obj$p_value[(nb_relevant + 1):
  length(obj$p_value)]))
  }
  return(mean(obj$p_value[(nb_relevant + 1):
  length(obj$p_value)] < upper_bound))
}


compute_power <- function(obj,
                          nb_relevant = 20,
                          upper_bound = 0.05) {
  if (length(obj$p_value[!is.na(obj$p_value)]) == 0) {
    return(NULL)
  }
  if (obj$method == "lazyvi") {
    return(mean(obj$p_value[1:nb_relevant]))
  }
  return(mean(obj$p_value[1:nb_relevant] < upper_bound))
}


compute_pred <- function(obj,
                         nb_relevant = 20,
                         upper_bound = 0.05) {
  if (length(obj$score[!is.na(obj$score)]) == 0) {
    return(NULL)
  }
  return (obj$score[1])
}


plot_time <- function(source_file,
                      output_file,
                      list_func = NULL,
                      N_CPU = 10) {
  df <- fread(source_file)

  res <- df[,
    mean(elapsed),
    by = .(
      n_samples,
      method,
      iteration,
      prob_data
    )
  ]

  res <- res[,
    sum(V1) / N_CPU,
    by = .(
      n_samples,
      method
    )
  ]
  write.csv(res, file=file.path(
      "results/results_csv",
      paste0(
        output_file, ".csv"
      )
    ))
}


plot_method <- function(source_file,
                        output_file,
                        func = NULL,
                        cor_coef = 0.5,
                        nb_relevant = 20,
                        upper_bound = 0.05,
                        title = "AUC",
                        list_func = NULL,
                        mediane_bool = FALSE) {
  df <- fread(source_file)
  df <- df[df$method %in% list_func]
  df <- cbind(df, prob_type = str_split_fixed(df$prob_data, "_", 2)[, 1])
  yintercept <- ifelse(
    as.character(substitute(func)) == "compute_auc", 0.5, upper_bound
  )

  res <- df[,
    func(c(.BY, .SD),
      nb_relevant = nb_relevant,
      upper_bound = upper_bound
    ),
    by = .(
      method,
      correlation,
      n_samples,
      prob_data,
      iteration,
      prob_type
    )
  ]

  write.csv(res, file=file.path(
      "results/results_csv",
      paste0(
        output_file, ".csv"
      )
    ))
}