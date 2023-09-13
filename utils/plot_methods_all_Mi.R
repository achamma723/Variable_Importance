# load relevant R package
require(data.table)
suppressMessages({
  library("ROCR")
  library("ggplot2")
  library("dplyr")
  library("scales")
})


compute_auc <- function(obj,
                        nb_relevant = 20,
                        ...) {
  p_vals <- as.numeric(-obj$p_value)

  ground_tr <- rep(0, length(p_vals))
  ground_tr[1] <- 1
  ground_tr[11] <- 1
  ground_tr[21] <- 1
  ground_tr[31] <- 1
  ground_tr[41] <- 1

  return(performance(
    prediction(
      p_vals,
      ground_tr
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
  return(mean(obj$p_value[-c(1, 11, 21, 31, 41)] < upper_bound))
}


compute_power <- function(obj,
                          nb_relevant = 20,
                          upper_bound = 0.05) {
  if (length(obj$p_value[!is.na(obj$p_value)]) == 0) {
    return(NULL)
  }
return(mean(obj$p_value[c(1, 11, 21, 31, 41)] < upper_bound))
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
                        mediane_bool = NULL) {
  df <- fread(source_file)

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
      iteration
    )
  ]
  write.csv(res, file=file.path(
      "results/results_csv",
      paste0(
        output_file, ".csv"
      )
    ))
}