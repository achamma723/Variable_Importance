# load relevant R package
require(data.table)
suppressMessages({
  library("ROCR")
  library("ggplot2")
  library("dplyr")
})


compute_auc <- function(obj,
                        nb_relevant = 20,
                        ...) {
  p_vals <- as.numeric(-obj$p_value)

  return(performance(
    prediction(
      p_vals,
      rep(c(1, 0), c(nb_relevant, length(p_vals) - nb_relevant))
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
  return(mean(obj$p_value[(nb_relevant + 1):
  length(obj$p_value)] < upper_bound))
}


plot_method <- function(source_file,
                        output_file,
                        cor_coef = 0.5,
                        nb_relevant = 20,
                        upper_bound = 0.05,
                        title = "AUC",
                        list_func = NULL,
                        mediane_bool = FALSE) {
  df <- fread(source_file)
  df <- df[df$method %in% list_func]

  res1 <- df[,
    compute_auc(c(.BY, .SD),
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

  res1 <- res1[,
    list(V2 = median(V1),
        mean = mean(V1),
        sd = sd(V1)),
    by = .(
      method,
      correlation,
      n_samples,
      prob_data
    )]

  res2 <- df[,
    compute_pval(c(.BY, .SD),
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

  res2 <- res2[,
    list(V2 = V1),
    by = .(
      method,
      correlation,
      n_samples,
      prob_data
    )]

  filename_lbl = strsplit(output_file, "combine_")[[1]][2]
  output_file_1 = paste0("AUC_", filename_lbl)
  write.csv(res1, file=file.path(
    "results/results_csv",
    paste0(
      output_file_1, ".csv"
    )
  ))
 output_file_2 = paste0("TypeIerror_", filename_lbl)
  write.csv(res2, file=file.path(
      "results/results_csv",
      paste0(
        output_file_2, ".csv"
      )
    ))
}