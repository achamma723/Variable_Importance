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


compute_fdr <- function(obj,
                        nb_relevant = 20,
                        upper_bound = 0.1) {
  thresh <- Inf
  if (grepl("Knockoff_deep_", obj$method, fixed = TRUE)) {
    test_score <- obj$importance
    t_mesh <- sort(abs(test_score[test_score != 0]))
    for (t in t_mesh) {
      false_pos <- sum(test_score <= -t)
      selected <- sum(test_score >= t)
      if (((1 + false_pos) / max(selected, 1)) <= upper_bound) {
        thresh <- t
        break
      }
    }
    selected_set <- which(test_score >= thresh)
    fdp <- length(intersect(selected_set,
      c(21:50))) / (max(length(selected_set), 1))
    return(fdp)
  }
}


compute_pred <- function(obj,
                         nb_relevant = 20,
                         upper_bound = 0.05) {
  if (length(obj$score[!is.na(obj$score)]) == 0) {
    return(NULL)
  }
  return (obj$score[1])
}


plot_func <- function(obj,
                      not_time = TRUE,
                      output_file = "default",
                      yintercept = 0.5,
                      title = "AUC",
                      list_func = NULL) {
  obj$method <- factor(obj$method,
    levels = list_func)

  if (not_time) {
    obj$prob_data <- factor(obj$prob_data,
      levels = c(
        "classification",
        "regression",
        "regression_relu",
        "regression_combine",
        "regression_product"
      )
    )
    obj$method <- factor(obj$method,
      levels = c(
        "Marg",
        "Knockoff_lasso",
        "Knockoff_bart",
        "Knockoff_deep",
        "Shap",
        "SAGE",
        "MDI",
        "Strobl",
        "d0CRT",
        "BART",
        "lazyvi",
        "Permfit-DNN",
        "CPI-DNN",
        "CPI-RF",
        "vimp"
      )
    )
    p <- ggplot(
      data = data.frame(obj),
      aes(
        x = method,
        y = V1      
        )
    ) +
      geom_boxplot(
        alpha = 0.4, aes(
          fill = prob_data,
          color = prob_data
        ),
        outlier.size = 3
      ) +
      guides(color = "none") +
      rotate() +
      ggtitle(title)
    saveRDS(p, file.path(
      "results/plot_all",
      paste0(
        output_file, ".rds"
      )
    ))
  }

  else {
    p <- ggplot(
      data = data.frame(obj),
      aes(
        x = method,
        y = V1
      )
    ) +
      geom_bar(
        stat = "identity",
        width = 0.4
      ) +
      scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
        labels = trans_format("log10", math_format(10^.x))) +
      labs(y = "Time") +
      xlab("Method")
    saveRDS(p, file.path(
      "results/plot_all",
      paste0(
        output_file, ".rds"
      )
    ))
  }
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
  print("Done")
  stop()
  res[,
    plot_func(c(.BY, .SD),
      FALSE,
      output_file = output_file,
      list_func = list_func
    ),
    by = .(n_samples)
  ]
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
  print("Done")
  stop()

  res[,
    plot_func(
      c(.BY, .SD),
      TRUE,
      output_file,
      yintercept,
      title,
      list_func
    ),
    by = .(
      n_samples
    )
  ]

  graphics.off()
}