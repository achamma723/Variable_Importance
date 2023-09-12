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


compute_power <- function(obj,
                          nb_relevant = 20,
                          upper_bound = 0.05) {
  if (length(obj$p_value[!is.na(obj$p_value)]) == 0) {
    return(NULL)
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


plot_func <- function(obj,
                      not_time = TRUE,
                      output_file = "default",
                      yintercept = 0.5,
                      title = "AUC",
                      list_func = NULL) {
  obj$method <- factor(obj$method,
    levels = list_func)

  obj$prob_data <- factor(obj$prob_data,
    levels = c(
      "classification",
      "regression",
      "regression_relu",
      "regression_combine",
      "regression_product"
    )
  )

  p <- ggplot(
    data = data.frame(obj),
    aes(
      x = n_samples,
      y = V1,
      group = method
    )
  ) +
    geom_boxplot(
      alpha = 0.4, aes(
        fill = method,
        color = method
      ),
      outlier.size = 3
    ) +
    guides(color = "none") +
    ylab("AUC Score") +
    xlab("Number of samples") +
    rotate() +
    ggtitle(title) +
    geom_hline(
      yintercept = yintercept,
      color = "red"
    )
  saveRDS(p, file.path(
    "results/plot_all",
    paste0(
      output_file, obj$n_samples, "_sim_",
      ".rds"
    )
  ))
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
  output_file_1 = "AUC_blocks_100_dnn_dnn_py_perm_100--1000"
  write.csv(res1, file=file.path(
    "results/results_csv",
    paste0(
      output_file_1, ".csv"
    )
  ))
 output_file_2 = "TypeIerror_blocks_100_dnn_dnn_py_perm_100--1000"
  write.csv(res2, file=file.path(
      "results/results_csv",
      paste0(
        output_file_2, ".csv"
      )
    ))
  print("Done")
  stop()
  res1$prob_data <- factor(res1$prob_data,
    levels = c(
      "classification",
      "regression",
      "regression_relu",
      "regression_combine",
      "regression_product"
    )
    )

  res2$prob_data <- factor(res2$prob_data,
    levels = c(
      "classification",
      "regression",
      "regression_relu",
      "regression_combine",
      "regression_product"
    )
    )

  p1 <- ggplot(
    data = res1,
    aes(
      x = n_samples,
      y = V2
    )
  )

  p1 <- p1 + theme_light(base_size = 60) +
            theme(
                axis.title.y = element_blank(),
                legend.key.size = unit(4, "cm"),
                # legend.position = c(0.725, 0.125),
                legend.position = "bottom",
                legend.justification = c("center", "bottom"),
                legend.text = element_text(colour="black",    
                                          face="bold"),
                # legend.background = element_rect(fill="grey"),
                legend.title = element_blank(),
                legend.box.just = "left",
                legend.margin = margin(6, 6, 6, 6),
                strip.text.x = element_text(size = 50),
                plot.title = element_text(
                hjust = 0,
                vjust = 0,
                face = "bold"
                )
            ) +
            ggtitle("AUC score (medianes) & Type I error (boxplots)") +
            xlab("Number of samples") +
            geom_line(aes(color = method),
                      size = 3,
                      position = position_dodge(width=125)) +
            geom_errorbar(aes(ymin = mean - sd,
                              ymax = mean + sd,
                              color = method),
                          size = 3,
                          width = 100,
                          position = position_dodge(width=125)
                          ) +
            geom_point(aes(color = method),
                          size = 10,
                          position = position_dodge(width=125)) +
            geom_hline(yintercept = 0.05,
                       color = "black"
                      ) +
            geom_hline(yintercept = 0.5,
                       color = "green"
                      ) +
            geom_boxplot(data = res2,
                          alpha = 0.4, aes(
                          group = interaction(n_samples, method),
                          fill = method,
                          color = method),
                          outlier.shape = NA,
                          coef = 0,
                          width = 85) +
            coord_cartesian(ylim = c(0.0, 1.0)) +
            scale_fill_discrete(labels=c("CPI-DNN", "Permfit-DNN")) +
            scale_colour_discrete(labels=c("CPI-DNN", "Permfit-DNN")) +
            facet_wrap(~ prob_data,
            ncol = 5,
            nrow = 1,
            strip.position = c("top"),
            scales = "free_y",
            labeller = labeller(prob_data =
            c("classification" = "Classification",
              "regression" = "Plain linear",
              "regression_relu" = "Regression \n with Relu",
              "regression_combine" = "Main effects \n and Interactions",
              "regression_product" = "Interactions only"))
            )

  ggsave(file.path(
        "results/plot_all",
        paste0(
          output_file, ".pdf"
        )
    ),
        p1,
        width = 45,
        height = 15
  )

  unlink("Rplots.pdf")
  graphics.off()
}