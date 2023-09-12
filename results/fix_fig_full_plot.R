library(ggplot2)
library(ggthemes)
library(plyr)
library(patchwork)
library(grid)
library(gridExtra)
library(tidyverse)
library(scales)

p1 <- readRDS("plot_all/AUC_blocks_100_allMethods_pred_imp_withPval.rds")
p2 <- readRDS("plot_all/AUC_blocks_100_allMethods_pred_imp_withoutPval.rds")
p3 <- readRDS("plot_all/type1error_blocks_100_allMethods_pred_imp.rds")
p4 <- readRDS("plot_all/power_blocks_100_allMethods_pred_imp.rds")
p5 <- readRDS("plot_all/time_bars_blocks_100_allMethods_pred_imp.rds")
p6 <- readRDS("plot_all/pred_blocks_100_allMethods_pred_imp.rds")
p7 <- readRDS("plot_all/fdr_blocks_10_knockoffDeep_dnnPy_with_single_1_n1000_sim_with_corr.rds")

p1 <- p1 + theme_light(base_size = 65) +
      theme(
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank(),
            legend.key.size = unit(2, "cm"),
            legend.title = element_blank(),
            legend.position = "top",
            legend.justification = c(1.0, 0),
            legend.box.just = "left",
            legend.margin = margin(0, 0, 0, 0),
            strip.text.y.left = element_text(size = 50, angle = 0),
            plot.margin = grid::unit(
                  c(10., 10., 0., 10.),
                  "mm"
            )
      ) +
      ggtitle("Auc Score") +
      xlab("Correlation") +
      ylim(0.4, 1.0) +
      scale_fill_discrete(labels = c("Classification",
                                      "Plain linear",
                                      "Regression with Relu",
                                      "Main effects and Interactions",
                                      "Interactions only")) +
      theme(plot.title = element_text(
            hjust = -0.4,
            vjust = 0,
            face = "bold"
      )) +
      geom_hline(
        yintercept = 0.5,
        color = "black"
      ) +
      guides(fill = guide_legend(ncol = 3)) +
      facet_wrap(~ method,
                 ncol = 1,
                 strip.position = c("left"),
                 scales = "free_y",
                 labeller = labeller(
                   method =
                  c("Marg" = "Marginal",
                    "Knockoff_lasso" = "Knockoff-Lasso",
                    "Knockoff_bart" = "Knockoff-Bart",
                    "Shap" = "SHAP",
                    "MDI" = "MDI",
                    "Strobl" = "Conditional-RF",
                    "d0CRT" = "d0CRT",
                    "BART" = "BART",
                    "Knockoff_deep" = "Knockoff-Deep",
                    "Permfit-DNN" = "Permfit-DNN",
                    "CRF-DNN" = "CRF-DNN"
                    ))
                 )

p2 <- p2 + theme_light(base_size = 65) +
      theme(
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank(),
            legend.key.size = unit(2, "cm"),
            legend.title = element_blank(),
            legend.position = "none",
            legend.justification = c(1.0, 0),
            legend.box.just = "left",
            legend.margin = margin(0, 0, 0, 0),
            strip.text.y.left = element_text(size = 50, angle = 0),
            plot.margin = grid::unit(
                  c(0., 10., 10., 10.),
                  "mm"
            )
      ) +
      ggtitle("No statistical guarantees") +
      xlab("Correlation") +
      ylim(0.4, 1.0) +
      scale_fill_discrete(labels = c("Classification",
                                      "Plain linear",
                                      "Regression with Relu",
                                      "Main effects and Interactions",
                                      "Interactions only")) +
      theme(plot.title = element_text(
            hjust = -0.9,
            vjust = 0,
            face = "bold.italic"
      )) +
      geom_hline(
        yintercept = 0.5,
        color = "black"
      ) +
      guides(fill = guide_legend(ncol = 3)) +
      facet_wrap(~ method,
                 ncol = 1,
                 strip.position = c("left"),
                 scales = "free_y",
                 labeller = labeller(
                   method =
                  c("Marg" = "Marginal",
                    "Knockoff_lasso" = "Knockoff-Lasso",
                    "Knockoff_bart" = "Knockoff-Bart",
                    "Shap" = "SHAP",
                    "MDI" = "MDI",
                    "Strobl" = "Conditional-RF",
                    "d0CRT" = "d0CRT",
                    "BART" = "BART",
                    "Knockoff_deep" = "Knockoff-Deep",
                    "Permfit-DNN" = "Permfit-DNN",
                    "CRF-DNN" = "CRF-DNN"
                    ))
                 )

p3 <- p3 + theme_light(base_size = 65) +
      theme(
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank(),
            legend.title = element_blank(),
            legend.position = "none", #c(0.667, 0.6),
            legend.justification = c("left", "bottom"),
            legend.box.just = "left",
            legend.margin = margin(6, 6, 6, 6),
            strip.text.y.left = element_text(size = 50, angle = 0)
      ) +
      ggtitle("Type I error") +
      xlab("Correlation") +
      theme(
            plot.title = element_text(
                  hjust = -1.2,
                  vjust = 0,
                  face = "bold"
            ),
            plot.margin = grid::unit(
                  c(10., 10., 0., 10.),
                  "mm"
            )
      ) +
      geom_hline(
        yintercept = 0.05,
        color = "black"
      ) +
      scale_fill_discrete(labels = c("Permfit-DNN",
                                    "CRF-DNN",
                                    "PyPermfit-DNN")) +
      facet_wrap(~ method,
                 ncol = 1,
                 strip.position = c("left"),
                 scales = "free_y",
                 labeller = labeller(
                   method =
                  c("Marg" = "Marginal",
                    "Knockoff_path" = "KPath",
                    "Knockoff_bart" = "KBart",
                    "Shap" = "SHAP",
                    "MDI" = "MDI",
                    "Strobl" = "Conditional-RF",
                    "d0CRT" = "d0CRT",
                    "BART" = "BART",
                    "Knockoff_deep" = "KDeep",
                    "Permfit-DNN" = "Permfit-DNN",
                    "CRF-DNN" = "CRF-DNN"
                    ))
                 )

p4 <- p4 + theme_light(base_size = 65) +
      theme(
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank(),
            legend.title = element_blank(),
            legend.position = "none",
            legend.justification = c("left", "bottom"),
            legend.box.just = "left",
            legend.margin = margin(0, 0, 0, 0),
            strip.text.y.left = element_text(size = 50, angle = 0)
      ) +
      ggtitle("Power") +
      xlab("Correlation") +
      theme(
            plot.title = element_text(
                  hjust = -0.75,
                  vjust = 0,
                  face = "bold"
            ),
            plot.margin = grid::unit(
                  c(0., 10., 0., 10.),
                  "mm"
            )
      ) +
      guides(fill = guide_legend(reverse = TRUE, ncol = 3)) +
      facet_wrap(~ method,
                 ncol = 1,
                 strip.position = c("left"),
                 scales = "free_y",
                 labeller = labeller(
                   method =
                  c("Marg" = "Marginal",
                    "Knockoff_path" = "KPath",
                    "Knockoff_bart" = "KBart",
                    "Shap" = "SHAP",
                    "MDI" = "MDI",
                    "Strobl" = "Conditional-RF",
                    "d0CRT" = "d0CRT",
                    "BART" = "BART",
                    "Knockoff_deep" = "KDeep",
                    "Permfit-DNN" = "Permfit-DNN",
                    "CRF-DNN" = "CRF-DNN"
                    ))
                  )

p5 <- p5 + theme_light(base_size = 55) +
      ggtitle("Computation time") +
      theme(
            axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            strip.background = element_blank(),
            strip.text.y = element_blank(),
            plot.title = element_text(
                  hjust = 0,
                  vjust = 0,
                  face = "bold"
            ),
            plot.margin = grid::unit(
                  c(0., 10., 10., 10.),
                  "mm"
            )
      ) +
     scale_x_discrete(labels = c("Marginal",
                                 "Knockoff-BART",
                                 "Knockoff-Lasso",
                                 "SHAP",
                                 "MDI",
                                 "Conditional-RF",
                                 "d0CRT",
                                 "BART",
                                 "Knockoff-Deep",
                                 "Permfit-DNN",
                                 "CRF-DNN"))

p6 <- p6 + theme_light(base_size = 65) +
      theme(
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank(),
            legend.key.size = unit(2, "cm"),
            legend.title = element_blank(),
            legend.position = "top",
            legend.justification = c(0.5, 0),
            legend.box.just = "left",
            legend.margin = margin(0, 0, 0, 0),
            strip.text.y.left = element_text(size = 50, angle = 0),
            plot.margin = grid::unit(
                  c(10., 10., 10., 10.),
                  "mm"
            )
      ) +
      ggtitle("Prediction Scores") +
      scale_fill_discrete(labels = c("Classification",
                                      "Plain linear",
                                      "Regression with Relu",
                                      "Main effects and Interactions",
                                      "Interactions only")) +
      theme(plot.title = element_text(
            hjust = 0,
            vjust = 0,
            face = "bold"
      )) +
      guides(fill = guide_legend(ncol = 3)) +
      facet_grid(rows = vars(method),
                 cols = vars(prob_type),
                 scales = "free_y",
                 switch = 'y',
                 labeller = labeller(
                  method =
                  c("Marg" = "Marginal",
                    "Knockoff_lasso" = "Lasso",
                    "MDI" = "Random Forest",
                    "BART" = "BART",
                    "Permfit-DNN" = "DNN"),
                 prob_type =
                 c("classification" = "Classification",
                   "regression" = "Plain linear")
                 ))
ggsave(file.path("plot_all", "test_blocks_100_allMethods_pred_imp_predScores.pdf"),
       p6,
       width = 42,
       height = 25
)

p7 <- p7 + theme_light(base_size = 42) +
      theme(
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            legend.position = "none",
            strip.text.y = element_blank()
      ) +
      ggtitle("FDR Control") + theme(
            plot.title = element_text(
                  hjust = -0.9,
                  vjust = -2,
                  face = "bold"
            ),
            plot.margin = grid::unit(
                  c(0., 0., 0., 0.),
                  "mm"
            )
      ) +
      guides(fill = guide_legend(reverse = TRUE, ncol = 2)) +
      facet_wrap(~ correlation,
                 ncol = 1,
                 strip.position = c("left"),
                 scales = "free_y")

lay <- rbind(
      c(1, 1, 1, 3, 3),
      c(1, 1, 1, 4, 4),
      c(2, 2, 2, 5, 5)
)

p <- grid.arrange(p1, p2, p3, p4, p5, layout_matrix = lay)

ggsave(file.path("plot_all", "test_blocks_100_allMethods_pred_imp.pdf"),
      p,
      width = 42,
      height = 30
)

unlink("Rplots.pdf")