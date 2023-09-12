library(ggplot2)
library(ggthemes)
library(plyr)
library(patchwork)
library(grid)
library(gridExtra)
library(tidyverse)
library(scales)

p1 <- readRDS("plot_all/AUC_blocks_100_dnn_dnn_py_perm_100--1000_mediane.rds")
p2 <- readRDS("plot_all/type1error_blocks_100_dnn_dnn_py_perm_100--1000_boxplots.rds")

p1 <- p1 + theme_light(base_size = 90) +
      theme(
            axis.title.y = element_blank(),
            legend.key.size = unit(5, "cm"),
            legend.title = element_blank(),
            legend.position = c(0.725, 0.125),
            legend.justification = c("left", "bottom"),
            legend.text = element_text(colour="white",    
                                       face="bold"),
            legend.background = element_rect(fill="grey"),
            legend.box.just = "left",
            legend.margin = margin(6, 6, 6, 6),
            strip.text.x = element_text(size = 70),
            plot.margin = grid::unit(
                  c(10., 10., 10., 10.),
                  "mm"
            )
      ) +
      ggtitle("Auc Score") +
      xlab("Number of samples") +
      theme(plot.title = element_text(
            hjust = 0,
            vjust = 0,
            face = "bold"
      )) +
      geom_hline(
        yintercept = 0.5,
        color = "black"
      ) +
      scale_color_discrete(labels = c("CRF-DNN", "Permfit-DNN")) +
      coord_cartesian(ylim = c(0.4, 1.0)) +
      facet_wrap(~ prob_data,
                 ncol = 3,
                 nrow = 2,
                 strip.position = c("top"),
                 scales = "free_y",
                 labeller = labeller(prob_data =
                  c("classification" = "Classification",
                    "regression" = "Regression",
                    "regression_relu" = "Regression with Relu",
                    "regression_combine" = "Main Effects and Interactions",
                    "regression_product" = "Interactions only"))
                 )

ggsave(file.path("plot_all", "AUC_blocks_100_dnn_dnn_py_perm_100--1000_mediane.pdf"),
      p1,
      width = 48,
      height = 35
)
# stop()
p2 <- p2 + theme_light(base_size = 90) +
      theme(
            axis.title.y = element_blank(),
            legend.key.size = unit(5, "cm"),
            legend.position = c(0.725, 0.125),
            legend.justification = c("left", "bottom"),
            legend.text = element_text(colour="white",    
                                       face="bold"),
            legend.background = element_rect(fill="grey"),
            legend.title = element_blank(),
            legend.box.just = "left",
            legend.margin = margin(6, 6, 6, 6),
            strip.text.x = element_text(size = 70)
      ) +
      ggtitle("Type I error") +
      xlab("Number of samples") +
      theme(
            plot.title = element_text(
                  hjust = 0,
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
      coord_cartesian(ylim = c(0.0, 0.4)) +
      scale_fill_discrete(labels = c("CRF-DNN", "Permfit-DNN")) +
      facet_wrap(~ prob_data,
                 ncol = 3,
                 nrow = 2,
                 strip.position = c("top"),
                 scales = "free_y",
                 labeller = labeller(prob_data =
                  c("classification" = "Classification",
                    "regression" = "Regression",
                    "regression_relu" = "Regression with Relu",
                    "regression_combine" = "Main Effects and Interactions",
                    "regression_product" = "Interactions only"))
                 )
ggsave(file.path("plot_all", "type1error_blocks_100_dnn_dnn_py_perm_100--1000_boxplots.pdf"),
      p2,
      width = 48,
      height = 35
)

unlink("Rplots.pdf")