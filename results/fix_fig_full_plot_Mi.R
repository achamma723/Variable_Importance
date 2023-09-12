library(ggplot2)
library(ggthemes)
library(plyr)
library(patchwork)
library(grid)
library(gridExtra)
library(tidyverse)
library(scales)

p1 <- readRDS("plot_all/AUC_blocks_100_Mi_dnn_dnn_py_300:100.rds")
p2 <- readRDS("plot_all/type1error_blocks_100_Mi_dnn_dnn_py_300:100.rds")
p3 <- readRDS("plot_all/power_blocks_100_Mi_dnn_dnn_py_300:100.rds")
p4 <- readRDS("plot_all/time_bars_blocks_100_Mi_dnn_dnn_py_300:100.rds")

p1 <- p1 + theme_light(base_size = 55) +
      theme(
            axis.title.x = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank(),
            legend.key.size = unit(5, "cm"),
            legend.title = element_blank(),
            legend.position = "none",
            legend.justification = c("left", "bottom"),
            legend.box.just = "left",
            legend.margin = margin(6, 6, 6, 6),
            strip.text.y.left = element_text(size = 50, angle = 0),
            plot.margin = grid::unit(
                  c(10., 10., 10., 10.),
                  "mm"
            )
      ) +
      ggtitle("Auc Score") +
      xlab("Correlation") +
      ylim(0.4, 1.0) +
      theme(plot.title = element_text(
            hjust = 0,
            vjust = 0,
            face = "bold"
      )) +
      geom_hline(
        yintercept = 0.5,
        color = "black"
      ) +
      guides(fill = guide_legend(reverse = TRUE, ncol = 3)) +
      scale_fill_manual(values = c("cyan3", "brown2")) +
      scale_colour_manual(values = c("cyan3", "brown2")) +
      facet_wrap(~ correlation,
                 ncol = 1,
                 strip.position = c("left"),
                 scales = "free_y"
                 )

p2 <- p2 + theme_light(base_size = 55) +
      theme(
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank(),
            legend.key.size = unit(5, "cm"),
            legend.title = element_blank(),
            # legend.position = c(0.5256, 0),
            legend.position = "none",
            legend.justification = c("left", "bottom"),
            legend.box.just = "left",
            legend.margin = margin(6, 6, 6, 6),
            strip.text.y.left = element_text(size = 50, angle = 0)
      ) +
      ggtitle("Type I error") +
      xlab("Correlation") +
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
      guides(fill = guide_legend(reverse = TRUE, ncol = 3)) +
      scale_fill_manual(values = c("cyan3", "brown2")) +
      scale_colour_manual(values = c("cyan3", "brown2")) +
      facet_wrap(~ correlation,
                 ncol = 1,
                 strip.position = c("left"),
                 scales = "free_y"
                 )

p3 <- p3 + theme_light(base_size = 55) +
      theme(
            axis.title.x = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.y = element_blank(),
            legend.key.size = unit(5, "cm"),
            legend.title = element_blank(),
            legend.position = "none",
            legend.justification = c("left", "bottom"),
            legend.box.just = "left",
            legend.margin = margin(6, 6, 6, 6),
            strip.text.y.left = element_text(size = 50, angle = 0)
      ) +
      ggtitle("Power") +
      xlab("Correlation") +
      theme(
            plot.title = element_text(
                  hjust = 0,
                  vjust = 0,
                  face = "bold"
            ),
            plot.margin = grid::unit(
                  c(0., 10., 0., 10.),
                  "mm"
            )
      ) +
      guides(fill = guide_legend(reverse = TRUE, ncol = 3)) +
      scale_fill_manual(values = c("cyan3", "brown2")) +
      scale_colour_manual(values = c("cyan3", "brown2")) +
      facet_wrap(~ correlation,
                 ncol = 1,
                 strip.position = c("left"),
                 scales = "free_y"
                  )

p4 <- p4 + theme_light(base_size = 55) +
      ggtitle("Computation time") +
      theme(
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
      )


lay <- rbind(
      c(1, 1, 2, 2),
      c(4, 4, 3, 3)
)

p <- p1 + p2 + p3 + p4 + plot_layout(ncol = 4, guides = 'collect') & theme(legend.position = 'bottom')

ggsave(file.path("plot_all", "test_blocks_100_Mi_dnn_dnn_py_300:100.pdf"),
      p,
      width = 42,
      height = 10
)

unlink("Rplots.pdf")