library(Rmisc)
library(tidyverse)
library(stringr)
library(scales)
library(grid)
library(ggpubr)
library(ggforce)
library(ggrepel)
library(ggh4x)
library(showtext)
library(ggsci)
library(MASS)
library(lme4)
library(brms)
library(tidybayes)
library(stats)
library(modelr)
library(plotrix)
library(mgcv)
library(hexbin)
library(formattable)
library(scico)

rm(list=ls())

font_add_google("Roboto Condensed", "roboto_condensed")
showtext_auto()

surp_cols <- c("surp_base_gpt2", "surp_lambda0p0", "surp_lambda0p001", "surp_lambda0p01", "surp_lambda0p1", "surp_lambda1p0")

data_NSC <- read.csv('../MazeNSC/nsc_corpus_data/NSC_lm_features.csv') %>%
  dplyr::select(surp_base_gpt2, surp_lambda0p0, surp_lambda0p001, surp_lambda0p01, surp_lambda0p1, surp_lambda1p0)
data_Provo <- read.csv('../Provo/provo_corpus_data/Provo_lm_features.csv') %>%
  dplyr::select(surp_base_gpt2, surp_lambda0p0, surp_lambda0p001, surp_lambda0p01, surp_lambda0p1, surp_lambda1p0)
data <- data_NSC %>% rbind(data_Provo)

long_df <- data %>%
  # dplyr::select(surp_base_gpt2, surp_lambda0p0, surp_lambda0p001, surp_lambda0p01, surp_lambda0p1, surp_lambda1p0) %>%
  pivot_longer(
    cols = everything(),
    names_to = "lambda",
    values_to = "surprisal"
  ) %>%
  drop_na()

global_min <- min(long_df$surprisal, na.rm = TRUE)
global_max <- max(long_df$surprisal, na.rm = TRUE)

breaks  <- c(0, 5, 10, 15, 20, 25, Inf)
labels_fixed  <- c("[0, 5)", "[5, 10)", "[10, 15)", "[15, 20)", "[20, 25)", "≥ 25")


long_df <- long_df %>%
  mutate(
    interval = cut(surprisal, breaks = breaks, include.lowest = TRUE, right = FALSE),
    interval = fct_relabel(interval, function(s) {
      s <- gsub(",", ", ", s, fixed = TRUE)                
      s <- gsub("\\[(\\d+),\\s*Inf\\]", "[\\1, +∞)", s)     
      s
    }),
    interval = fct_rev(interval)
  )

plot_df <- long_df %>%
  dplyr::count(lambda, interval) %>%
  group_by(lambda) %>%
  mutate(prop = n / sum(n)) %>%
  ungroup()


lambda_labs <- c(
  surp_base_gpt2 = "base",
  surp_lambda0p0 = expression(0),
  surp_lambda0p001 = expression(0.001),
  surp_lambda0p01 = expression(0.01),
  surp_lambda0p1 = expression(0.1),
  surp_lambda1p0 = expression(1)
)

plot_surp_breakdown <- ggplot(plot_df, aes(x = lambda, y = prop, fill = interval)) +
  geom_col(color='black') +
  geom_text(
    data = subset(plot_df, prop > 0.03),
    aes(label = sprintf("%.2f", prop), group = interval),
    position = position_stack(vjust = 0.5),
    color = "black",
    size = 3
  ) +
  scale_fill_viridis_d(option = "viridis", direction = 1) +
  scale_x_discrete(labels = lambda_labs) +
  labs(
    x = expression(bold("Memory constraint " * lambda)),
    y = expression(bold("Proportion of observations")),
    fill = "surprisal interval",
  ) +
  theme_classic(base_size = 12,
                base_family = "roboto_condensed") +
  theme(
    legend.position = "right",
    axis.title.x = element_text(size = 14),
    axis.title.y = element_text(size = 14),
  )
plot_surp_breakdown
ggsave("surprisal_breakdown_all.pdf", plot = plot_surp_breakdown, width=4.2, height=3)

