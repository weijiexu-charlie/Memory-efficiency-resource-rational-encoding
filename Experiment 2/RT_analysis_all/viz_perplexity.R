library(Rmisc)
library(tidyverse)
library(stringr)
library(scales)
library(grid)
library(ggpubr)
library(ggtext)
library(MASS)
library(lmerTest)
library(lme4)
library(brms)
library(stats)
library(modelr)
library(plotrix)
library(mgcv)
library(hexbin)
library(formattable)
library(ggnewscale)
library(showtext)
library(scico)


rm(list=ls())

font_add_google("Roboto Condensed", "roboto_condensed")
showtext_auto()

d.NSC <- read.csv('../SPRNSC/corpus_data/NSC_lm_features.csv') 
d.Provo <- read.csv('../Provo/corpus_data/Provo_lm_features.csv') 

d.NSC.surp <- d.NSC %>%
  dplyr::select(surp_base_gpt2, surp_lambda0p0, surp_lambda0p001, surp_lambda0p01, surp_lambda0p1, surp_lambda1p0) %>%
  mutate(corpus = 'NSC') %>%
  tidyr::pivot_longer(
    cols = starts_with("surp_"),
    names_to = "model",
    values_to = "surprisal"
  )
d.Provo.surp <- d.Provo %>%
  dplyr::select(surp_base_gpt2, surp_lambda0p0, surp_lambda0p001, surp_lambda0p01, surp_lambda0p1, surp_lambda1p0) %>%
  mutate(corpus = 'Provo') %>%
  tidyr::pivot_longer(
    cols = starts_with("surp_"),
    names_to = "model",
    values_to = "surprisal"
  )

d.surp.all <- d.NSC.surp %>% 
  rbind(d.Provo.surp) %>% 
  mutate(model_label = case_when(
    grepl("base_gpt2$", model) ~ "base",
    grepl("lambda0p0$", model) ~ "0",
    grepl("lambda0p001$", model) ~ "0.001",
    grepl("lambda0p01$", model) ~ "0.01",
    grepl("lambda0p1$", model) ~ "0.1",
    grepl("lambda1p0$", model) ~ "1"
  )) %>%
  mutate(model_label = factor(model_label, levels=c("base", "0", "0.001", "0.01", "0.1", "1")))


perplexity_plot_line <- d.surp.all %>%
  group_by(model_label) %>%
  summarise(perplexity = 2^(mean(surprisal, na.rm = TRUE))) %>%
  ggplot(aes(x = model_label, y = perplexity)) +
  geom_line(aes(x = model_label, y = perplexity, group=1), color = "black", size=1) +
  geom_point(aes(fill = model_label), 
             shape = 21,        # circle with fill + border
             color = "black",   # border
             size = 3,
             stroke = 1.4) +
  coord_cartesian(ylim = c(100, NA)) +
  scale_fill_scico_d(palette = 'davos', direction = -1) +
  new_scale_fill() +
  scale_y_continuous(
    breaks = 2^(0:10), # Set breaks at powers of 2 from 2^0 to 2^10
    labels = trans_format("log2", math_format(2^.x)) # Format labels as 2^power
  ) +
  labs(x = expression(bold("Memory constraint " * lambda)),
       y = expression(bold("Perplexity"))) +
  scale_x_discrete(drop = FALSE) +
  theme_classic(base_family = "roboto_condensed") +
  theme(legend.position = "none",
        axis.text.x = element_text(size = 9.5),
        axis.text.y = element_text(size = 11),
        axis.title.x = element_text(size = 13),
        axis.title.y = element_text(size = 13),
        strip.text = element_markdown(size = 11, face = "plain"))
perplexity_plot_line
ggsave("perplexity_RT_corpus_line.pdf", plot = perplexity_plot_line, width=2.5, height=2.5)






