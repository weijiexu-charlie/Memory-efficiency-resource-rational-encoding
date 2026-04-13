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
library(RColorBrewer)


rm(list=ls())

font_add_google("Roboto Condensed", "roboto_condensed")
showtext_auto()

deltaLL.Provo.firstfix.logRT <- read.csv('data/Provo_logRT_first_fix_deltaLL_crossval.csv') 
deltaLL.Provo.totalRT.logRT <- read.csv('data/Provo_logRT_totalRT_deltaLL_crossval.csv') 
deltaLL.MazeNSC.logRT <- read.csv('data/MazeNSC_logRT_deltaLL_crossval.csv') 
deltaLL.SPRNSC.crit.logRT <- read.csv('data/SPRNSC_critical_logRT_deltaLL_crossval.csv') 
deltaLL.SPRNSC.spill.logRT <- read.csv('data/SPRNSC_spillover_logRT_deltaLL_crossval.csv') 

deltaLL.Provo.firstfix.logRT.allsurp <- read.csv('data/Provo_logRT_allsurp_first_fix_deltaLL_crossval.csv') 
deltaLL.Provo.totalRT.logRT.allsurp <- read.csv('data/Provo_logRT_allsurp_totalRT_deltaLL_crossval.csv') 
deltaLL.MazeNSC.logRT.allsurp <- read.csv('data/MazeNSC_logRT_allsurp_deltaLL_crossval.csv') 
deltaLL.SPRNSC.crit.logRT.allsurp <- read.csv('data/SPRNSC_critical_logRT_allsurp_deltaLL_crossval.csv') 
deltaLL.SPRNSC.spill.logRT.allsurp <- read.csv('data/SPRNSC_spillover_logRT_allsurp_deltaLL_crossval.csv') 



process_RT_data <- function(df, corpus_name){
  df %>% mutate(model = factor(model_pair, levels=c('base', 'lambda0p0', 'lambda0p001', 'lambda0p01', 'lambda0p1', 'lambda1p0')),
                dataset = corpus_name,
                deltaLL = mean_delta_sum,
                se_deltaLL = se_delta_sum)
}

d.Provo.firstfix.logRT <- deltaLL.Provo.firstfix.logRT %>% process_RT_data('Provo (first fixation)')
d.Provo.totalRT.logRT <- deltaLL.Provo.totalRT.logRT %>% process_RT_data('Provo (total RT)')
d.SPRNSC.crit.logRT <- deltaLL.SPRNSC.crit.logRT %>% process_RT_data('SPR NSC (critical)')
d.SPRNSC.spill.logRT <- deltaLL.SPRNSC.spill.logRT %>% process_RT_data('SPR NSC (spillover)')
d.MazeNSC.logRT <- deltaLL.MazeNSC.logRT %>% process_RT_data('Maze NSC')

d.Provo.firstfix.logRT.allsurp <- deltaLL.Provo.firstfix.logRT.allsurp %>% process_RT_data('Provo (first fixation)')
d.Provo.totalRT.logRT.allsurp <- deltaLL.Provo.totalRT.logRT.allsurp %>% process_RT_data('Provo (total RT)')
d.SPRNSC.crit.logRT.allsurp <- deltaLL.SPRNSC.crit.logRT.allsurp %>% process_RT_data('SPR NSC (critical)')
d.SPRNSC.spill.logRT.allsurp <- deltaLL.SPRNSC.spill.logRT.allsurp %>% process_RT_data('SPR NSC (spillover)')
d.MazeNSC.logRT.allsurp <- deltaLL.MazeNSC.logRT.allsurp %>% process_RT_data('Maze NSC')



get_baseline <- function(df){
  df %>%
    filter(model == "base") %>%
    pull(deltaLL)
}

base.gpt2.Provo.firstfix.logRT <- d.Provo.firstfix.logRT %>% get_baseline()
base.gpt2.Provo.totalRT.logRT <- d.Provo.totalRT.logRT %>% get_baseline()
base.gpt2.SPRNSC.crit.logRT <- d.SPRNSC.crit.logRT %>% get_baseline()
base.gpt2.SPRNSC.spill.logRT <- d.SPRNSC.spill.logRT %>% get_baseline()
base.gpt2.MazeNSC.logRT <- d.MazeNSC.logRT %>% get_baseline()

base.gpt2.Provo.firstfix.logRT.allsurp <- d.Provo.firstfix.logRT.allsurp %>% get_baseline()
base.gpt2.Provo.totalRT.logRT.allsurp <- d.Provo.totalRT.logRT.allsurp %>% get_baseline()
base.gpt2.SPRNSC.crit.logRT.allsurp <- d.SPRNSC.crit.logRT.allsurp %>% get_baseline()
base.gpt2.SPRNSC.spill.logRT.allsurp <- d.SPRNSC.spill.logRT.allsurp %>% get_baseline()
base.gpt2.MazeNSC.logRT.allsurp <- d.MazeNSC.logRT.allsurp %>% get_baseline()


ref_lines_base_logRT <- data.frame(
  dataset = c("Provo (first fixation)", "Provo (total RT)", "SPR NSC (critical)", "SPR NSC (spillover)", "Maze NSC"),
  yint  = c(base.gpt2.Provo.firstfix.logRT, base.gpt2.Provo.totalRT.logRT, base.gpt2.SPRNSC.crit.logRT, base.gpt2.SPRNSC.spill.logRT, base.gpt2.MazeNSC.logRT)
) %>%
  mutate(dataset = factor(dataset, levels=c("Provo (first fixation)", "Provo (total RT)", "SPR NSC (critical)", "SPR NSC (spillover)", "Maze NSC")))

ref_lines_base_logRT_allsurp <- data.frame(
  dataset = c("Provo (first fixation)", "Provo (total RT)", "SPR NSC (critical)", "SPR NSC (spillover)", "Maze NSC"),
  yint  = c(base.gpt2.Provo.firstfix.logRT.allsurp, base.gpt2.Provo.totalRT.logRT.allsurp, base.gpt2.SPRNSC.crit.logRT.allsurp, base.gpt2.SPRNSC.spill.logRT.allsurp, base.gpt2.MazeNSC.logRT.allsurp)
) %>%
  mutate(dataset = factor(dataset, levels=c("Provo (first fixation)", "Provo (total RT)", "SPR NSC (critical)", "SPR NSC (spillover)", "Maze NSC")))


df_plot_logRT <- d.Provo.firstfix.logRT %>%
  rbind(d.Provo.totalRT.logRT) %>%
  rbind(d.SPRNSC.crit.logRT) %>%
  rbind(d.SPRNSC.spill.logRT) %>%
  rbind(d.MazeNSC.logRT) %>%
  filter(model != "base")
df_plot_logRT_allsurp <- d.Provo.firstfix.logRT.allsurp %>%
  rbind(d.Provo.totalRT.logRT.allsurp) %>%
  rbind(d.SPRNSC.crit.logRT.allsurp) %>%
  rbind(d.SPRNSC.spill.logRT.allsurp) %>%
  rbind(d.MazeNSC.logRT.allsurp) %>%
  filter(model != "base")



prepare_plot_data <- function(df_plot){
  df_plot %>%
    mutate(lambda = case_when(
      grepl("^lambda0p001$", model) ~ 0.001,
      grepl("^lambda0p01$",  model) ~ 0.01,
      grepl("^lambda0p1$",   model) ~ 0.1,
      grepl("^lambda1p0$",   model) ~ 1,
      grepl("^lambda0p0$",   model) ~ 0
    )) %>%
    mutate(lambda = factor(lambda, levels = c(0, 0.001, 0.01, 0.1, 1), labels = c("0","0.001","0.01","0.1","1"))) %>%
    mutate(dataset = factor(dataset, levels=c("Provo (first fixation)", "Provo (total RT)", "SPR NSC (critical)", "SPR NSC (spillover)", "Maze NSC")))
}


################ Plotting ################


plot_deltaLL <- function(df_plot, ref_lines_base){
  df_plot2 <- df_plot %>% prepare_plot_data()
  df_plot2 %>%
    mutate(dataset = factor(dataset, levels = c(
      "Provo (first fixation)",
      "Provo (total RT)",
      "SPR NSC (critical)",
      "SPR NSC (spillover)",
      "Maze NSC"
    ))) %>%
    ggplot(aes(x = lambda, y = deltaLL)) +
    # ---- Provo ----
    geom_col(data = subset(df_plot2, dataset%in%c("Provo (first fixation)", "Provo (total RT)")), aes(fill=lambda), color = "black") +
    geom_errorbar(data = subset(df_plot2, dataset%in%c("Provo (first fixation)", "Provo (total RT)")),
                  aes(ymin = deltaLL - se_deltaLL,
                      ymax = deltaLL + se_deltaLL),
                  width = 0, linewidth = 0.9, color = "black") +
    scale_fill_brewer(palette = "Blues", direction = 1) +
    new_scale_fill() +
    # ---- SPR NSC ----
    geom_col(data = subset(df_plot2, dataset %in% c("SPR NSC (critical)", "SPR NSC (spillover)")), aes(fill=lambda), color = "black") +
    geom_errorbar(data = subset(df_plot2, dataset%in%c("SPR NSC (critical)", "SPR NSC (spillover)")),
                  aes(ymin = deltaLL - se_deltaLL,
                      ymax = deltaLL + se_deltaLL),
                  width = 0, linewidth = 0.9, color = "black") +
    scale_fill_brewer(palette = "Oranges", direction = 1) +
    new_scale_fill() +
    # ---- Maze ----
    geom_col(data = subset(df_plot2, dataset=="Maze NSC"), aes(fill=lambda), color = "black") +
    geom_errorbar(data = subset(df_plot2, dataset=="Maze NSC"),
                  aes(ymin = deltaLL - se_deltaLL,
                      ymax = deltaLL + se_deltaLL),
                  width = 0, linewidth = 0.9, color = "black") +
    scale_fill_brewer(palette = "Purples", direction = 1) +
    facet_wrap(~dataset, scales = "free", nrow = 1, drop = FALSE,
               labeller = as_labeller(c(
                 "Provo (first fixation)" = "<b>Provo</b> (first fixation)",
                 "Provo (total RT)"       = "<b>Provo</b> (total RT)",
                 "SPR NSC (critical)"     = "<b>SPR NSC</b> (critical)",
                 "SPR NSC (spillover)"    = "<b>SPR NSC</b> (spillover)",
                 "Maze NSC"               = "<b>Maze NSC</b>"
               ))) +
    geom_hline(data = ref_lines_base,
               aes(yintercept = yint),
               linetype = "dashed",
               color = "black") +
    labs(x = expression(bold("Memory constraint " * lambda)),
         y = expression(bold(Delta * "log-Likihood"))) +
    scale_x_discrete(drop = FALSE) +
    theme_classic(base_family = "roboto_condensed") +
    theme(legend.position = "none",
          axis.title.x = element_text(size = 13),
          axis.title.y = element_text(size = 13),
          strip.text = element_markdown(size = 11, face = "plain"))
}

p.logRT <- df_plot_logRT %>% plot_deltaLL(ref_lines_base_logRT)
p.logRT.allsurp <- df_plot_logRT_allsurp %>% plot_deltaLL(ref_lines_base_logRT_allsurp)


ggsave("RT_plots/logRT_deltaLL_crossval.pdf", plot = p.logRT, width=11, height=2.8)
ggsave("RT_plots/logRT_allsurp_deltaLL_crossval.pdf", plot = p.logRT.allsurp, width=11, height=2.8)





