library(Rmisc)
library(tidyverse)
library(stringr)
library(scales)
library(grid)
library(ggpubr)
library(MASS)
# library(lmerTest)
library(rsample)
library(lme4)
library(brms)
library(stats)
library(modelr)
library(plotrix)
library(mgcv)
library(hexbin)
library(formattable)

rm(list=ls())

data_all <- read.csv('corpus_data/NSC_RT_lm_features.csv') %>%
  filter(distractor!="x-x-x") %>%
  dplyr::select(subject, story_id, sentence_id, word_id, word, has_punct, correct, rt, total_rt, 
                word_surp_base_gpt2, word_surp_lambda0p0, 
                word_surp_lambda0p001, word_surp_lambda0p01, word_surp_lambda0p1,
                word_surp_lambda1p0, logfreq, global_word_id)

data <- data_all %>%
  mutate(subject = as.numeric(subject),
         story_id = as.numeric(story_id),
         sentence_id = as.numeric(sentence_id),
         word_id = as.numeric(word_id),
         RT = as.numeric(rt),
         total_RT = as.numeric(total_rt),
         word_len = nchar(word),
         word_surp_base_gpt2 = as.numeric(word_surp_base_gpt2),
         word_surp_lambda0p0 = as.numeric(word_surp_lambda0p0),
         word_surp_lambda0p001 = as.numeric(word_surp_lambda0p001),
         word_surp_lambda0p01 = as.numeric(word_surp_lambda0p01),
         word_surp_lambda0p1 = as.numeric(word_surp_lambda0p1),
         word_surp_lambda1p0 = as.numeric(word_surp_lambda1p0),
         logfreq = as.numeric(logfreq)) %>%
  filter(has_punct=='False')

dat <- data %>%
  filter(rt>=100 & rt<=3000) %>%
  filter(correct=='yes') %>%
  mutate(
    logRT = log(RT)
  ) %>%
  group_by(subject, story_id, sentence_id) %>%
  mutate(
    RT_SPILL1 = lag(RT, 1),
    logRT_SPILL1 = log(RT_SPILL1),
    
    word_len_SPILL1 = lag(word_len, 1),
    word_len_SPILL2 = lag(word_len, 2),
    
    logfreq_SPILL1 = lag(logfreq, 1),
    logfreq_SPILL2 = lag(logfreq, 2),
    
    word_surp_base_gpt2_SPILL1 = lag(word_surp_base_gpt2, 1),
    word_surp_base_gpt2_SPILL2 = lag(word_surp_base_gpt2, 2),
    
    word_surp_lambda0p0_SPILL1 = lag(word_surp_lambda0p0, 1),
    word_surp_lambda0p0_SPILL2 = lag(word_surp_lambda0p0, 2),
    
    word_surp_lambda0p001_SPILL1 = lag(word_surp_lambda0p001, 1),
    word_surp_lambda0p001_SPILL2 = lag(word_surp_lambda0p001, 2),
    
    word_surp_lambda0p01_SPILL1 = lag(word_surp_lambda0p01, 1),
    word_surp_lambda0p01_SPILL2 = lag(word_surp_lambda0p01, 2),
    
    word_surp_lambda0p1_SPILL1 = lag(word_surp_lambda0p1, 1),
    word_surp_lambda0p1_SPILL2 = lag(word_surp_lambda0p1, 2),
    
    word_surp_lambda1p0_SPILL1 = lag(word_surp_lambda1p0, 1),
    word_surp_lambda1p0_SPILL2 = lag(word_surp_lambda1p0, 2)
  ) %>%
  ungroup() %>%
  filter(RT > 100 & RT < 3000)

dat.surp.cleaned <- dat %>%
  filter(word_surp_base_gpt2 <= 25) %>%
  filter(word_surp_lambda0p0 <= 25) %>%
  filter(word_surp_lambda0p001 <= 25) %>%
  filter(word_surp_lambda0p01 <= 25) %>%
  filter(word_surp_lambda0p1 <= 25) %>%
  filter(word_surp_lambda1p0 <= 25)

# write.csv(dat.surp.cleaned, file = "../RT_analysis_all/data/MazeNSC_data.csv", row.names = FALSE)


#################### Stats Models ####################

prepare_stats_vars <- function(df){
  df %>%
    mutate(
      word_id.s = scale(word_id),
      
      word_len.s = scale(word_len),
      word_len_SPILL1.s = scale(word_len_SPILL1),
      word_len_SPILL2.s = scale(word_len_SPILL2),
      
      logfreq.s = scale(logfreq),
      logfreq_SPILL1.s = scale(logfreq_SPILL1),
      logfreq_SPILL2.s = scale(logfreq_SPILL2),
      
      word_surp_base_gpt2.s = scale(word_surp_base_gpt2),
      word_surp_base_gpt2_SPILL1.s = scale(word_surp_base_gpt2_SPILL1),
      word_surp_base_gpt2_SPILL2.s = scale(word_surp_base_gpt2_SPILL2),
      
      word_surp_lambda0p0.s = scale(word_surp_lambda0p0),
      word_surp_lambda0p0_SPILL1.s = scale(word_surp_lambda0p0_SPILL1),
      word_surp_lambda0p0_SPILL2.s = scale(word_surp_lambda0p0_SPILL2),
      
      word_surp_lambda0p001.s = scale(word_surp_lambda0p001),
      word_surp_lambda0p001_SPILL1.s = scale(word_surp_lambda0p001_SPILL1),
      word_surp_lambda0p001_SPILL2.s = scale(word_surp_lambda0p001_SPILL2),
      
      word_surp_lambda0p01.s = scale(word_surp_lambda0p01),
      word_surp_lambda0p01_SPILL1.s = scale(word_surp_lambda0p01_SPILL1),
      word_surp_lambda0p01_SPILL2.s = scale(word_surp_lambda0p01_SPILL2),
      
      word_surp_lambda0p1.s = scale(word_surp_lambda0p1),
      word_surp_lambda0p1_SPILL1.s = scale(word_surp_lambda0p1_SPILL1),
      word_surp_lambda0p1_SPILL2.s = scale(word_surp_lambda0p1_SPILL2),
      
      word_surp_lambda1p0.s = scale(word_surp_lambda1p0),
      word_surp_lambda1p0_SPILL1.s = scale(word_surp_lambda1p0_SPILL1),
      word_surp_lambda1p0_SPILL2.s = scale(word_surp_lambda1p0_SPILL2),
      
      logRT_SPILL1.s = scale(logRT_SPILL1),
      RT_SPILL1.s = scale(RT_SPILL1),
      subj = as.factor(subject)
    )
}

d <- dat %>% prepare_stats_vars()
d.surp.cleaned <- dat.surp.cleaned %>% prepare_stats_vars()


#################### Setting up cross validation ####################

set.seed(1)

# Per-observation Gaussian log-likelihood on held-out data
ll_gaussian_oos <- function(mod, newdata, re_form = NULL) {
  mu <- predict(mod, newdata = newdata, re.form = re_form, allow.new.levels = FALSE)
  sigma <- sigma(mod) 
  y <- model.response(model.frame(formula(mod), newdata, na.action = na.pass))
  stats::dnorm(y, mean = mu, sd = sigma, log = TRUE)
}

score_split_cumulative <- function(split, form_full, form_null, subj_var = "subj") {
  train <- analysis(split)
  test <- assessment(split)
  
  ok <- test[[subj_var]] %in% train[[subj_var]]
  test_keep <- test[ok, , drop = FALSE]
  n_dropped <- sum(!ok)
  
  ctrl <- lmerControl(calc.derivs = FALSE)
  m_full <- lmer(form_full, data = train, REML = FALSE, control = ctrl)
  m_null <- lmer(form_null, data = train, REML = FALSE, control = ctrl)
  
  ll_full <- ll_gaussian_oos(m_full, test_keep, re_form = NULL)  
  ll_null <- ll_gaussian_oos(m_null, test_keep, re_form = NULL)
  
  diff_ll <- ll_full - ll_null
  keep <- is.finite(diff_ll)
  
  tibble(
    n_obs = sum(keep),
    n_dropped = n_dropped,
    delta_sum = sum(diff_ll[keep]),      
    delta_mean  = mean(diff_ll[keep])     
  )
}

# Run CV
run_cv_cumulative <- function(data, model_pairs, subj_var = subj, v = 10) {
  # row-wise folds, stratified so each fold gets rows from most subjects
  folds <- vfold_cv(data, v = v, strata = {{ subj_var }})
  
  per_fold <- imap_dfr(model_pairs, function(pair, pair_id) {
    map_dfr(folds$splits, function(splt) {
      score_split_cumulative(
        splt,
        form_full = pair$full,
        form_null = pair$null,
        subj_var = rlang::as_name(rlang::ensym(subj_var))
      )
    }, .id = "fold") |>
      mutate(model_pair = pair$name, .before = 1)
  })
  
  summary_tbl <- per_fold |>
    group_by(model_pair) |>
    summarise(
      folds = n(),
      mean_delta_sum  = mean(delta_sum),                  
      se_delta_sum = sd(delta_sum) / sqrt(n()),          
      micro_delta_sum = sum(delta_sum),                      
      mean_n_per_fold = mean(n_obs),
      total_heldout = sum(n_obs),
      total_dropped = sum(n_dropped),
      .groups = "drop"
    )
  
  list(per_fold = per_fold, summary = summary_tbl)
}


#################### Run stats models ####################

form_m0_logRT <- logRT ~
  word_id.s + word_len.s + word_len_SPILL1.s + word_len_SPILL2.s +
  logfreq.s + logfreq_SPILL1.s + logfreq_SPILL2.s +
  (1 | subj)

form_m1_base_logRT <- logRT ~
  word_id.s + word_len.s + word_len_SPILL1.s + word_len_SPILL2.s +
  logfreq.s + logfreq_SPILL1.s + logfreq_SPILL2.s +
  word_surp_base_gpt2.s + word_surp_base_gpt2_SPILL1.s + word_surp_base_gpt2_SPILL2.s +
  (1 | subj)
form_m1_lambda0p0_logRT <- logRT ~
  word_id.s + word_len.s + word_len_SPILL1.s + word_len_SPILL2.s +
  logfreq.s + logfreq_SPILL1.s + logfreq_SPILL2.s +
  word_surp_lambda0p0.s + word_surp_lambda0p0_SPILL1.s + word_surp_lambda0p0_SPILL2.s +
  (1 | subj)
form_m1_lambda0p001_logRT <- logRT ~
  word_id.s + word_len.s + word_len_SPILL1.s + word_len_SPILL2.s +
  logfreq.s + logfreq_SPILL1.s + logfreq_SPILL2.s +
  word_surp_lambda0p001.s + word_surp_lambda0p001_SPILL1.s + word_surp_lambda0p001_SPILL2.s +
  (1 | subj)
form_m1_lambda0p01_logRT <- logRT ~
  word_id.s + word_len.s + word_len_SPILL1.s + word_len_SPILL2.s +
  logfreq.s + logfreq_SPILL1.s + logfreq_SPILL2.s +
  word_surp_lambda0p01.s + word_surp_lambda0p01_SPILL1.s + word_surp_lambda0p01_SPILL2.s +
  (1 | subj)
form_m1_lambda0p1_logRT <- logRT ~
  word_id.s + word_len.s + word_len_SPILL1.s + word_len_SPILL2.s +
  logfreq.s + logfreq_SPILL1.s + logfreq_SPILL2.s +
  word_surp_lambda0p1.s + word_surp_lambda0p1_SPILL1.s + word_surp_lambda0p1_SPILL2.s +
  (1 | subj)
form_m1_lambda1p0_logRT <- logRT ~
  word_id.s + word_len.s + word_len_SPILL1.s + word_len_SPILL2.s +
  logfreq.s + logfreq_SPILL1.s + logfreq_SPILL2.s +
  word_surp_lambda1p0.s + word_surp_lambda1p0_SPILL1.s + word_surp_lambda1p0_SPILL2.s +
  (1 | subj)


model_pairs_logRT <- list(
  list(name = "base", full = form_m1_base_logRT, null = form_m0_logRT),
  list(name = "lambda0p0", full = form_m1_lambda0p0_logRT, null = form_m0_logRT),
  list(name = "lambda0p001", full = form_m1_lambda0p001_logRT, null = form_m0_logRT),
  list(name = "lambda0p01", full = form_m1_lambda0p01_logRT, null = form_m0_logRT),
  list(name = "lambda0p1", full = form_m1_lambda0p1_logRT, null = form_m0_logRT),
  list(name = "lambda1p0", full = form_m1_lambda1p0_logRT, null = form_m0_logRT)
)

# Run cleaned surp
res_logRT <- run_cv_cumulative(
  data = d.surp.cleaned,
  model_pairs = model_pairs_logRT,
  subj_var = subj,  # same subject levels appear in train & test
  v = 10
)
res_logRT$per_fold
res_logRT$summary
write.csv(res_logRT$summary, file = "../RT_analysis_all/data/MazeNSC_logRT_deltaLL_crossval.csv", row.names = FALSE)

# Run all surp
res_allsurp_logRT <- run_cv_cumulative(
  data = d,
  model_pairs = model_pairs_logRT,
  subj_var = subj,  # same subject levels appear in train & test
  v = 10
)
res_allsurp_logRT$per_fold
res_allsurp_logRT$summary
write.csv(res_allsurp_logRT$summary, file = "../RT_analysis_all/data/MazeNSC_logRT_allsurp_deltaLL_crossval.csv", row.names = FALSE)

