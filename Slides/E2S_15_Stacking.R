###########################################################
# STACKING THE BOSTON HOUSING DATA                        #
#                                                         #
# We use four different algorithms to predict the median  #
# value of owner-occupied homes (cmedv, in $1000s):       #
#   (1) k-nearest neighbors;                              #
#   (2) elastic net regression;                           #
#   (3) random forests; and                               #
#   (4) gradient boosting (XGBoost).                      #
#                                                         #
# The individual learners are then combined into a        #
# stacked ensemble using the stacks package. Stacking     #
# (a.k.a. stacked generalization) fits a meta-learner     #
# on the out-of-fold predictions of the base learners.    #
# Here, the meta-learner is a non-negative LASSO, so      #
# that weak or redundant candidates are automatically     #
# zeroed out. Terminology used throughout:                #
#   CANDIDATE = any tuned configuration offered to the    #
#               meta-learner (here: 60 in total);         #
#   MEMBER    = a candidate that received a non-zero      #
#               stacking weight and thus survives into    #
#               the final ensemble.                       #
###########################################################

# Packages ------------------------------------------------
# mlbench     : contains the Boston housing data
# stacks      : builds the stacked ensemble
# tidymodels  : the modeling meta-package (rsample, recipes,
#               parsnip, tune, yardstick, workflows, ...)
# tidyverse   : data wrangling and plotting
# Engines used behind the scenes (install if missing):
#   kknn (knn), glmnet (elastic net), ranger (random
#   forest), and xgboost (boosting).

library(mlbench)
library(stacks)
library(tidymodels)
library(tidyverse)

# tidymodels_prefer() resolves name clashes in favor of
# tidymodels/tidyverse functions.
tidymodels_prefer()

# Data ----------------------------------------------------
# BostonHousing2 ships with mlbench: 506 census tracts and
# the CORRECTED median home value (cmedv), which we use as
# the outcome. The 13 predictors include crime rate (crim),
# average number of rooms (rm), pupil-teacher ratio
# (ptratio), etc. chas and rad are stored as factors, so we
# convert them to numeric; we drop the identifiers (town,
# tract), the coordinates (lon, lat), and the uncorrected
# outcome (medv).
data("BostonHousing2")
Boston <- BostonHousing2 |>
  mutate(chas = as.numeric(chas),
         rad = as.numeric(rad),
         tax = as.numeric(tax)) |>
  dplyr::select(c(-town, -tract, -lon, -lat, -medv))
Boston <- as_tibble(Boston)
glimpse(Boston)

# Resampling Steps ----------------------------------------
# We hold out 20% of the data as a test set. Stratifying
# on the outcome ensures that the distribution of cmedv is
# similar in the training and test sets (initial_split
# bins a numeric outcome into quartiles for this purpose).
set.seed(10)
boston_split <- initial_split(Boston, prop = 0.80, strata = cmedv)
train <- training(boston_split)
test  <- testing(boston_split)

# 10-fold cross-validation on the training data. The SAME
# folds must be used for every learner: stacking relies on
# out-of-fold predictions that are comparable across
# candidates, which requires identical resamples.
set.seed(20)
folds <- vfold_cv(train, v = 10)

# Recipe --------------------------------------------------
# We start with a generic recipe (just the model formula);
# for each algorithm, we then add the preprocessing steps
# that this algorithm needs.
boston_rec <- recipe(
  cmedv ~ ., data = train
)

# Metric --------------------------------------------------
# We tune and compare on R-squared: the proportion of the
# variance in cmedv that a model explains out of sample.
# (Later, on the test set, we also report RMSE, which is
# on the same scale as cmedv and hence easy to interpret.)
metric <- metric_set(rsq)

# Control -------------------------------------------------
# To combine tuning results into an ensemble later, we
# need to save the out-of-fold predictions and the fitted
# workflows. The control_stack_grid() function takes care
# of this for models tuned via tune_grid(). (If we had a
# model without tuning parameters, we would use
# control_stack_resamples() with fit_resamples() instead.)
ctrl_grid <- control_stack_grid()

# =========================================================
# (1) k-NEAREST NEIGHBORS
# =========================================================
# knn predicts cmedv for a tract as the (weighted) average
# of the k most similar tracts. "Similar" is measured by
# distance in predictor space, so the scale of the
# predictors matters enormously: without normalization,
# a variable like tax (in the hundreds) would dominate a
# variable like nox (below 1).

# Specification: we tune the number of neighbors k.
# Small k = flexible but noisy; large k = smooth but
# potentially biased.
knn_spec <-
  nearest_neighbor(
    mode = "regression",
    neighbors = tune("k")
  ) |>
  set_engine("kknn")
knn_spec

# Extended recipe: center and scale all predictors so that
# every variable contributes comparably to the distance.
knn_rec <- boston_rec |>
  step_normalize(all_numeric_predictors())
knn_rec

# Create a work flow (model + recipe travel together)
knn_wf <- workflow() |>
  add_model(knn_spec) |>
  add_recipe(knn_rec)

# Tuning: grid = 15 asks tune_grid() to pick 15 sensible
# values of k automatically. Each value of k becomes a
# separate CANDIDATE for the ensemble.
set.seed(30)
knn_res <- tune_grid(
  knn_wf,
  resamples = folds,
  metrics = metric,
  grid = 15,
  control = ctrl_grid
)
knn_res

# =========================================================
# (2) ELASTIC NET REGRESSION
# =========================================================
# The elastic net is penalized linear regression. It
# shrinks coefficients toward zero, trading a little bias
# for a reduction in variance. Two tuning parameters:
#   penalty (lambda) : overall amount of shrinkage
#   mixture (alpha)  : blend of the two penalty types;
#                      0 = ridge (L2), 1 = LASSO (L1),
#                      values in between = elastic net.
# The LASSO component can set coefficients exactly to
# zero, performing variable selection.
enet_spec <-
  linear_reg(
    penalty = tune("penalty"),
    mixture = tune("mixture")
  ) |>
  set_engine("glmnet")
enet_spec

# Extended recipe: glmnet requires the predictors to be on
# a common scale, since a single penalty is applied to all
# coefficients.
enet_rec <- boston_rec |>
  step_normalize(all_numeric_predictors())
enet_rec

# Create a work flow
enet_wf <- workflow() |>
  add_model(enet_spec) |>
  add_recipe(enet_rec)

# Tuning over 15 (penalty, mixture) combinations.
set.seed(40)
enet_res <- tune_grid(
  enet_wf,
  resamples = folds,
  metrics = metric,
  grid = 15,
  control = ctrl_grid
)
enet_res

# =========================================================
# (3) RANDOM FORESTS
# =========================================================
# A random forest is an ensemble of decision trees, each
# grown on a bootstrap sample of the training data and
# using only a random subset of predictors at each split.
# Averaging many de-correlated trees reduces variance.
# Tuning parameters:
#   mtry  : number of predictors sampled at each split
#   min_n : minimum node size before splitting stops
# Trees are invariant to monotone transformations of the
# predictors, so NO normalization is needed -- the base
# recipe suffices.
rf_spec <-
  rand_forest(
    mode = "regression",
    mtry = tune("mtry"),
    min_n = tune("min_n"),
    trees = 500
  ) |>
  set_engine("ranger")
rf_spec

# Extended recipe: none needed, we use the base recipe.
rf_rec <- boston_rec
rf_rec

# Create a work flow
rf_wf <- workflow() |>
  add_model(rf_spec) |>
  add_recipe(rf_rec)

# Tuning: mtry has a data-dependent upper bound (the
# number of predictors), which tune_grid() resolves
# automatically from the recipe.
set.seed(50)
rf_res <- tune_grid(
  rf_wf,
  resamples = folds,
  metrics = metric,
  grid = 15,
  control = ctrl_grid
)
rf_res

# =========================================================
# (4) GRADIENT BOOSTING (XGBOOST)
# =========================================================
# Boosting also combines many trees, but SEQUENTIALLY:
# each new tree is fit to the residuals of the current
# ensemble, slowly correcting its mistakes. Key tuning
# parameters:
#   trees      : number of boosting iterations
#   tree_depth : depth of each individual tree
#   learn_rate : how much each tree contributes; small
#                values learn slowly but generalize better
# Like random forests, boosted trees need no
# normalization. (With categorical predictors we would
# need step_dummy(), but our predictors are all numeric.)
xgb_spec <-
  boost_tree(
    mode = "regression",
    trees = tune("trees"),
    tree_depth = tune("depth"),
    learn_rate = tune("learn_rate")
  ) |>
  set_engine("xgboost")
xgb_spec

# Extended recipe: none needed, we use the base recipe.
xgb_rec <- boston_rec
xgb_rec

# Create a work flow
xgb_wf <- workflow() |>
  add_model(xgb_spec) |>
  add_recipe(xgb_rec)

# Tuning over 15 combinations of the three parameters.
set.seed(60)
xgb_res <- tune_grid(
  xgb_wf,
  resamples = folds,
  metrics = metric,
  grid = 15,
  control = ctrl_grid
)
xgb_res

# =========================================================
# PUTTING TOGETHER A STACK
# =========================================================
# add_candidates() collects the out-of-fold predictions of
# every tuned configuration. Note the bookkeeping: 15 knn
# candidates + 15 elastic nets + 15 forests + 15 boosters
# = 60 candidate columns (before any pruning of redundant
# candidates by stacks itself).
boston_data_st <- stacks() |>
  add_candidates(knn_res) |>
  add_candidates(enet_res) |>
  add_candidates(rf_res) |>
  add_candidates(xgb_res)
as_tibble(boston_data_st)

# Fitting the stack ---------------------------------------
# blend_predictions() fits the meta-learner: a non-negative
# LASSO regression of the observed cmedv on the candidates'
# out-of-fold predictions. The LASSO penalty (itself chosen
# by cross-validation) forces most candidate weights to
# zero, so only the strongest and most complementary
# candidates survive as members.
#
# IMPORTANT: the default penalty grid is 10^(-6:-1), i.e.,
# the largest penalty considered is 0.1. On the scale of
# cmedv (roughly 5-50), such penalties are so weak that the
# meta-learner barely shrinks at all and nearly ALL 60
# candidates keep a (tiny) non-zero weight. We therefore
# supply a stronger penalty range and let cross-validation
# pick. The result: a much sparser -- and equally accurate
# -- ensemble.
set.seed(70)
boston_model_st <-
  boston_data_st |>
  blend_predictions(penalty = 10^seq(-2, 1, length.out = 25))

# How does performance vary with the penalty, and how many
# members are retained at each value? Note how the error is
# essentially flat while the member count collapses: most
# of the 60 candidates add nothing.
autoplot(boston_model_st)
# Which members survived...
autoplot(boston_model_st, type = "members")
# ...and with what stacking weights? Do not be surprised if
# an entire model class (e.g., the elastic net) is missing
# here: the meta-learner keeps a candidate only if it
# improves the blend GIVEN the members already selected.
# A linear model captures a subset of what the tree
# ensembles capture, so conditional on those, its
# predictions are redundant and its weight is exactly 0.
# Stacking with a sparse meta-learner is thus as much a
# model SELECTION device as a model combination device.
autoplot(boston_model_st, type = "weights")

# Fit on full training set --------------------------------
# So far, the surviving members have only been fit on
# resamples. fit_members() refits each member (and ONLY the
# members -- zero-weight candidates are discarded) on the
# ENTIRE training set, so the ensemble is ready for
# prediction.
boston_model_st <-
  boston_model_st |>
  fit_members()

# Who is in, who is out? -----------------------------------
# The print method gives the quickest overview: it reports
# how many of the 60 candidates were retained, with their
# stacking coefficients, broken down by model type.
boston_model_st

# The names of the surviving members (these are exactly the
# columns that predict(..., members = TRUE) will return):
names(boston_model_st$member_fits)

# A full roster: every candidate, its stacking coefficient
# (coef = 0 means dropped), and a membership flag. The
# tuning-parameter columns differ across learners, so we
# keep just the name and the coefficient before row-binding.
membership <-
  c("knn_res", "enet_res", "rf_res", "xgb_res") |>
  map(\(cand) collect_parameters(boston_model_st, cand) |>
        dplyr::select(member, coef)) |>
  list_rbind() |>
  mutate(in_ensemble = coef != 0) |>
  arrange(desc(coef))
membership                          # all 60, ranked by weight
filter(membership, in_ensemble)     # the members only

# To see which tuning-parameter values a surviving member
# corresponds to, inspect its learner and filter on coef:
collect_parameters(boston_model_st, "xgb_res") |>
  filter(coef != 0)

# Correlations among surviving members ---------------------
# Stacking pays off when members make DIFFERENT mistakes:
# highly correlated members are redundant, while less
# correlated ones complement each other. We correlate the
# members' predictions on the test set. (Reminder: with the
# native pipe |>, the magrittr placeholder . does not
# exist, so we pass test to predict() explicitly.)
member_preds <-
  test |>
  dplyr::select(cmedv) |>
  bind_cols(predict(boston_model_st, test, members = TRUE))

# Drop the outcome and the ensemble's own prediction
# (.pred), then correlate and plot as a heatmap. Even after
# pruning, the surviving members correlate quite highly --
# the LASSO keeps them because each still contributes a
# distinct sliver of signal at the margin.
member_preds |>
  dplyr::select(-cmedv, -.pred) |>
  cor() |>
  as_tibble(rownames = "member1") |>
  pivot_longer(-member1, names_to = "member2", values_to = "r") |>
  ggplot(aes(member1, member2, fill = r)) +
  geom_tile() +
  geom_text(aes(label = round(r, 2)), size = 3) +
  scale_fill_gradient(limits = c(NA, 1)) +
  labs(x = NULL, y = NULL, title = "Correlations among stack members") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Prediction ----------------------------------------------
# Evaluate the ensemble on the held-out test set. We store
# the predictions in a NEW object rather than overwriting
# test: re-running "test <- test |> bind_cols(...)" a
# second time would accumulate duplicate .pred columns.
test_pred <-
  test |>
  bind_cols(predict(boston_model_st, test))

# The scatter of predicted vs. observed cmedv should hug
# the 45-degree line.
ggplot(test_pred, aes(x = cmedv, y = .pred)) +
  geom_point(alpha = 0.6) +
  geom_abline(linetype = "dashed") +
  labs(
    x = "Observed cmedv ($1000s)",
    y = "Predicted cmedv ($1000s)",
    title = "Stacked ensemble predictions on the test set"
  )

# Test-set performance of the ensemble
rmse(test_pred, truth = cmedv, estimate = .pred)
rsq(test_pred, truth = cmedv, estimate = .pred)

# Comparison ----------------------------------------------
# Does the ensemble beat its individual members? The
# member_preds object built above contains the test-set
# predictions of the blend (.pred) and of every member, so
# we can compute a test RMSE for each. The first entry
# (cmedv against itself) is trivially 0; among the rest,
# the blend should sit at or near the top.
map_dbl(member_preds, rmse_vec, truth = member_preds$cmedv) |>
  sort()

# NOTE on re-running pieces of this script: blend_
# predictions() and fit_members() REPLACE boston_model_st,
# but objects computed from an earlier version (such as
# member_preds or test_pred) are NOT updated automatically.
# After re-blending, re-run everything from fit_members()
# down.
