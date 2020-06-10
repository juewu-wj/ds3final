# Load Packages -----------------------------------------------------------
library(tidyverse)
library(janitor)
library(modelr)
library(rsample)
library(ranger)
library(vip)
library(pdp)
library(xgboost)
library(glmnet)
library(glmnetUtils) 
library(e1071)
library(nnet)
library(keras)
library(data.table)
library(mltools)

# Load Data ---------------------------------------------------------------
# Read in cleaned data
hsls <- read_rds("data/processed/hsls.rds")

# Set seed
set.seed(3)

# Split: 15% testing, 85% modeling (80% training, 20% EDA)
hsls_split_info <- hsls %>% 
  initial_split(prop = 0.85) 

hsls_split <- tibble(
  eda = hsls_split_info %>% training() %>% sample_frac(0.2) %>% list(),
  train = hsls_split_info %>% training() %>% setdiff(eda) %>% list(),
  test = hsls_split_info %>% testing() %>% list()
)

# Select train and test data
train <- hsls_split %>% 
  pluck("train", 1) 

test <- hsls_split %>% 
  pluck("test", 1) 

# Model Building ----------------------------------------------------------
### SVM
# Create svm_dat
svm_dat <- tibble(
  train = train %>% list(),
  test  = test %>% list()
)

# Linear
svm_cv <- svm_dat %>%
  mutate(tune_svm_linear = map(.x = train, 
                               .f = function(x){ 
                                 return(tune(svm, stem ~., data = x, kernel = "linear", 
                                             # let cost range over several values
                                             ranges = list(cost = c(0.01, 0.1, 1, 5, 10)))
                                 )
                               }))

# get best parameters
linear_params <- svm_cv %>% 
  pluck("tune_svm_linear", 1) %>%
  pluck("best.parameters")

linear_params

# fit the model with best cost: linear_params cost = 1
svm_linear_model <- svm_dat %>%
  mutate(model_fit = map(.x = train, # fit the model
                         .f = function(x) svm(stem ~ ., data = x, cost = linear_params$cost, kernel = "linear")), 
         test_pred = map2(model_fit, test, predict), # get predictions on test set
         confusion_matrix = map2(.x = test, .y = test_pred,  # get confusion matrix
                                 .f = function(x, y) caret::confusionMatrix(x$stem, y))
  )

# look at confusion matrix for test set
conf_mat_linear <- svm_linear_model %>% 
  pluck("confusion_matrix", 1)
conf_mat_linear

# get accuracy and test error
accu_linear <- conf_mat_linear$overall[["Accuracy"]]
accu_linear

error_linear <- 1 - accu_linear
error_linear


# Radial
svm_radial_cv <- svm_dat %>%
  mutate(train_cv = map(.x = train, 
                        .f = function(x) tune(svm, stem ~ ., data = x, kernel = "radial", 
                                              ranges = list(cost = c(0.01, 0.1, 1, 5, 10)))
  ))

radial_params <- svm_radial_cv %>%
  pluck("train_cv", 1)  %>%
  pluck("best.parameters")

radial_params

# fit the model with best cost: radial_params cost = 5
svm_radial_model <- svm_dat %>%
  mutate(model_fit = map(.x = train, # fit the model
                         .f = function(x) svm(stem ~ ., data = x, cost = radial_params$cost, kernel = "radial")), 
         test_pred = map2(model_fit, test, predict), # get predictions on test set
         confusion_matrix = map2(.x = test, .y = test_pred,  # get confusion matrix
                                 .f = function(x, y) caret::confusionMatrix(x$stem, y))
  )

# look at confusion matrix for test set
conf_mat_radial <- svm_radial_model %>% 
  pluck("confusion_matrix", 1)
conf_mat_radial

# get accuracy and test error
accu_radial <- conf_mat_radial$overall[["Accuracy"]]
accu_radial

error_radial <- 1 - accu_radial
error_radial


## Poly
svm_poly_cv <- svm_dat %>%
  mutate(train_cv = map(.x = train, 
                        .f = function(x) tune(svm, stem ~ ., data = x, kernel = "polynomial", 
                                              ranges = list(cost = c(0.01, 0.1, 1, 5, 10))
                        )
  ))

poly_params <- svm_poly_cv %>%
  pluck("train_cv", 1)  %>%
  pluck("best.parameters")

poly_params

# fit the model with best cost: poly_params cost = 10
svm_poly_model <- svm_dat %>%
  mutate(model_fit = map(.x = train, # fit the model
                         .f = function(x) svm(stem ~ ., data = x, cost = poly_params$cost, kernel = "polynomial")), 
         test_pred = map2(model_fit, test, predict), # get predictions on test set
         confusion_matrix = map2(.x = test, .y = test_pred,  # get confusion matrix
                                 .f = function(x, y) caret::confusionMatrix(x$stem, y))
  )

# look at confusion matrix for test set
conf_mat_poly <- svm_poly_model %>% 
  pluck("confusion_matrix", 1)
conf_mat_poly

# get accuracy and test error
accu_poly <- conf_mat_poly$overall[["Accuracy"]]
accu_poly

error_poly <- 1 - accu_poly
error_poly

### boosted
# Helper function to convert tibble to DMatrix
xgb_matrix <- function(dat, outcome, exclude_vars){
  # Sanitize input: check that data has factors, not characters
  dat_types <- dat %>% map_chr(class)
  outcome_type <- class(dat[[outcome]])
  if("chr" %in% dat_types){
    # If we need to re-code, leave that outside of the function
    print("You must encode characters as factors.")
    return(NULL)
  } else {
    lab <- as.integer(dat[[outcome]]) - 1
    # Make our DMatrix
    mat <- dat %>% dplyr::select(-outcome, -all_of(exclude_vars)) %>% # encode on full boston df
      onehot::onehot() %>% # use onehot to encode variables
      predict(dat) # get OHE matrix
    return(xgb.DMatrix(data = mat, 
                       label = lab))
  }
}



# Helper function to get error (either mse or misclass)
xg_error <- function(model, test_mat, metric = "mse"){
  # Get predictions and actual values
  preds = predict(model, test_mat)
  vals = getinfo(test_mat, "label")
  if(metric == "mse"){
    # Compute MSE if that's what we need
    err <- mean((preds - vals)^2)
  } else if(metric == "misclass") {
    # Otherwise, get the misclass rate
    err <- mean(preds != vals)
  }
  return(err)
}

# 5-fold
train_5fold <- train %>% 
  crossv_kfold(5, id = "fold") %>% 
  mutate(
    train = map(train, as_tibble),
    test = map(test, as_tibble)
  )

# Model fitting
xg <- train_5fold %>% 
  crossing(learn_rate = 10^seq(-10, -.1, length.out = 10), # tune learning rate
           nrounds = c(50, 100, 500)) %>% # tune number of trees
  mutate(# Build xgb Dmatrices for training and test set
    train_mat = map(train, xgb_matrix, outcome = "stem", exclude_vars = NULL), 
    test_mat = map(test, xgb_matrix, outcome = "stem", exclude_vars = NULL),
    # Fit models to each learning rate
    xg_model = pmap(list(x = train_mat, y = learn_rate, z = nrounds),
                    .f = function(x, y, z){xgb.train(params = list(eta = y, # set learning rate
                                                                   depth = 10, # tree depth, can tune
                                                                   objective = "multi:softmax", # output labels
                                                                   num_class = 3), # multiple classes problem 
                                                     data = x, 
                                                     nrounds = z, # set number of trees
                                                     silent = TRUE)}), 
    # Get training and test error
    xg_train_misclass = map2(xg_model, train_mat, xg_error, metric = "misclass"),
    xg_test_misclass = map2(xg_model, test_mat, xg_error, metric = "misclass") 
  )

# Get best model
xg_learn_rate <- xg %>% 
  group_by(learn_rate, nrounds) %>% 
  summarize(
    xg_test_misclass = mean(unlist(xg_test_misclass))
  ) %>% 
  arrange(xg_test_misclass) %>% 
  pluck(1, 1)

xg_nrounds <- xg %>% 
  group_by(learn_rate, nrounds) %>% 
  summarize(
    xg_test_misclass = mean(unlist(xg_test_misclass))
  ) %>% 
  arrange(xg_test_misclass) %>% 
  pluck(2, 1)

xg_mod <- xg %>% 
  filter(learn_rate == xg_learn_rate,
         nrounds == xg_nrounds) %>% 
  pluck("xg_model", 1)

# Save model
xgb.save(xg_mod, "xg_mod")

# Variable importance
vip(xg_mod)

# Load model
xg_mod <- xgb.load("xg_mod")

# Candidate model
xg_class <- tibble(
  train = train %>% list(),
  test  = test %>% list()
) %>%
  mutate(# Build xgb Dmatrices for training and test set
    train_mat = map(train, xgb_matrix, outcome = "stem", exclude_vars = NULL), 
    test_mat = map(test, xgb_matrix, outcome = "stem", exclude_vars = NULL)
  ) %>% 
  mutate(
    xg_model = list(xg_mod)
  ) %>% 
  pivot_longer(cols = c(-test, -train, -test_mat, -train_mat), names_to = "method", values_to = "fit") 

# Test error
xg_misclass <- xg_class %>% 
  mutate(
    test_misclass = map2_dbl(fit, test_mat, xg_error, metric = "misclass")
  ) %>% 
  select(method, test_misclass) 

xg_misclass

## Random Forests
# Helper function to get misclass rate for bagging and random forests
misclass_ranger <- function(model, test, outcome){
  # check if test is a tibble
  if(!is_tibble(test)){
    test <- test %>% as_tibble()
  }
  # create predicted values
  preds <- predict(model, test)$predictions
  # misclassification rate
  misclass <- mean(test[[outcome]] != preds)
  return(misclass)
}

# Model fitting
rf_5fold <- train_5fold %>% 
  crossing(mtry = 1:(ncol(train) - 1)) %>% # exclude stem
  mutate(
    model = map2(.x = train, .y = mtry,
                 .f = function(x, y){ranger(stem ~ . , # exclude stem itselfs
                                            mtry = y,
                                            data = x,
                                            splitrule = "gini", # use the gini index
                                            importance = "impurity")}),
    # get training, testing, OOB error
    fold_train_misclass = map2(model, train, misclass_ranger, outcome = "stem"),
    fold_test_misclass = map2(model, test, misclass_ranger, outcome = "stem"),
    fold_oob_misclass = map(.x = model, 
                            .f = function(x){x[["prediction.error"]]})
  )

# Table
rf_5fold %>% 
  group_by(mtry) %>% 
  summarize(
    train_misclass = mean(unlist(fold_train_misclass)),
    test_misclass = mean(unlist(fold_test_misclass)),
    oob_misclass = mean(unlist(fold_oob_misclass))
  ) %>% 
  arrange(test_misclass)

# Candidate model
rf_class <- tibble(
  train = train %>% list(),
  test  = test %>% list()
) %>%
  mutate(
    rf_mtry2 = map(train, ~ ranger(stem ~ . , 
                                   data = .x, 
                                   mtry = 2,
                                   importance = "impurity", 
                                   splitrule = "gini")),
    rf_mtry3 = map(train, ~ ranger(stem ~ . , 
                                   data = .x, 
                                   mtry = 3,
                                   importance = "impurity", 
                                   splitrule = "gini")),
    bagging_mtry15 = map(train, ~ ranger(stem ~ . , 
                                         data = .x, 
                                         mtry = 15,
                                         importance = "impurity", 
                                         splitrule = "gini"))
  ) %>% 
  pivot_longer(cols = c(-test, -train), names_to = "method", values_to = "fit")

# Examine variable importance
rf_mtry2 = ranger(stem ~ . , 
                  data = train, 
                  mtry = 2,
                  importance = "impurity", 
                  splitrule = "gini",
                  probability = TRUE)

vip(rf_mtry2)

bagging_mtry15 = ranger(stem ~ . , 
                        data = train, 
                        mtry = 15,
                        importance = "impurity", 
                        splitrule = "gini",
                        probability = TRUE)

vip(bagging_mtry15)

# Test error
rf_misclass <- rf_class %>% 
  mutate(
    test_misclass = map2_dbl(fit, test, misclass_ranger, outcome = "stem")
  ) %>% 
  select(method, test_misclass) 
rf_misclass


### Multinomial Logistic Regression
mlr <- multinom(stem ~ ., data = train)

summary(mlr)

pred_stem <- predict(mlr, test)

mlr_misclass <- test %>% 
  mutate(
    error = pred_stem != test$stem
  ) %>% 
  pull(error) %>% 
  mean()

mlr_misclass

### Ridge and Lasso
## Ridge
# lambda grid to search -- use for ridge regression (200 values)
lambda_grid <- 10^seq(-2, 10, length = 200)

# ridge regression: 10-fold cv
ridge_cv <- train %>% 
  cv.glmnet(
    formula = stem ~ ., 
    family = "multinomial",
    type.multinomial = "grouped",
    data = ., 
    alpha = 0, 
    nfolds = 10,
    lambda = lambda_grid
  )

# Check plot of cv error
plot(ridge_cv)

# ridge's best lambdas
ridge_lambda_min <- ridge_cv$lambda.min
ridge_lambda_1se <- ridge_cv$lambda.1se

# Candidate model
ridge_class <- tibble(
  train = train %>% list(),
  test  = test %>% list()
) %>%
  mutate(
    ridge_min = map(train, ~ glmnet(stem ~ ., family = "multinomial", type.multinomial = "grouped",
                                    data = .x,
                                    alpha = 0, lambda = ridge_lambda_min)),
    ridge_1se = map(train, ~ glmnet(stem ~ ., family = "multinomial", type.multinomial = "grouped",
                                    data = .x,
                                    alpha = 0, lambda = ridge_lambda_1se)) 
  ) %>% 
  pivot_longer(cols = c(-test, -train), names_to = "method", values_to = "fit")

# Inspect/compare model coefficients 
ridge_class %>% 
  pluck("fit") %>% 
  map( ~ coef(.x) %>% 
         as.matrix() %>% 
         as.data.frame() %>% 
         rownames_to_column("name")) %>%
  reduce(full_join, by = "name") %>% 
  mutate_if(is.double, ~ if_else(. == 0, NA_real_, .)) %>% 
  rename(ridge_min = s0.x,
         ridge_1se = s0.y) %>% 
  knitr::kable(digits = 3)

# Test Error
ridge_min_pred <- predict(ridge_class$fit[[1]], test, type = "class")

ridge_min_misclass <- test %>% 
  mutate(
    error = ridge_min_pred != test$stem
  ) %>% 
  pull(error) %>% 
  mean()

ridge_min_misclass

ridge_1se_pred <- predict(ridge_class$fit[[2]], test, type = "class")

ridge_1se_misclass <- test %>% 
  mutate(
    error = ridge_1se_pred != test$stem
  ) %>% 
  pull(error) %>% 
  mean()

ridge_1se_misclass

# Lasso
# Lasso regression: 10-fold cv
lasso_cv <- train %>% 
  cv.glmnet(
    formula = stem ~ ., 
    family = "multinomial",
    type.multinomial = "grouped",
    data = ., 
    alpha = 1, 
    nfolds = 10,
    lambda = lambda_grid
  )

# Check plot of cv error
plot(lasso_cv)

# lasso's best lambdas
lasso_lambda_min <- lasso_cv$lambda.min
lasso_lambda_1se <- lasso_cv$lambda.1se

# Candidate model
lasso_class <- tibble(
  train = train %>% list(),
  test  = test %>% list()
) %>%
  mutate(
    ridge_min = map(train, ~ glmnet(stem ~ ., family = "multinomial", type.multinomial = "grouped",
                                    data = .x,
                                    alpha = 1, lambda = ridge_lambda_min)),
    ridge_1se = map(train, ~ glmnet(stem ~ ., family = "multinomial", type.multinomial = "grouped",
                                    data = .x,
                                    alpha = 1, lambda = ridge_lambda_1se)) 
  ) %>% 
  pivot_longer(cols = c(-test, -train), names_to = "method", values_to = "fit")

# Inspect/compare model coefficients 
lasso_class %>% 
  pluck("fit") %>% 
  map( ~ coef(.x) %>% 
         as.matrix() %>% 
         as.data.frame() %>% 
         rownames_to_column("name")) %>%
  reduce(full_join, by = "name") %>% 
  mutate_if(is.double, ~ if_else(. == 0, NA_real_, .)) %>% 
  rename(ridge_min = s0.x,
         ridge_1se = s0.y) %>% 
  knitr::kable(digits = 3)

# Test Error
lasso_min_pred <- predict(lasso_class$fit[[1]], test, type = "class")

lasso_min_misclass <- test %>% 
  mutate(
    error = lasso_min_pred != test$stem
  ) %>% 
  pull(error) %>% 
  mean()

lasso_min_misclass

lasso_1se_pred <- predict(lasso_class$fit[[2]], test, type = "class")

lasso_1se_misclass <- test %>% 
  mutate(
    error = lasso_1se_pred != test$stem
  ) %>% 
  pull(error) %>% 
  mean()

ridge_1se_misclass

rl_misclass <- tibble(
  "method" = c("ridge_min", "ridge_1se", "lasso_min", "lasso_1se"),
  "test_misclass" = c(ridge_min_misclass, ridge_1se_misclass, lasso_min_misclass, lasso_1se_misclass)
) 

rl_misclass

### Neural Network
# load data
train_data <- train %>% 
  select(-stem) %>% 
  as.data.table() %>% 
  one_hot() %>% # one hot encoding categorical variables
  data.matrix()

train_targets <- train %>% 
  pull(stem) %>% 
  as.numeric() - 1 # label value should be 0, 1, 2

test_data <- test %>% 
  select(-stem) %>% 
  as.data.table() %>% 
  one_hot() %>% 
  data.matrix()

test_targets <- test %>% 
  pull(stem) %>% 
  as.numeric() - 1

# normalize data
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data)
test_data <- scale(test_data, center = mean, scale = std)

# function to build network
build_model <- function() {
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", 
                input_shape = dim(train_data)[[2]]) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 3, activation = "softmax") 
  
  model %>% compile(
    optimizer = "rmsprop", 
    loss = "sparse_categorical_crossentropy", # integer labels
    metrics = c("accuracy")
  )
}

# k-fold cv
k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(1:length(indices), breaks = k, labels = FALSE)
num_epochs <- 100
all_accuracy_histories <- NULL
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  # Build the Keras model (already compiled)
  model <- build_model()
  
  # Train the model (in silent mode, verbose=0)
  history <- model %>% fit(
    partial_train_data, partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs, batch_size = 1, verbose = 0
  )
  accuracy_history <- history$metrics$val_accuracy
  all_accuracy_histories <- rbind(all_accuracy_histories, accuracy_history)
}

average_accuracy_history <- data.frame(
  epoch = seq(1:ncol(all_accuracy_histories)),
  validation_accuracy = apply(all_accuracy_histories, 2, mean)
)

# save average_accuracy_history
saveRDS(average_accuracy_history, "average_accuracy_history")

# load average_accuracy_history
average_accuracy_history <- readRDS("average_accuracy_history")

# plot
ggplot(average_accuracy_history, aes(x = epoch, y = validation_accuracy)) + 
  geom_line()

ggplot(average_accuracy_history, aes(x = epoch, y = validation_accuracy)) + 
  geom_smooth()

# train the final model on the entire training set
model <- build_model()

model %>% 
  fit(train_data, train_targets,
      epochs = 90, batch_size = 16, verbose = 0)

# save model
model %>% save_model_tf("neural_net")

# load model
neural_net <- load_model_tf("neural_net")

# evaluate on test set
result <- neural_net %>% 
  evaluate(test_data, test_targets)
result

# test misclass
neural_net_misclass <- 1 - result$accuracy
neural_net_misclass <- tibble(
  "method" = "neural_net",
  "test_misclass" = neural_net_misclass
)

neural_net_misclass %>% 
  knitr::kable(digits = 3)

## Test errors combined and organized
rf_misclass %>% 
  bind_rows(mlg_misclass) %>% 
  bind_rows(xg_misclass) %>% 
  bind_rows(svm_misclass) %>% 
  bind_rows(rl_misclass) %>% 
  bind_rows(neural_net_misclass) %>% 
  arrange(test_misclass) %>% 
  knitr::kable(digits = 3)

# Inspect lasso min
cf_lasso_min <- coef(lasso_class$fit[[1]]) %>% 
  lapply(as.matrix) %>% 
  Reduce(cbind, x = .) 

colnames(cf_lasso_min) <- c("Undecided", "Non-STEM", "STEM")

cf_lasso_min[apply(cf_lasso_min[,-1], 1, function(x) !all(x==0)),] %>% 
  knitr::kable(digits = 3)
