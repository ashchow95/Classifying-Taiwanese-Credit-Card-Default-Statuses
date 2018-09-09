#################################################
#                    Packages                   #
#################################################

library(caret)
library(caTools) 
library(doParallel)
library(kernlab)
library(klaR)
library(MASS)
library(modelr)
library(pROC)
library(rpart)
library(randomForest)

#################################################
#                 Data Importing                #
#################################################
## Read Data
setwd("C:/Users/Alaric/Desktop/WINTER 2018/STAT 441/Project Finale")
data <- read.csv('default of credit card clients.csv')

#################################################
#                 Data Processing               #
#################################################
## Defining factors that will be characteristics
factor_vars <- c('SEX','EDUCATION','MARRIAGE','default.payment.next.month','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6')
data[factor_vars] <- lapply(data[factor_vars], function(x) as.factor(x))
levels(data$default.payment.next.month) <- list(no="0", yes="1") 
# Binning ages
data <- transform(data, MARRIAGE_BIN = ifelse(MARRIAGE==0, 3, MARRIAGE))  # marriage = 0 becomes 3
data <- transform(data, EDUCATION_BIN = ifelse(EDUCATION==5 | EDUCATION==6, 4, EDUCATION))  # education = 5 or 6 becomes 4
age_int <- seq(from=20, to=80, by=5)
#data$AGE_INT <- cut(data$AGE,age_int) # for reference, if you wanted to see which age interval AGE_BIN fell into you can uncomment this
data$AGE_BIN <- cut(data$AGE,age_int,labels=FALSE) # age banded into intervals of 5 years
#data_mod <- subset(data, select=-c(AGE_INT,ID)) #can use this line to drop certain columns in select=-c(.) if the dataset is too large 

# Reduced Model Analysis
data_reduced <- step(glm(default.payment.next.month ~ ., data = data, family = binomial),
                     direction = 'both')
data_reduced_col = c(all.vars(formula(data_reduced)[-2]), 'default.payment.next.month')
data_reduced_final = data[data_reduced_col]

#################################################
#                 Data Splitting                #
#################################################
# divide into train and test set
set.seed(5309)
# 80% Training 20% Testing
split = sample.split(data$default.payment.next.month, SplitRatio = 0.8)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

train_set_reduce = subset(data_reduced_final, split == TRUE)
test_set_reduce = subset(data_reduced_final, split == FALSE)

#################################################
#                 Data Tuning                   #
#################################################
# https://topepo.github.io/caret/model-training-and-tuning.html#control

# Tuning Control: 
# TwoClassSummary - Caret will find the best hyperparmeter in regards to sensitivity, specificity, and area under ROC curve.
# ClassProbs - Compute the class probability
# Goal is to find the best hyper parameter for each model before we train
tuner <- trainControl(summaryFunction = twoClassSummary, classProbs = TRUE)


# Evaluation Control:
# We are using Repeated K-Fold CV, 10 folds repeated 5 times.
# Still use ClassProb and calculate TwoClassSummary
# Save Prediction to pass onto predict later
evaluater <- trainControl(method = "repeatedcv" , number = 10, repeats = 5, 
                          savePredictions = T, summaryFunction = twoClassSummary, 
                          classProbs = T)

# Tuning 
# Tune Length = 2. Caret will find optimal parameter and evaluate two of them
train(default.payment.next.month ~ ., 
      metric = "ROC", preProc = c("center", "scale"), 
      tuneLength = 3, trControl = tuner, data = train_set, method = "rf")

train(default.payment.next.month ~ ., 
      metric = "ROC", preProc = c("center", "scale"), 
      tuneLength = 3, trControl = tuner, data = train_set_reduce, method = "rf")

# Alternatively, for multiple parameter models, use 
# grid <- expand.grid(size=c(5,10,20,50), k=c(1,2,3,4,5))
# tuneGrid = grid replaces tuneLength in the above code
# 24000 samples
# 27 predictor
# 2 classes: 'no', 'yes' 
# 
# Pre-processing: centered (86), scaled (86) 
# Resampling: Bootstrapped (25 reps) 
# Summary of sample sizes: 24000, 24000, 24000, 24000, 24000, 24000, ... 
# Resampling results across tuning parameters:
#   
#   mtry  ROC        Sens       Spec     
# 2    0.7405772  0.9797134  0.1701797
# 44    0.7609169  0.9420768  0.3661727
# 86    0.7569522  0.9400491  0.3650296
# 
# ROC was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 44. 

#for train_set_reduced
# Random Forest 
# 
# 24000 samples
# 17 predictor
# 2 classes: 'no', 'yes' 
# 
# Pre-processing: centered (76), scaled (76) 
# Resampling: Bootstrapped (25 reps) 
# Summary of sample sizes: 24000, 24000, 24000, 24000, 24000, 24000, ... 
# Resampling results across tuning parameters:
#   
#   mtry  ROC        Sens       Spec     
# 2    0.7372549  0.9793354  0.1745063
# 39    0.7516327  0.9366579  0.3700789
# 76    0.7464290  0.9330469  0.3699205
# 
# ROC was used to select the optimal model using the largest value.
# The final value used for the model was mtry = 39.
#################################################
#                 Data Training                 #
#################################################
# Above code will train to find the best hyper parameter to use in the model
# Please follow the following convention model_METHODNAME

model_rf <- train(default.payment.next.month ~ ., data = train_set,
                   metric = "ROC", preProc = c("center", "scale"),
                   tuneGrid = data.frame(mtry=44), trControl = evaluater, method = "rf")

saveRDS(model_rf, "model_rf.rds")

model_rf_reduced <- train(default.payment.next.month ~ ., data = train_set_reduce,
                   metric = "ROC", preProc = c("center", "scale"),
                   tuneGrid = data.frame(mtry=39), trControl = evaluater, method = "rf")

saveRDS(model_rf_reduced, "model_rf_reduced.rds")
 
#################################################
#                Data Prediction                #
#################################################
predict_rf <- predict(model_rf, newdata = test_set)
predict_rf_reduce <- predict(model_rf_reduced, newdata = test_set_reduce)

saveRDS(predict_rf, "model_rf_predict.rds")
saveRDS(predict_rf_reduce, "model_rf_reduce_predict.rds")


model_nnets <- readRDS("model_nnet.rds")
model_nnet_reduced <- readRDS("model_nnet_reduced.rds")

predict_nnets <- predict(model_nnets, newdata = test_set)
predict_nnets_reduce <- predict(model_nnet_reduced, newdata = test_set_reduce)

cm <- confusionMatrix(predict_nnets_reduce, test_set_reduce$default.payment.next.month)
# Error Rate Calc

err_count = 0
for (i in 1:nrow(test_set))
{
  if (test_set$default.payment.next.month[i]!= predict_rf[i]) {err_count=err_count+1}
}

err_rate_test_rf = err_count/nrow(test_set);err_rate_test_rf
1-err_rate_test_rf

err_count = 0
for (i in 1:nrow(test_set_reduce))
{
  if (test_set_reduce$default.payment.next.month[i]!= predict_rf_reduce[i]) {err_count=err_count+1}
}

err_rate_test_rf_reduce = err_count/nrow(test_set_reduce);err_rate_test_rf_reduce
1-err_rate_test_rf_reduce
## Give me suggestions here as to how to demonstrate data. Confusion matrix? Help

                 