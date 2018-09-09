# STAT 441 Final Project Code
# Alaric Chow, ID 20517917
# Karan Mehta, ID 20512167
# Yoon Soo Shin, ID 20516955
# Melody Tam,  ID 20518019

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
#               Data Visualization              #
#################################################

# FIGURES

# Figure 1: Histograms of Pre-Processed Variables
data2 <- read.csv('default of credit card clients.csv')
par(mfrow=c(1,3))
hist(data2$AGE,main='AGE',xlab="",ylab="Frequency", breaks=60,col='lightskyblue')
hist(data2$EDUCATION,main='EDUCATION',xlab="",ylab="Frequency",col='lightskyblue')
hist(data2$MARRIAGE,main='MARRIAGE',xlab="",ylab="Frequency",col='lightskyblue')
mtext("Figure 1: Histograms of Pre-Processed Variables", side = 1, line = -1.5, outer = TRUE)

# Figure 2: Correlation Plot of Bill Statement and Payment Variables 
par(mfrow=c(1,1))
data <- read.csv('default of credit card clients.csv')
nums <- sapply(data, is.numeric)
library(corrplot)
corrplot.mixed(cor(data[,nums]), lower="circle", upper="color", 
               tl.pos="lt", diag="n", order="hclust", hclust.method="complete")
mtext("Figure 2: Correlation Plot of Bill Statement and Payment Variables", side = 1, line = -1, outer = TRUE)

# Figure 3: Box Plots of Population Defining Variables
library(ggplot2)   # For plotting
library(gridExtra)
library(grid)

# Look at correlation between defaulting / limit balance / age / sex / marriage / education

data <- read.csv('default_data.csv')
factor_vars <- c('SEX','EDUCATION','MARRIAGE','default.payment.next.month')

data$AGE<-cut(data$AGE, breaks = c( 10, 30,50,100), labels = c("10-30", "30-50","50-100"))
data$SEX<-cut(data$SEX, 2,labels = c("Female","Male"))
data$MARRIAGE<-cut(data$MARRIAGE, 4,labels = c("Married","Single","Divorced","Other"))
data[factor_vars] <- lapply(data[factor_vars], function(x) as.factor(x))

g_legend <-function(a.gplot){
  tmp <- ggplot_gtable(ggplot_build(a.gplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend) }
p0 = ggplot(data=data,mapping = aes(x=AGE,y=data$LIMIT_BAL,fill=default.payment.next.month)) + geom_boxplot()  + ylab("LIMIT_BAL")
leg = g_legend(p0)
p1 = ggplot(data=data,mapping = aes(x=AGE,y=data$LIMIT_BAL,fill=default.payment.next.month)) + geom_boxplot() + guides(fill=FALSE) + ylab("LIMIT_BAL")
p2 = ggplot(data=data,mapping = aes(x=EDUCATION,y=data$LIMIT_BAL,fill=default.payment.next.month)) + geom_boxplot() + guides(fill=FALSE) + ylab("LIMIT_BAL")
p3 = ggplot(data=data,mapping = aes(x=SEX,y=data$LIMIT_BAL,fill=default.payment.next.month)) + geom_boxplot() + guides(fill=FALSE) + ylab("LIMIT_BAL")
p4 = ggplot(data=data,mapping = aes(x=MARRIAGE,y=data$LIMIT_BAL,fill=default.payment.next.month)) + geom_boxplot() + guides(fill=FALSE)+ theme(legend.position="bottom") + ylab("LIMIT_BAL")
# plotting box plots
grid.arrange(p1,p2,p3,p4,leg,bottom = textGrob("Figure 3: Box Plots of Population Defining Variables",gp=gpar(fontsize=12,font=3)),nrow=1, ncol=5)

#################################################
#                 Data Processing               #
#################################################

data <- read.csv('default of credit card clients.csv')
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

## RF ##
train(default.payment.next.month ~ ., 
      metric = "ROC", preProc = c("center", "scale"), 
      tuneLength = 3, trControl = tuner, data = train_set, method = "rf")

train(default.payment.next.month ~ ., 
      metric = "ROC", preProc = c("center", "scale"), 
      tuneLength = 3, trControl = tuner, data = train_set_reduce, method = "rf")

# Optimal mtry was 44 and 39

## KNN ##
train(default.payment.next.month ~ ., 
      metric = "ROC", preProc = c("center", "scale"), 
      tuneLength = 3, trControl = tuner, data = train_set, method = "knn")

#k-Nearest Neighbors 

#24000 samples
#27 predictor
#2 classes: 'no', 'yes' 

#Pre-processing: centered (86), scaled (86) 
#Resampling: Bootstrapped (25 reps) 
#Summary of sample sizes: 24000, 24000, 24000, 24000, 24000, 24000, ... 
#Resampling results across tuning parameters:
  
#  k  ROC        Sens       Spec     
#5  0.6799558  0.8754009  0.3756420
#7  0.6956607  0.8960229  0.3667920
#9  0.7056040  0.9105915  0.3601912


train(default.payment.next.month ~ ., 
      metric = "ROC", preProc = c("center", "scale"), 
      tuneLength = 3, trControl = tuner, data = data_reduced_final, method = "knn")

# k-Nearest Neighbors 

#30000 samples
#17 predictor
#2 classes: 'no', 'yes' 

#Pre-processing: centered (76), scaled (76) 
#Resampling: Bootstrapped (25 reps) 
#Summary of sample sizes: 30000, 30000, 30000, 30000, 30000, 30000, ... 
#Resampling results across tuning parameters:
  
#  k  ROC        Sens       Spec     
# 5  0.6813612  0.8764144  0.3760076
# 7  0.6971551  0.8976877  0.3682093
# 9  0.7073394  0.9117669  0.3614223

# ROC was used to select the optimal model using the largest value.
# The final value used for the model was k = 9.

## Neural Network
grid <- expand.grid(size=c(5,10,20,50), decay=c(0.1,0.2,0.3,0.4,0.5))

# Tuning 
train(default.payment.next.month ~ ., 
      metric = "ROC", preProc = c("center", "scale"), 
      tuneGrid = grid, trControl = tuner, data = train_set, method = "nnet")
train(default.payment.next.month ~ ., 
      metric = "ROC", preProc = c("center", "scale"), 
      tuneGrid = grid, trControl = tuner, data = train_set_reduce, method = "nnet")

# SVM
train(default.payment.next.month ~ ., 
      metric = "ROC", preProc = c("center", "scale"), 
      tuneLength = 3, trControl = tuner, data = train_set, method = "svmLinear")
train(default.payment.next.month ~ ., 
      metric = "ROC", preProc = c("center", "scale"), 
      tuneLength = 3, trControl = tuner, data = train_set_reduce, method = "svmLinear")

# Optimal Cost for both of them is C = 0.5


#################################################
#                 Data Training                 #
#################################################

# Logistic Regression
model_lr <- train(default.payment.next.month ~ ., data = train_set, 
                  metric = "ROC", trControl = evaluater, method = "glm", family = binomial)
saveRDS(model_lr, "model_lr.rds")
model_lr_reduced <- train(default.payment.next.month ~ ., data = train_set_reduce, 
                  metric = "ROC", trControl = evaluater, method = "glm", family = binomial)
saveRDS(model_lr_reduced, "model_lr_reduced.rds")

# Random Forest
model_rf <- train(default.payment.next.month ~ ., data = train_set,
                  metric = "ROC", preProc = c("center", "scale"),
                  tuneGrid = data.frame(mtry=44), trControl = evaluater, method = "rf")
saveRDS(model_rf, "model_rf.rds")
model_rf_reduced <- train(default.payment.next.month ~ ., data = train_set_reduce,
                          metric = "ROC", preProc = c("center", "scale"),
                          tuneGrid = data.frame(mtry=39), trControl = evaluater, method = "rf")
saveRDS(model_rf_reduced, "model_rf_reduced.rds")


# KNN
model_KNN <- train(default.payment.next.month ~ ., data = train_set,
                   metric = "ROC", preProc = c("center", "scale"),
                   tuneGrid = data.frame(k = 9), trControl = evaluater, method = "knn")

saveRDS(model_KNN, "model_KNN.rds")

model_KNN_reduced <- train(default.payment.next.month ~ ., data = train_set_reduce,
                   metric = "ROC", preProc = c("center", "scale"),
                   tuneGrid = data.frame(k = 9), trControl = evaluater, method = "knn")

# saveRDS(model_KNN_reduced, "model_KNN_reduced.rds")

# Neural Network

model_nnet <- train(default.payment.next.month ~ ., data = train_set,
                    metric = "ROC", preProc = c("center", "scale"),
                    tuneGrid = data.frame(size = 5, decay = 0.5
                    ), trControl = evaluater, method = "nnet")
saveRDS(model_nnet, "model_nnet.rds")
model_nnet_reduced <- train(default.payment.next.month ~ ., data = train_set_reduce,
                            metric = "ROC", preProc = c("center", "scale"),
                            tuneGrid = data.frame(size = 5, decay = 0.5
                            ), trControl = evaluater, method = "nnet")

saveRDS(model_nnet_reduced, "model_nnet_reduced.rds")

# SVM
model_svm <- train(default.payment.next.month ~ ., data = train_set,
                   metric = "ROC", preProc = c("center", "scale"),
                   tuneGrid = data.frame(C = 0.5), trControl = evaluater, method = "svmLinear")

saveRDS(model_svm, "model_svm.rds")
model_svm_Reduced <-train(default.payment.next.month ~ ., data = train_set_reduce,
                          metric = "ROC", preProc = c("center", "scale"),
                          tuneGrid = data.frame(C = 0.5), trControl = evaluater, method = "svmLinear")
saveRDS(model_svm_Reduced, "model_svm_Reduced.rds")

# For future runs
model_lr <- readRDS("model_lr.rds")
model_lr_reduced <- readRDS("model_lr_reduced.rds")
model_svm <- readRDS("model_svm.rds")
model_svm_Reduced <- readRDS("model_svm_Reduced.rds")
model_nnet <- readRDS("model_nnet.rds")
model_nnet_reduced <- readRDS("model_nnet_reduced.rds")
model_KNN <-  readRDS("model_KNN.rds")
model_KNN_reduced <- readRDS("model_KNN_reduced.rds")
model_rf <- readRDS("model_rf.rds")
model_rf_reduced <- readRDS("model_rf_reduced.rds")

#################################################
#                Model Analysis                 #
#################################################
result <-list("Full KNN"=model_KNN, 
              "Full Logistic Regression"= model_lr,
              "Full Random Forest" = model_rf,
              "Full Neural Network" = model_nnet,
              "Full SVM Linear" = model_svm,
              "Reduced KNN" = model_KNN_reduced,
              "Reduced Logistic Regression" = model_lr_reduced,
              "Reduced Random Forest" = model_rf_reduced,
              "Reduced Neural Network" = model_nnet_reduced,
              "Reduced SVM Linear" = model_svm_Reduced)

summary(resamples(result))

## Neural Network has best performing AUC and the reduced version as well as the full version is very similar. Therefore we will assume the simpler model.


roc_lr <- roc(model_lr$pred$obs, model_lr$pred$no)
roc_lr_reduced <- roc(model_lr_reduced$pred$obs, model_lr_reduced$pred$no)
roc_KNN <- roc(model_KNN$pred$obs, model_KNN$pred$no)
roc_KNN_reduced <-  roc(model_KNN_reduced$pred$obs, model_KNN_reduced$pred$no)
roc_rf <- roc(model_rf$pred$obs, model_rf$pred$no)
roc_rf_reduced <- roc(model_rf_reduced$pred$obs, model_rf_reduced$pred$no)
roc_nn <- roc(model_nnet$pred$obs, model_nnet$pred$no)
roc_nn_reduced <- roc(model_nnet_reduced$pred$obs, model_nnet_reduced$pred$no)
roc_svm <- roc(model_svm$pred$obs, model_svm$pred$no)
roc_svm_reduced <- roc(model_svm_Reduced$pred$obs, model_svm_Reduced$pred$no)

# Plotting ROC graphs
par(mfrow=c(1,2))

par(pty="s")
plot(roc_lr, col = 1, lty = 1, main = "ROC Curve for Full Models")
plot(roc_KNN, col = 2, lty = 1, add = TRUE)
plot(roc_rf, col = 3, lty = 1, add = TRUE)
plot(roc_nn, col = 4, lty = 1, add = TRUE)
plot(roc_svm, col = 5, lty = 1, add = TRUE)
grid(lwd = 2) # grid only in y-direction
legend("bottomright", legend=c("LR", "KNN", "RF", "NN", "SVM"),
       col=c(1,2,3,4,5), lwd=5)

par(pty="s")
plot(roc_lr_reduced, col = 1, lty = 1, main = "ROC Curve for Reduced Models")
plot(roc_KNN_reduced, col = 2, lty = 1, add = TRUE)
plot(roc_rf_reduced, col = 3, lty = 1, add = TRUE)
plot(roc_nn_reduced, col = 4, lty = 1, add = TRUE)
plot(roc_svm_reduced, col = 5, lty = 1, add = TRUE)
grid(lwd = 2) # grid only in y-direction
legend("bottomright", legend=c("LR", "KNN", "RF", "NN", "SVM"),
       col=c(1,2,3,4,5), lwd=5)
#################################################
#                Data Prediction                #
#################################################

# Random Forest
predict_rf <- predict(model_rf, newdata = test_set)
predict_rf_reduce <- predict(model_rf_reduced, newdata = test_set_reduce)
# saveRDS(predict_rf, "model_rf_predict.rds")
# saveRDS(predict_rf_reduce, "model_rf_reduce_predict.rds")

# K-Nearest Neighbour
predict_KNN <- predict(model_KNN, newdata = test_set)
predict_KNN_reduce <- predict(model_KNN_reduced, newdata = test_set_reduce)
# saveRDS(predict_KNN, "predict_KNN.rds")
# saveRDS(predict_KNN_reduce, "predict_KNN_reduce.rds")

# Neural Network
predict_NN <- predict(model_nnet, newdata = test_set)
predict_NN_reduce <- predict(model_nnet_reduced, newdata = test_set_reduce)
# saveRDS(predict_NN, "predict_NN.rds")
# saveRDS(predict_NN_reduce, "predict_NN_reduce.rds")

# LR
predict_LR <- predict(model_lr_reduced, newdata = test_set)
predict_LR_reduce <- predict(model_lr_reduced, newdata = test_set_reduce)

# SVM
predict_SVM <- predict(model_svm, newdata = test_set)
predict_SVM_reduce <- predict(model_svm_Reduced, newdata = test_set_reduce)



# predict_rf <- readRDS("model_rf_predict.rds")
# predict_rf_reduce <- readRDS("model_rf_reduce_predict.rds")
# predict_KNN <- readRDS("predict_KNN.rds")
# predict_KNN_reduce <- readRDS("predict_KNN_reduce.rds")
# predict_NN <- readRDS("predict_nnet.rds")
# predict_NN_reduce <- readRDS("predict_nnet_reduce.rds")

#cm_LR <- confusionMatrix(predict_LR, test_set$default.payment.next.month)
#cm_LR_reduced <-  confusionMatrix(predict_LR_reduce, test_set_reduce$default.payment.next.month)
#cm_rf <- confusionMatrix(predict_rf, test_set$default.payment.next.month)
#cm_rf_reduced <- confusionMatrix(predict_rf_reduce, test_set_reduce$default.payment.next.month)
#cm_KNN <- confusionMatrix(predict_KNN, test_set$default.payment.next.month)
#cm_KNN_reduced <- confusionMatrix(predict_KNN_reduce, test_set_reduce$default.payment.next.month)
#cm_NN <-  confusionMatrix(predict_NN, test_set$default.payment.next.month)
cm_NN_reduced <-  confusionMatrix(predict_NN_reduce, test_set_reduce$default.payment.next.month)
#cm_SVM <-  confusionMatrix(predict_SVM, test_set$default.payment.next.month)
#cm_SVM_reduced <-  confusionMatrix(predict_SVM_reduce, test_set_reduce$default.payment.next.month)

