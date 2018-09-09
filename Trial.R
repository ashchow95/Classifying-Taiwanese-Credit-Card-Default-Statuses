#### Load Packages & Data ####
#install.packages("klaR")
library(klaR)
library(MASS)
#install.packages('modelr')
library(modelr)
library(caret)
library(rpart)
library(randomForest)
library(doParallel)
library(pROC)
library(kernlab)
#install.packages('tidyverse')
library(tidyverse)
library(caTools) 

library(caTools)

setwd("C:/Users/Alaric/Desktop/WINTER 2018/STAT 441/Project Finale")
data <- read.csv('default.payment.next.month of credit card clients.csv')
factor_vars <- c('SEX','EDUCATION','MARRIAGE','default.payment.next.month.payment.next.month','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6')
data[factor_vars] <- lapply(data[factor_vars], function(x) as.factor(x))
levels(data$default.payment.next.month.payment.next.month) <- list(no="0", yes="1") 

data <- transform(data, MARRIAGE_BIN = ifelse(MARRIAGE==0, 3, MARRIAGE))  # marriage = 0 becomes 3
data <- transform(data, EDUCATION_BIN = ifelse(EDUCATION==5 | EDUCATION==6, 4, EDUCATION))  # education = 5 or 6 becomes 4

age_int <- seq(from=20, to=80, by=5)
#data$AGE_INT <- cut(data$AGE,age_int) # for reference, if you wanted to see which age interval AGE_BIN fell into you can uncomment this
data$AGE_BIN <- cut(data$AGE,age_int,labels=FALSE) # age banded into intervals of 5 years
#data_mod <- subset(data, select=-c(AGE_INT,ID)) #can use this line to drop certain columns in select=-c(.) if the dataset is too large 

# divide into train and test set
set.seed(1337)
split = sample.split(data$default.payment.next.month.payment.next.month, SplitRatio = 0.8)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

# evaluation control:
eva_ctrl <-  trainControl(method = 'repeatedcv',number = 10,repeats = 5,
                          savePredictions = T, summaryFunction = twoClassSummary, classProbs = T)

## Utilize Caret re-sampling methods
# repeatedcv -- is probably repeated random sub-sampling validation, i.e division to train and test data is done in random way.
# https://en.wikipedia.org/wiki/Cross-validation_(statistics)#Repeated_random_sub-sampling_validation

# http://www.edii.uclm.es/~useR-2013/Tutorials/kuhn/user_caret_2up.pdf

# *Full logistic regression: -----
set.seed(7777)
glm.full <- train(default.payment.next.month ~ .,
                  data = mydata,
                  method = 'glm',
                  family = binomial,
                  metric = 'ROC',
                  trControl = eva_ctrl
)

# use the default bootstrapping technique and set my tuning controls to turn on the options ???twoclassSummary??? and ???classProbs??? to choose the optimal parameter on metric AUC (or ???ROC???). Then I tuned on KNN, linear SVM, and RF using 5 values to tune for each parameter (I will extend the range if the optimal parameter is at the endpoints). I also scale and center the data when tune and train the KNN and SVM models since the algorithms are distance-based. The optimal parameters I found for the full models are: k = 13 (KNN),
# mtry = 2 (RF) and cost = 0.5 (linear SVM).

# After that I train my 5 full models with 5 10-fold cross-validation technique. I also turned on ???twoclassSummary??? and ???classProbs??? to use ???ROC??? metric and ???savePredictions??? to save predictions. I used the optimal parameters found in tuning process and just as above, I scale and center data for KNN and linear SVM.


# tuning control:
tune_ctrl <- trainControl(summaryFunction = twoClassSummary, classProbs = TRUE)
## Generate paramters that further control how models are created
(two class summary)

#KNN
train(default.payment.next.month.payment.next.month ~ .,
      data = train_set,
      method = "knn",
      metric = "ROC",
      preProc = c("center","scale"),
      tuneLength = 5,
      trControl = tune_ctrl) # Optimal k = 13


#Tune for SVM Linear
set.seed(1234)
train(default.payment.next.month ~ .,
      data = mydata[sample(nrow(mydata),5000),],
      method = "svmLinear2",
      preProc = c('center','scale'),
      metric = "ROC",
      tuneLength = 5,
      trControl = tune_ctrl)    # Optimal cost = 0.5 


#### Full model building -------

# *Full Random Forest: ------
set.seed(7777)
rf.full <- train(default.payment.next.month ~ ., 
                 data = mydata, 
                 method = "rf",
                 metric = "ROC",
                 tuneGrid = data.frame(mtry=2),
                 trControl = eva_ctrl)

# * Full Naive Bayes: -----
set.seed(7777)
nb.full <- train(default.payment.next.month ~ ., 
                 data = mydata, 
                 method = "nb",
                 metric = "ROC",
                 tuneGrid = data.frame(fL=0,usekernel=TRUE, adjust=1),
                 trControl = eva_ctrl)

# * Full SVM Linear: ------
set.seed(7777)
svmLinear.full <- train(default.payment.next.month ~ .,
                        data = mydata,
                        method = "svmLinear",
                        preProc = c('center','scale'),
                        metric = "ROC",
                        tuneGrid = data.frame(C = 0.5),
                        trControl = eva_ctrl) 



#### Feature Reduction #####
reduced <- step(glm(default.payment.next.month ~ ., data = mydata, family = binomial),
                direction = 'both')


#### EVALUATE #######
summary(resamples(list(`Full KNN`=knn.full, 
                       `Full Logistic Regression`=glm.full,
                       `Full Random Forest` = rf.full,
                       `Full Naive Bayes`= nb.full,
                       `Full SVM Linear` = svmLinear.full,
                       )))

# Values are AUC
# https://stats.stackexchange.com/questions/132777/what-does-auc-stand-for-and-what-is-it

## PREDICT ON TRAINING SET
# prob.nb<- predict(nb.reduced,train_set,type = 'prob')
# roc(train_set$default,prob.nb$yes)     
## PREDICT ON TESTING SET
# prob.nb<- predict(nb.reduced,test_set,type = 'prob')
# roc(test_set$default,prob.nb$yes) 

#24000 samples
#27 predictor
#2 classes: 'no', 'yes' 

#Pre-processing: centered (86), scaled (86) 
#Resampling: Bootstrapped (25 reps) 
#Summary of sample sizes: 24000, 24000, 24000, 24000, 24000, 24000, ... 
#Resampling results across tuning parameters:
#  
#  k   ROC        Sens       Spec     
#5  0.6793716  0.8767672  0.3747946
#7  0.6933661  0.8991700  0.3617133
#9  0.7027048  0.9136846  0.3544072
#11  0.7100527  0.9237415  0.3507521
#13  0.7153684  0.9304767  0.3470546##

#ROC was used to select the optimal model using the largest value.
#The final value used for the model was k = 13. KNN