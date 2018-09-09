setwd("C:/Users/Alaric/Desktop/WINTER 2018/STAT 441/Project Finale")
data <- read.csv('default of credit card clients.csv')
factor_vars <- c('SEX','EDUCATION','MARRIAGE','default.payment.next.month')
data[factor_vars] <- lapply(data[factor_vars], function(x) as.factor(x))

data <- transform(data, MARRIAGE_BIN = ifelse(MARRIAGE==0, 3, MARRIAGE))  # marriage = 0 becomes 3
data <- transform(data, EDUCATION_BIN = ifelse(EDUCATION==5 | EDUCATION==6, 4, EDUCATION))  # education = 5 or 6 becomes 4
age_int <- seq(from=20, to=80, by=5)
#data$AGE_INT <- cut(data$AGE,age_int) # for reference, if you wanted to see which age interval AGE_BIN fell into you can uncomment this
data$AGE_BIN <- cut(data$AGE,age_int,labels=FALSE) # age banded into intervals of 5 years
#data_mod <- subset(data, select=-c(AGE_INT,ID)) #can use this line to drop certain columns in select=-c(.) if the dataset is too large 

#Creating Net Amts 
for (i in 1:5)
{ 
  name_net <- paste("NET_AMT",i,sep="")
  name_net <- paste("NET_AMT_INT",i,sep="")
  name_net_bin <- paste("NET_AMT_BIN",i,sep="")
  name_bill <- paste("BILL_AMT",i+1,sep="")
  name_pay <- paste("PAY_AMT",i,sep="")
  data[name_net] = data[name_bill] - data[name_pay]
  
  net_int <- seq(from=min(data[name_net])-1, to=max(data[name_net])+1, by=25000)  # binning net amounts by intervals of $25000
  data[name_net_bin] <- cut(data[[name_net]],net_int,labels=FALSE) #comment out if you don't want to bin the net_amounts
  #data[name_net_bin] <- cut(data[[name_net]],net_int) # for reference, if you wanted to see the label of which interval the net_amts fell into
}


library(ggplot2) # Data visualization
library(readr) # CSV file I/O, e.g. the read_csv function
library(caTools)
library(class)
library(caret)
library(ROCR)
library(pROC)
library(factoextra)

# KNN
set.seed(12345)
split = sample.split(data$default.payment.next.month, SplitRatio = 0.7)
training_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

#Fitting the K Nearest Neighbour algorithm to classify the output as desired. 

KNN_pred = knn(train = training_set[, -24],
               test = test_set[, -24],
               cl = training_set[, 24],
               k = 95,
               prob = TRUE)

# Making the Confusion Matrix
Knn_cm = table(test_set[, 24], KNN_pred)
Knn_cm
confusionMatrix(Knn_cm)
#As we can see from the output, our Accuracy ( Sum of True positive and True Negative / Number of Test Observations) is 80.8%
#Also, our sensitivity is 82.99%. 

#Lets try and fit a Logistic Regression to the same data. 

#################################
#########Logistic Regression#####
#################################
mylogit<-glm(default.payment.next.month~., data = training_set,family = "binomial")

prob_pred<-predict(mylogit,type='response',newdata = test_set[-24])
prob_pred
#As we know, we get probabilities out of the logistic regression model. So in this case, we are keeping of threshold of 50%. Anything below 50% are taken as zero and above it is 1. 

y_pred=ifelse(prob_pred>0.5,1,0)
cm1=table(test_set[,24],y_pred)
confusionMatrix(cm1)
