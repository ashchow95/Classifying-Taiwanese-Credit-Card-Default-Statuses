setwd("C:/Users/Alaric/Desktop/WINTER 2018/STAT 441/Project Finale")
data <- read.csv('default of credit card clients.csv')
factor_vars <- c('SEX','EDUCATION','MARRIAGE','default.payment.next.month')


data$AGE<-cut(data$AGE, breaks = c( 10, 30,50,100), labels = c("10-30", "30-50","50-100"))
data$SEX<-cut(data$SEX, 2,labels = c("Female","Male"))
data$MARRIAGE<-cut(data$MARRIAGE, 4,labels = c("Married","Single","Divorced","Other"))
data[factor_vars] <- lapply(data[factor_vars], function(x) as.factor(x))


#install.packages('corrplot')
nums <- sapply(data, is.numeric)
library(corrplot)
corrplot.mixed(cor(data[,nums]), lower="circle", upper="color", 
               tl.pos="lt", diag="n", order="hclust", hclust.method="complete")
## No particular severe correlations

install.packages('ggplot2')
library(ggplot2)   # For plotting
# Look at correlation between defaulting / limit balance / age / sex / marriage / education
ggplot(data=data,mapping = aes(x=AGE,y=data$LIMIT_BAL,fill=default.payment.next.month)) + geom_boxplot() 
## Seems to be no relationship between age and default payment next month. It appears that 30-50 range has a higher limit balance than 50-100 and 10-30
ggplot(data=data,mapping = aes(x=EDUCATION,y=data$LIMIT_BAL,fill=default.payment.next.month)) + geom_boxplot() 
## Education with defualt, no relationship as well.
ggplot(data=data,mapping = aes(x=SEX,y=data$LIMIT_BAL,fill=default.payment.next.month)) + geom_boxplot() 
## No direct relationship either
ggplot(data=data,mapping = aes(x=MARRIAGE,y=data$LIMIT_BAL,fill=default.payment.next.month)) + geom_boxplot() 
## No direct relationship either



