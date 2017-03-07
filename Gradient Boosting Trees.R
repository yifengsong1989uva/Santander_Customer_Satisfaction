###################################################################################
###Run Gradient Boosting Trees Model on the Santander Customer Satisfaction Dataset
###################################################################################
mydata <- read.csv("train.csv",sep=",")

#check if there are categorical variables
feature_types <- sapply(mydata,class)
which(feature_types=="character" | feature_types=="factor")
table(feature_types)

#Number of positive and negative response values
table(mydata$TARGET)

#check the presence of missing values by each column
missing_matrix <- is.na(mydata)
missing_column <- colSums(missing_matrix)
which(missing_column!=0)

#ID column should be removed
mydata$ID <- NULL



########################################################################################
###Building gradient tree boosting model
########################################################################################
library(xgboost)
library(Matrix)

#split the data into training part and validation part
set.seed(1)
train<- sample(1:nrow(mydata),floor(nrow(mydata)*0.7))
training.part <- mydata[train,]
validation.part <- mydata[-train,]

training.x <- training.part[,1:(ncol(training.part)-1)]
training.y <- training.part[,ncol(training.part)]
validation.x <- validation.part[,1:(ncol(validation.part)-1)]
validation.y <- validation.part[,ncol(validation.part)]

#convert to DMatrix data type that will be used in gradient boosting tree
DMatrix.training<-xgb.DMatrix(data=as.matrix(training.x),
                              label=training.y)
DMatrix.validation<-xgb.DMatrix(data=as.matrix(validation.x),
                                label=validation.y)



#####################################################
###(1) find the best guess of max_depth (check 4 - 8)
#####################################################
#start with the parameters with learning rate 0.01, number of iterations = 4000,
#subsample & colsample_bytree = 0.7
for (i in 3:8) {
  watchlist <- list(val=DMatrix.validation, train=DMatrix.training)
  params1 <- list(booster = "gbtree",
                  eval_metric = "auc",
                  eta = 0.01, 
                  max_depth = i,
                  subsample = 0.7,
                  colsample_bytree = 0.7,
                  objective = "binary:logistic")
  set.seed(12345)
  xgb1 <- xgb.train(params = params1,
                    data = DMatrix.training, 
                    nrounds = 4000,
                    verbose = 1,
                    watchlist = watchlist,
                    print.every.n = 40)
}
#max_depth = 5 is the best



##########################################################
###(2) find the best guess of learning rate (0.001 - 0.05)
##########################################################
#use max_depth = 5
for (i in c(0.001,0.002,0.004,0.005,0.006,0.008,0.01,0.02,0.05)) {
  watchlist <- list(val=DMatrix.validation, train=DMatrix.training)
  params1 <- list(booster = "gbtree",
                  eval_metric = "auc",
                  eta = i, 
                  max_depth = 5,
                  subsample = 0.7,
                  colsample_bytree = 0.7,
                  objective = "binary:logistic")
  set.seed(12345)
  if (i==0.001) {
    xgb1 <- xgb.train(params = params1,
                      data = DMatrix.training, 
                      nrounds = 9000,
                      verbose = 1,
                      watchlist = watchlist,
                      print.every.n = 50)
  }
  else if (i==0.002) {
    xgb1 <- xgb.train(params = params1,
                      data = DMatrix.training, 
                      nrounds = 6000,
                      verbose = 1,
                      watchlist = watchlist,
                      print.every.n = 50)
  }
  else {
    xgb1 <- xgb.train(params = params1,
                      data = DMatrix.training, 
                      nrounds = 3000,
                      verbose = 1,
                      watchlist = watchlist,
                      print.every.n = 40)
  }
}
#eta=0.004 is the best, however, 0.005 and 0.006 are close

#use max_depth = 4
for (i in c(0.001,0.002,0.004,0.005,0.006,0.008,0.01,0.02,0.05)) {
  watchlist <- list(val=DMatrix.validation, train=DMatrix.training)
  params1 <- list(booster = "gbtree",
                  eval_metric = "auc",
                  eta = i, 
                  max_depth = 4,
                  subsample = 0.7,
                  colsample_bytree = 0.7,
                  objective = "binary:logistic")
  set.seed(12345)
  if (i==0.001) {
    xgb1 <- xgb.train(params = params1,
                      data = DMatrix.training, 
                      nrounds = 11000,
                      verbose = 1,
                      watchlist = watchlist,
                      print.every.n = 50)
  }
  else if (i==0.002) {
    xgb1 <- xgb.train(params = params1,
                      data = DMatrix.training, 
                      nrounds = 6000,
                      verbose = 1,
                      watchlist = watchlist,
                      print.every.n = 50)
  }
  else {
    xgb1 <- xgb.train(params = params1,
                      data = DMatrix.training, 
                      nrounds = 2800,
                      verbose = 1,
                      watchlist = watchlist,
                      print.every.n = 40)
  }
}
#use max_depth = 6
for (i in c(0.001,0.002,0.004,0.005,0.006,0.008,0.01,0.02,0.05)) {
  watchlist <- list(val=DMatrix.validation, train=DMatrix.training)
  params1 <- list(booster = "gbtree",
                  eval_metric = "auc",
                  eta = i, 
                  max_depth = 6,
                  subsample = 0.7,
                  colsample_bytree = 0.7,
                  objective = "binary:logistic")
  set.seed(12345)
  if (i==0.001) {
    xgb1 <- xgb.train(params = params1,
                      data = DMatrix.training, 
                      nrounds = 9000,
                      verbose = 1,
                      watchlist = watchlist,
                      print.every.n = 50)
  }
  else if (i==0.002) {
    xgb1 <- xgb.train(params = params1,
                      data = DMatrix.training, 
                      nrounds = 6000,
                      verbose = 1,
                      watchlist = watchlist,
                      print.every.n = 50)
  }
  else {
    xgb1 <- xgb.train(params = params1,
                      data = DMatrix.training, 
                      nrounds = 2600,
                      verbose = 1,
                      watchlist = watchlist,
                      print.every.n = 40)
  }
}
#with different learning rates, max_depth=5 is still slightly better than max_depth=4 or 6



##########################################################
###(3) find the best guess of subsample size (0.4 - 0.9)
##########################################################
for (i in seq(0.4,0.95,by=0.05)) {
  watchlist <- list(val=DMatrix.validation, train=DMatrix.training)
  params1 <- list(booster = "gbtree",
                  eval_metric = "auc",
                  eta = 0.004, 
                  max_depth = 5,
                  subsample = i,
                  colsample_bytree = 0.7,
                  objective = "binary:logistic")
  set.seed(12345)
  xgb1 <- xgb.train(params = params1,
                    data = DMatrix.training, 
                    nrounds = 3000,
                    verbose = 1,
                    watchlist = watchlist,
                    print.every.n = 50)
}
#subsample=0.70 is the best, however, this parameter does not seem to affect the auc much



###############################################################
###(4) find the best guess of colsample_bytree size (0.3 - 0.9)
###############################################################
#try eta = 0.005
for (i in seq(0.25,0.95,by=0.05)) {
  watchlist <- list(val=DMatrix.validation, train=DMatrix.training)
  params1 <- list(booster = "gbtree",
                  eval_metric = "auc",
                  eta = 0.005, 
                  max_depth = 5,
                  subsample = 0.7,
                  colsample_bytree = i,
                  objective = "binary:logistic")
  set.seed(12345)
  xgb1 <- xgb.train(params = params1,
                    data = DMatrix.training, 
                    nrounds = 3000,
                    verbose = 1,
                    watchlist = watchlist,
                    print.every.n = 50)
}
#try eta = 0.004
for (i in seq(0.25,0.95,by=0.05)) {
  watchlist <- list(val=DMatrix.validation, train=DMatrix.training)
  params1 <- list(booster = "gbtree",
                  eval_metric = "auc",
                  eta = 0.004, 
                  max_depth = 5,
                  subsample = 0.7,
                  colsample_bytree = i,
                  objective = "binary:logistic")
  set.seed(12345)
  xgb1 <- xgb.train(params = params1,
                    data = DMatrix.training, 
                    nrounds = 4000,
                    verbose = 1,
                    watchlist = watchlist,
                    print.every.n = 50)
}
#colsample_bytree between 0.35 and 0.45 gives the best auc in the validation set,
#which may not be true

#compare colsample_bytree=0.4 and colsample_bytree=0.7, for eta = 0.004, validation 5 times
#(different training/validation splitting)
for (i in c(1,12,123,1234,12345)) {
  set.seed(i)
  train<- sample(1:nrow(mydata),floor(nrow(mydata)*0.7))
  training.part <- mydata[train,]
  validation.part <- mydata[-train,]
  
  training.x <- training.part[,1:(ncol(training.part)-1)]
  training.y <- training.part[,ncol(training.part)]
  validation.x <- validation.part[,1:(ncol(validation.part)-1)]
  validation.y <- validation.part[,ncol(validation.part)]

  DMatrix.training<-xgb.DMatrix(data=as.matrix(training.x),
                                label=training.y)
  DMatrix.validation<-xgb.DMatrix(data=as.matrix(validation.x),
                                  label=validation.y)
  
  watchlist <- list(val=DMatrix.validation, train=DMatrix.training)
  params1 <- list(booster = "gbtree",
                  eval_metric = "auc",
                  eta = 0.004, 
                  max_depth = 5,
                  subsample = 0.7,
                  colsample_bytree = 0.7,
                  objective = "binary:logistic")
  set.seed(12345)
  xgb1 <- xgb.train(params = params1,
                    data = DMatrix.training, 
                    nrounds = 3600,
                    verbose = 1,
                    watchlist = watchlist,
                    print.every.n = 60)
}
for (i in c(1,12,123,1234,12345)) {
  set.seed(i)
  train<- sample(1:nrow(mydata),floor(nrow(mydata)*0.7))
  training.part <- mydata[train,]
  validation.part <- mydata[-train,]
  
  training.x <- training.part[,1:(ncol(training.part)-1)]
  training.y <- training.part[,ncol(training.part)]
  validation.x <- validation.part[,1:(ncol(validation.part)-1)]
  validation.y <- validation.part[,ncol(validation.part)]

  DMatrix.training<-xgb.DMatrix(data=as.matrix(training.x),
                                label=training.y)
  DMatrix.validation<-xgb.DMatrix(data=as.matrix(validation.x),
                                  label=validation.y)

  watchlist <- list(val=DMatrix.validation, train=DMatrix.training)
  params1 <- list(booster = "gbtree",
                  eval_metric = "auc",
                  eta = 0.004, 
                  max_depth = 5,
                  subsample = 0.7,
                  colsample_bytree = 0.4,
                  objective = "binary:logistic")
  set.seed(12345)
  xgb1 <- xgb.train(params = params1,
                    data = DMatrix.training, 
                    nrounds = 3600,
                    verbose = 1,
                    watchlist = watchlist,
                    print.every.n = 60)
}
#on average, colsample_bytree=0.4 is slightly worser than colsample_bytree=0.7



###############################################################
###(5) find the best guess of number of trees
###############################################################
#use colsample_bytree=0.7 and colsample_bytree=0.7, eta = 0.004, max_depth=5 to
#find the best guess of number of trees, validation 10 times
#(different training/validation splitting)
for (i in c(1,12,123,1234,12345,123456,1234567,12345678,123456789,1234567890)) {
  set.seed(i)
  train<- sample(1:nrow(mydata),floor(nrow(mydata)*0.7))
  training.part <- mydata[train,]
  validation.part <- mydata[-train,]
  
  training.x <- training.part[,1:(ncol(training.part)-1)]
  training.y <- training.part[,ncol(training.part)]
  validation.x <- validation.part[,1:(ncol(validation.part)-1)]
  validation.y <- validation.part[,ncol(validation.part)]
  
  DMatrix.training<-xgb.DMatrix(data=as.matrix(training.x),
                                label=training.y)
  DMatrix.validation<-xgb.DMatrix(data=as.matrix(validation.x),
                                  label=validation.y)
  
  watchlist <- list(val=DMatrix.validation, train=DMatrix.training)
  params1 <- list(booster = "gbtree",
                  eval_metric = "auc",
                  eta = 0.004, 
                  max_depth = 5,
                  subsample = 0.7,
                  colsample_bytree = 0.7,
                  objective = "binary:logistic")
  set.seed(12345)
  xgb1 <- xgb.train(params = params1,
                    data = DMatrix.training, 
                    nrounds = 3200,
                    verbose = 1,
                    watchlist = watchlist,
                    print.every.n = 40)
}
#Based on the results from the 10 validations,
#it was determined that number of trees=1900 gives the optimal prediction



##############################################################################################
###(6) ###build the model with the selected training set and evaluate on the selected test set
##############################################################################################
#read in the train data
train_data <- read.csv("TrainData.csv",sep=",")
x <- train_data[,3:(ncol(train_data)-1)]
y <- train_data[,ncol(train_data)]
#convert to DMatrix data type that will be used in gradient boosting tree
DMatrix.train_data <- xgb.DMatrix(data=as.matrix(x),label=y)

#read in the test data
mytest <- read.csv("TestData.csv",sep=",")
test.x <- as.matrix(mytest[,3:(ncol(mytest)-1)])
test.y <- mytest$TARGET
DMatrix.test_data <- xgb.DMatrix(data=test.x,label=test.y)

#set up the parameters with eta 0.004, number of iterations = 1900, subsample & colsample_bytree = 0.7, max_depth=5 
watchlist <- list(train=DMatrix.train_data, val=DMatrix.test_data)
params1 <- list(booster = "gbtree",
                eval_metric = "auc",
                eta = 0.004, 
                max_depth = 5,
                subsample = 0.7,
                colsample_bytree = 0.7,
                objective = "binary:logistic")
set.seed(12345)
xgb1 <- xgb.train(params = params1,
                  data = DMatrix.train_data, 
                  nrounds = 1900,
                  verbose = 1,
                  watchlist = watchlist,
                  print.every.n = 20)
pred_target <- predict(xgb1,test.x)

library(ROCR)
prob.pred <- prediction(pred_target, mytest$TARGET)
auc <- performance(prob.pred,"auc")
auc <- unlist(slot(auc, "y.values"))
auc
#the auc on the selected test set is 0.8377

#save predicted probabilities into .csv file
write.csv(pred_target,file="GBT_pred.csv",row.names=F)

#save true TARGET values of the test set into .csv file
write.csv(mytest$TARGET,file="true_values.csv",row.names=F)

#plot the feature importances resulting from the GBT Model
train.x.sparse <- sparse.model.matrix(~.,data=train_data[,-c(1:2,372)])
train.x.sparse <- train.x.sparse[,-1] #the first column "(Intercept)" should be dropped
test.x.sparse <- sparse.model.matrix(~.,data=mytest[,-c(1:2,372)])
test.x.sparse <- test.x.sparse[,-1]
DMatrix.training<-xgb.DMatrix(data=train.x.sparse,
                              label=y)
DMatrix.test<-xgb.DMatrix(data=test.x.sparse,
                                label=test.y)
watchlist <- list(train=DMatrix.training, val=DMatrix.test)
params2 <- list(booster = "gbtree",
                eval_metric = "auc",
                eta = 0.004, 
                max_depth = 5,
                subsample = 0.7,
                colsample_bytree = 0.7,
                objective = "binary:logistic")
set.seed(12345)
xgb2 <- xgb.train(params = params2,
                  data = DMatrix.training, 
                  nrounds = 1900,
                  verbose = 1,
                  watchlist = watchlist,
                  print.every.n = 20)

importance_matrix <- xgb.importance(train.x.sparse@Dimnames[[2]], model = xgb2)
#top 10 features:
top_10 <- cbind(importance_matrix$Feature[1:10],importance_matrix$Gain[1:10])
top_10 <- as.data.frame(top_10)
#save to csv
write.csv(top_10,"GBT_feature_importances.csv",row.names=F)