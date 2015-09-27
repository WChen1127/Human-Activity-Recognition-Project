#### Setup
training = read.csv("~/Desktop/weisdata-home/machinelearning/pml-training.csv")
testing = read.csv("~/Desktop/weisdata-home/machinelearning/pml-testing.csv")

#### Preprocess
head(training)
summary(training)
# NA's (19216): column index: 18:19, 21:22,24:25,27:36,50:59,75:83,93:94:,96:97,99:100,103:112,131:132,
# 134:135, 137:138,141:150,
# blank (19216): column index: 12:17,20,23,26,69:74,87:92,95,98,101,125:130,133,136,139,
# 19216/19622=98% missing values, delete columns
# total: 12:36,50:59,69:83,87:101,103:112, 125:139,141:150
training <- training[,-c(12:36,50:59,69:83,87:101,103:112,125:139,141:150)]
summary(training)
testing <- testing[,-c(12:36,50:59,69:83,87:101,103:112,125:139,141:150)]
# outliers:
names(training)
par(mfrow=c(1,1))    
lapply(training[,10:18],function(x) ifelse(class(x)=="factor", plot(x), boxplot(x)))
# near zero variable
nsv <- nearZeroVar(training,saveMetrics=TRUE)
nsv
# detect: new_window
training <- training[,-6]
testing <- testing[,-6]
# check testing str
str(training)
str(testing)
lapply(names(training[,-59]), function(x) class(training[,x])==class(testing[,x])) 
# index: 45, 57:58 differ types
class(training[,45])
testing[,45] <- as.numeric(testing[,45])
testing[,57] <- as.numeric(testing[,57])
testing[,58] <- as.numeric(testing[,58])
# factor variables must have the same level in both training and testing
#index: 2,5
names(training)
levels(training[,2]); levels(testing[,2])
levels(training[,5]); levels(testing[,5])

training$cvtd_timestamp_date <- vector("integer", length(training$cvtd_timestamp))
training$cvtd_timestamp_time <- vector("integer", length(training$cvtd_timestamp))
for(i in 1:length(training$cvtd_timestamp)){
  date_time <- strsplit(as.character(training$cvtd_timestamp[i]), ' ')
  training$cvtd_timestamp_date[i] <- date_time[[1]][1]
  training$cvtd_timestamp_time[i] <- strsplit(date_time[[1]][2],":")[[1]][1]
}
training$cvtd_timestamp_date <- sapply(training$cvtd_timestamp_date, function(x) as.factor(x))
training$cvtd_timestamp_time <- sapply(training$cvtd_timestamp_time, function(x) as.factor(x))
training <- training[,-5]
# apply to testing set
testing$cvtd_timestamp_date <- vector("integer", length(testing$cvtd_timestamp))
testing$cvtd_timestamp_time <- vector("integer", length(testing$cvtd_timestamp))

for(i in 1:length(testing$cvtd_timestamp)){
  date_time <- strsplit(as.character(testing$cvtd_timestamp[i]), ' ')
  testing$cvtd_timestamp_date[i] <- date_time[[1]][1]
  testing$cvtd_timestamp_time[i] <- strsplit(date_time[[1]][2],":")[[1]][1]
}
testing$cvtd_timestamp_date <- sapply(testing$cvtd_timestamp_date, function(x) as.factor(x))
testing$cvtd_timestamp_time <- sapply(testing$cvtd_timestamp_time, function(x) as.factor(x))
testing <- testing[,-5]
# delete the first column, the serial number, meaningless
training <- training[,-1]
testing <- testing[,-1]
####Slice training
set.seed(1010)
inTrain <- createDataPartition(y=training$classe, p=0.60, list=FALSE)
trainS <- training[inTrain,]
testS <- training[-inTrain,]

####Models
## random forests
library(randomForest)
set.seed(101010)
modrf <- randomForest(classe ~. , data=trainS, importance=T)
predrf <- predict(modrf, testS[,-57], type = "class")
confusionMatrix(predrf, testS[,57])
head(predrf)

# predict in the testing set
predrf_test1 <- predict(modrf, testing[,-57], type = "class")
predrf_test1
# prediction: B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
# seems exactly the same to the results


varImp <- importance(modrf)
varImp[1:10,]
varImpPlot(modrf, type=1)
names(sort(varImp[,6], decreasing=T))

# the top most important features to predict
varImp30 <- names(sort(varImp[,6], decreasing=T))[1:30]
varImp30
set.seed(10101010)
modrf_varImp30 <- randomForest(classe ~. , data=trainS[,c(varImp30, "classe")], importance=T)
predrf_varImp30 <- predict(modrf, testS[,varImp30], type = "class")
confusionMatrix(predrf_varImp30, testS[,"classe"])
predrf_test_varImp30 <- predict(modrf, testing[,varImp30], type = "class")
predrf_test_varImp30

####submit answer
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
answers = rep("", 20)
for(i in 1:20){
  answers[i] <- as.character(predrf_test1[[i]])
}
setwd("Desktop/weisdata-home/machinelearning/write_files/")
pml_write_files(answers)
