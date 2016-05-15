#load library
library(caret)
#Download data
#trainingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
#testingUrl <-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
path <- "F://Data Science//ื๗าต//Practical Machine Learning"
#download.file(url = trainingUrl, destfile = paste(path, "//training.csv", sep = ""))
#download.file(url = testingUrl, destfile = paste(path, "//testing.csv", sep = ""))

#read data
set.seed(123456)
training <- read.csv(paste(path, "//training.csv", sep = ""))
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
training <- training[inTrain, ]
validation <- training[-inTrain, ]
testing <- read.csv(paste(path, "//testing.csv", sep = ""))

#Preprocess the data

rm_train <- grepl("^X|timestamp|window",names(training))
rm_valid <- grepl("^X|timestamp|window",names(validation))
rm_test <- grepl("^X|timestamp|window",names(testing))
rm <- rm_train & rm_test & rm_valid 
training <- training[,!rm]
testing <- testing[,!rm]
validation <- validation[,!rm]

remove_div <- function(dataset, outcome){
    classes <- sapply(dataset, class)
    #var_testing <- sapply(dataset[,which(outcomes != "factor")], sd)
    DIVs <- sapply(dataset[,which(classes == "factor")], function(dat) sapply(dat, function(x)
        grepl("#DIV/0!",x)))
    DIV_RM <- which(rowSums(DIVs) > 0)
    dataset <- dataset[-DIV_RM,]
    for(i in 1: (ncol(dataset)-1)){
        dataset[,i] <- as.numeric(dataset[,i])
    }
    dataset
}

training <- remove_div(training, classe)
validation <- remove_div(validation, classe)

na_testing_train <- sapply(training,function(dat) mean(is.na(dat)))
na_testing_validation <- sapply(validation,function(dat) mean(is.na(dat)))
na_testing_test <- sapply(testing,function(dat) mean(is.na(dat)))
na_rm <- which(na_testing_train > 0.95 | na_testing_test > 0.95 | na_testing_validation > 0.95)
training <- training[,-na_rm]
testing <- testing[,-na_rm]
validation <- validation[,-na_rm]

#Cleaning the memory
rm(path);rm(rm);rm(rm_test);rm(rm_train);rm(rm_valid);
rm(na_rm);rm(na_testing_test);rm(na_testing_train);rm(na_testing_validation);

#Train Control using 10 fold Cross Validation
ctrol <- trainControl(method = "cv", number = 10)

#Trees
mod1 <- train(classe ~ . , data = training, method = "rpart",trControl = ctrol)

#boosting
mod2 <- train(classe ~ . , data = training, method = "gbm",trControl = ctrol,verbose = FALSE)

#random forest
mod3 <- train(classe ~ ., data = training, method = "rf",trControl = ctrol)

#Bagging
mod4 <- train(classe ~ ., data = training,method = "treebag",trControl = ctrol)

#Linear discriminant analysis and Naive Bayes
mod5 <- train(classe ~ ., data = training, method = "lda",trControl = ctrol)

mod <- list(mod1, mod2, mod3, mod4, mod5)

Accuracy <- numeric()
for(i in 1:5){
    Accuracy[i] <- confusionMatrix(predict(mod[[i]], validation), validation$classe)$overall[1]
}


#Ensemble Learning
pred_train <- lapply(mod, function(model) predict(model, training))
names(pred_train) <- c("model1","model2","model3","model4","model5") 
pred_train$classe <- training$classe
Stack_data <- as.data.frame(pred_train)
comod <- train(classe ~ ., data = Stack_data, method = "rf",trControl = ctrol)

#Prediction on validation dataset
pred_validation <- lapply(mod, function(model) predict(model, validation))
names(pred_validation) <- c("model1","model2","model3","model4","model5") 
pred_validation$classe <- validation$classe
Stack_data_validation <- as.data.frame(pred_validation)
Accuracy[6] <- confusionMatrix(predict(comod, Stack_data_validation), validation$classe)$overall[1]

#Prediction on testing dataset
pred_testing <- lapply(mod, function(model) predict(model, testing[, -ncol(testing)]))
names(pred_testing) <- c("model1","model2","model3","model4","model5") 
pred_testing$classe <- testing$problem_id
Stack_data_testing <- as.data.frame(pred_testing)
predict(comod, Stack_data_testing)
