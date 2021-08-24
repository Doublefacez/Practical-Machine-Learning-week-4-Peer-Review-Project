# Week 4 Peer Review  project

library(dplyr)
library(ggplot2)
library(caret)
library(AppliedPredictiveModeling)
library(pgmm)
library(rpart)
library(lubridate)
library(forecast)
library(e1071)
library(ElemStatLearn)
library(gbm)
library(elasticnet)
library(rpart.plot)

getwd()
setwd("C:/Users/lamti/Desktop/datasciencecoursera/Course 8 Practical machine learning/Week 4/Peer assessment")


data_test <- read.csv("pml-testing.csv", header = TRUE, na.strings = c("NA", "#DIV/0!", ""))
data_train <- read.csv("pml-training.csv", header = TRUE, na.strings = c("NA", "#DIV/0!", ""))

colnames(data_train)
colnames(data_test)



## remove columns that have NA as 95% of the observations for both data set
col_na_filter <- colSums(is.na(data_train))/nrow(data_train) < 0.95
training_data <- data_train[, col_na_filter == TRUE]
testing_data <- data_test[, col_na_filter == TRUE]

### Checking the two data sets have been successfully filtered
dim(training_data)
dim(testing_data)

## Removing the first 7 variables which do not related to the training model.
training_data <- training_data[,-c(1:7)]
testing_data <- testing_data[,-c(1:7)]

### Double check the we dont filtered out classe and problem_id
colnames(training_data)
colnames(testing_data)

## Creating data partition
set.seed(1234)
inTrain = createDataPartition(y =training_data$classe, p = 3/4, list = FALSE)
training = training_data[ inTrain,]
testing = training_data[-inTrain,]
dim(training)

#Cross validation
control <- trainControl(method = "cv", number = 3)

### Predict with decision tree and confusion matrix

model_rpart <- train(classe ~ ., data = training, method = "rpart", trControl = control)
predict_rpart <- predict(model_rpart, testing)
rpart_cm <- confusionMatrix(predict_rpart, testing$classe)
rpart_cm
rpart.plot(model_rpart$finalModel)

### They showed that this model only has ~50% predication accuracy on the training data. 
### Which is not consider as a good model. 

## Random Forest model
model_rf <- train(classe ~ ., data = training, method = "rf", ntree = 100, trControl = control)
predict_rf <- predict(model_rf, testing)
rf_cm <- confusionMatrix(predict_rf, testing$classe)
rf_cm

### plotting RF accuracy

plot(rf_cm$table, main = "Random Forest Prediction Accuracy")


#### It shows that the Random Forest Predication model has an accuracy of ~99%, 
#### Which is a very high 

## Gradient Boosting Model

model_gbm <- train(classe ~ ., data = training, method = "gbm", verbose = FALSE, trControl = control)
predict_gbm <- predict(model_gbm, testing)
gbm_cm <- confusionMatrix(predict_gbm, testing$classe)
gbm_cm


### Plotting GBM accuracy
plot(gbm_cm$table, main = "Gradient Boosting Model Accuracy level")


# Model selection
### Comparing the appropriate model which has the highest predication 
### accuracy for the dataset.

compare <- data.frame(Model = c("Decision Trees (CART)", "Random Forest", "Gradient Boosting"),
                      Accuracy = rbind(rpart_cm$overall[1], rf_cm$overall[1], 
                                       gbm_cm$overall[1]))

compare
 

## Conclusion

### Random forest model has a higher prediction accuracy (~99%) followed by
### Gradient Boosting Model (~97%) then Decision Trees model (~50%). As a result, we will use the RF model to predict the test data.


# Predicting test-data
pre_rf_test <- predict(model_rf, testing_data)
pred_result <- data.frame(Problem_id = testing_data$problem_id,
                          Prediction_outcome = pre_rf_test)
pred_result





