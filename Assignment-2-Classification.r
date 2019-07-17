library(dplyr)
library(ggplot2)
library(caret)
library(glmnet)
library(corrplot)
library(gridExtra)


covtype =read.csv(file="D:/MSDS-SPU/DS630-Winter18-Wed6pm-Machine-Learning/DS630-Assignment-2-Classification/covtype.data", header=TRUE, sep=",", stringsAsFactors=FALSE)



# "Fixed" attributes' names
names <- c("Elevation","Aspect","Slope","HorDistToHydro","VertDistToHydro",
           "HorDistRoad","Hillshade09","Hillshade12","Hillshade15",
           "HorDistFire")
# Four binary attributes for wilderness areas:
names <- c(names,"WA_RWA","WA_NWA","WA_CPWA","WA_CLPWA")
# 40 (!) binary attributes for soil types:
names <- c(names,sprintf("ST%02d",1:40))
# The cover type
names <- c(names,"Class")

# Assign these names to the attributes
names(covtype) <- names
# Let's also assign labels to the coverage types.
covtype$Class <- as.factor(covtype$Class)
levels(covtype$Class) <- c("Spruce/Fir", "Lodgepole Pine",
                           "Ponderosa Pine","Cottonwood/Willow","Aspen",
                           "Douglas-fir","Krummholz")
# How does it looks like?
str(covtype)


# Which are the columns we want to consider?
searchOn <- c("WA_RWA","WA_NWA","WA_CPWA","WA_CLPWA")
# Get the index of each WA_
indexOfWA <- apply(covtype[,searchOn], 1, function(x) which(x == 1))
# Convert it to a factor with the column names we used.
factorOfWA <- factor(indexOfWA,labels = searchOn)
# Add it to the data frame
covtype$WildArea <- factorOfWA
# Drop the binary variables we don't need anymore
covtype[searchOn] <- list(NULL)

# Which are the columns we want to consider?
searchOn <- sprintf("ST%02d",1:40)
searchOn


# Get the index of 1 in ST01..ST40
indexOfST <- apply(covtype[,searchOn], 1, function(x) which(x == 1))
# Convert it to a factor with the column names we used.
factorOfST <- factor(indexOfST,labels = searchOn)
# Add it to the data frame
covtype$SoilType <- factorOfST
# Drop the binary variables we don't need anymore
covtype[searchOn] <- list(NULL)
str(covtype)


write.csv(covtype,"D:/MSDS-SPU/DS630-Winter18-Wed6pm-Machine-Learning/DS630-Assignment-2-Classification/covtype.csv",row.names = F)

index <- createDataPartition(covtype$Class, p = .10, list = FALSE)

covtype_sample <- covtype[index,]

write.csv(covtype,"D:/MSDS-SPU/DS630-Winter18-Wed6pm-Machine-Learning/DS630-Assignment-2-Classification/covtype_sample.csv",row.names = F)

## for faster speed convert soil type into numeric
covtype$SoilType <- as.numeric(covtype$SoilType)

covtype =read.csv(file="D:/MSDS-SPU/DS630-Winter18-Wed6pm-Machine-Learning/DS630-Assignment-2-Classification/covtype_sample.csv")

str(covtype)

## Train & Test data
set.seed(123)

# Take 75% Train  & 20% test'
index <- createDataPartition(covtype$Class, p = .75, list = FALSE)
train <- covtype[index,]
test <- covtype[-index,]



## Cross V parameter setting 5 fold
train_control<- trainControl(method="cv", number=5)

### Logistic regression
mlm_model <- train(as.factor(Class)~., data=train, trControl=train_control, method="multinom",maxit=20)

## Print the results
mlm_model$results

## Plot

## coef plot
options(repr.plot.width = 6, repr.plot.height = 3)
barplot(coef(mlm_model$finalModel), main="Coefficients of Multinomial",
        beside=TRUE,names.arg =rownames(mlm_model$coefnames),las=2,cex.names = 0.65,cex=0.65,col = c('red','blue','green','brown','grey','yellow'))

## lets test
mlm_predict = predict(mlm_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,mlm_predict))

### SVM
svm_model <- train(Class~., data=train, trControl=train_control, method="svmLinear",tuneGrid = expand.grid(C = c(1)))

## Print the results
svm_model$results

## lets test
svm_predict = predict(svm_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,svm_predict))

knn_model <- train(Class~., data=train, trControl=train_control, method="knn",tuneGrid = expand.grid(k = 5))

## Print the results
knn_model$results

## lets test
knn_predict = predict(knn_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,knn_predict)) 

### Decision tree
tree_model <- train(Class~., data=train, trControl=train_control, method="rpart",tuneGrid = expand.grid(cp = c(0.01)))

## Print the results
tree_model$results

## lets test
tree_predict = predict(tree_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,tree_predict))

### Random Forest
tunegrid <- expand.grid(.mtry=c(2))
rf_model <- train(Class~., data=train, trControl=train_control, method="rf",tuneGrid = tunegrid,ntree=100)

## Print the results
rf_model$results

## Variable Importance
options(repr.plot.width = 7, repr.plot.height = 5)
plot(varImp(rf_model))

## lets test
rf_predict = predict(rf_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,rf_predict))

### GBM
gbmGrid <-  expand.grid(interaction.depth = c(2),shrinkage = 0.1,n.minobsinnode = 10,n.trees=100)
gbm_model <- train(Class~., data=train, trControl=train_control, tuneGrid = gbmGrid,method="gbm")
1
## Print the results
gbm_model$results



## lets test
gbm_predict = predict(gbm_model,test,n.trees = 100)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,gbm_predict))

## Scale the values

normalize <- function(x)
{
  return((x- min(x)) /(max(x)-min(x)))
}



# To get a vector, use apply instead of lapply
covtype[,1:10] <- sapply(covtype[,1:10], function(x) ((x- min(x)) /(max(x)-min(x))) )

train <- covtype[index,]
test <- covtype[-index,]


## Cross V parameter setting 5 fold
train_control<- trainControl(method="cv", number=5)

### Logistic regression
mlm_model <- train(as.factor(Class)~., data=train, trControl=train_control, method="multinom",maxit=20)

## Print the results
mlm_model$results

## Plot

## coef plot
options(repr.plot.width = 6, repr.plot.height = 3)
barplot(coef(mlm_model$finalModel), main="Coefficients of Multinomial",
        beside=TRUE,names.arg =rownames(mlm_model$coefnames),las=2,cex.names = 0.65,cex=0.65,col = c('red','blue','green','brown','grey','yellow'))

## lets test
mlm_predict = predict(mlm_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,mlm_predict))

### SVM
svm_model <- train(Class~., data=train, trControl=train_control, method="svmLinear",tuneGrid = expand.grid(C = c(1)))

## Print the results
svm_model$results

## lets test
svm_predict = predict(svm_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,svm_predict))

knn_model <- train(Class~., data=train, trControl=train_control, method="knn",tuneGrid = expand.grid(k = 5))

## Print the results
knn_model$results

## lets test
knn_predict = predict(knn_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,knn_predict)) 

### Decision tree
tree_model <- train(Class~., data=train, trControl=train_control, method="rpart",tuneGrid = expand.grid(cp = c(0.01)))

## Print the results
tree_model$results

## lets test
tree_predict = predict(tree_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,tree_predict))

### Random Forest
tunegrid <- expand.grid(.mtry=c(2))
rf_model <- train(Class~., data=train, trControl=train_control, method="rf",tuneGrid = tunegrid,ntree=100)

## Print the results
rf_model$results

## Variable Importance
options(repr.plot.width = 7, repr.plot.height = 5)
plot(varImp(rf_model))

## lets test
rf_predict = predict(rf_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,rf_predict))

### GBM
gbmGrid <-  expand.grid(interaction.depth = c(2),shrinkage = 0.1,n.minobsinnode = 10,n.trees=100)
gbm_model <- train(Class~., data=train, trControl=train_control, tuneGrid = gbmGrid,method="gbm")

## Print the results
gbm_model$results



## lets test
gbm_predict = predict(gbm_model,test,n.trees = 100)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,gbm_predict))


#### Grid Search
set.seed(123)
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 1)

### SVM
grid <- expand.grid(C = c(0.01, 1))

svm_model <- train(Class~., data=train, trControl=train_control, method="svmLinear",tuneGrid = grid,tuneLength = 2)

## Print the results
svm_model$results

## Plot
options(repr.plot.width = 7, repr.plot.height = 4)
svm_grid_stra=plot(svm_model)

## lets test
svm_predict = predict(svm_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,svm_predict)) 

### KNN
knn_model <- train(Class~., data=train, trControl=train_control, method="knn",tuneLength = 2,tuneGrid = expand.grid(k = c(5,4,3)))

## Print the results
knn_model$results

## Plot
options(repr.plot.width = 7, repr.plot.height = 4)
knn_grid_stra=plot(knn_model)

## lets test
knn_predict = predict(knn_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,knn_predict))

### Random Forest
train_control <- trainControl(method="repeatedcv", number=5, repeats=1)
mtry <- sqrt(ncol(train))
tunegrid <- expand.grid(.mtry=c(mtry,2))
rf_model <- train(Class~., data=train, trControl=train_control, method="rf",tuneGrid = tunegrid,ntree=100)

## Print the results
rf_model$results

## Plot
options(repr.plot.width = 7, repr.plot.height = 4)
rf_grid_stra = plot(rf_model)

## Variable Importance
options(repr.plot.width = 7, repr.plot.height = 5)
plot(varImp(rf_model))

## lets test
rf_predict = predict(rf_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,rf_predict))

### GBM
gbmGrid <-  expand.grid(interaction.depth = c(2,3),shrinkage = 0.1,n.minobsinnode = 10,n.trees = 100)

gbm_model <- train(Class~., data=train, trControl=train_control, method="gbm",tuneGrid = gbmGrid)

## Print the results
gbm_model$results

## Plot
trellis.par.set(caretTheme())
gbm_grid_stra=plot(gbm_model)

## lets test
gbm_predict = predict(gbm_model,test,n.trees = 100)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,gbm_predict)) 


## KFold

### ML
## Train & Test data
set.seed(124)

# Take 75% Train  & 25% test'
index = sample(1:nrow(covtype),0.75*nrow(covtype))
train <- covtype[index,]
test <- covtype[-index,]

set.seed(124)
train_control <- trainControl(method = "cv", number = 5)

### SVM
grid <- expand.grid(C = c(0.01, 1))

svm_model <- train(Class~., data=train, trControl=train_control, method="svmLinear",tuneGrid = grid,tuneLength = 2)

## Print the results
svm_model$results

## Plot
options(repr.plot.width = 7, repr.plot.height = 4)
svm_grid_kfold=plot(svm_model)

## lets test
svm_predict = predict(svm_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,svm_predict))

### KNN
knn_model <- train(Class~., data=train, trControl=train_control, method="knn",tuneLength = 2,tuneGrid = expand.grid(k = c(5,4,3)))

## Print the results
knn_model$results

## Plot
options(repr.plot.width = 7, repr.plot.height = 4)
knn_grid_kfold=plot(knn_model)

## lets test
knn_predict = predict(knn_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,knn_predict))

### Random Forest
train_control <- trainControl(method="repeatedcv", number=5, repeats=1)
mtry <- sqrt(ncol(train))
tunegrid <- expand.grid(.mtry=c(mtry,2))
rf_model <- train(Class~., data=train, trControl=train_control, method="rf",tuneGrid = tunegrid,ntree=100)

## Print the results
rf_model$results

## Plot
options(repr.plot.width = 7, repr.plot.height = 4)
rf_grid_kfold = plot(rf_model)

## Variable Importance
options(repr.plot.width = 7, repr.plot.height = 5)
plot(varImp(rf_model))

## lets test
rf_predict = predict(rf_model,test)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,rf_predict))

### GBM
gbmGrid <-  expand.grid(interaction.depth = c(2,3),shrinkage = 0.1,n.minobsinnode = 10,n.trees = 100)

gbm_model <- train(Class~., data=train, trControl=train_control, method="gbm",tuneGrid = gbmGrid)

## Print the results
gbm_model$results

## Plot
trellis.par.set(caretTheme())
gbm_grid_kold=plot(gbm_model)

## lets test
gbm_predict = predict(gbm_model,test,n.trees = 100)## Predict

## Model Accuracy
confusionMatrix(table(test$Class,gbm_predict))

options(repr.plot.width = 14, repr.plot.height = 15)
grid.arrange(svm_grid_stra,svm_grid_kfold,knn_grid_stra,knn_grid_kfold,rf_grid_stra,rf_grid_kfold,gbm_grid_stra,gbm_grid_kold,ncol=2,nrow=4)








