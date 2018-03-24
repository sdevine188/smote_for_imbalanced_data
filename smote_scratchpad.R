library(DMwR)
library(caret)
library(ipred)
library(ROCR)

# http://amunategui.github.io/smote/


# build data
hyper <-read.csv('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.data', header=F)
names <- read.csv('http://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/hypothyroid.names', header=F, sep='\t')[[1]]
names <- gsub(pattern =":|[.]",x = names, replacement="")
colnames(hyper) <- names

names
glimpse(hyper)

colnames(hyper) <-c("target", "age", "sex", "on_thyroxine", "query_on_thyroxine",
                    "on_antithyroid_medication", "thyroid_surgery", "query_hypothyroid",
                    "query_hyperthyroid", "pregnant", "sick", "tumor", "lithium",
                    "goitre", "TSH_measured", "TSH", "T3_measured", "T3", "TT4_measured",
                    "TT4", "T4U_measured", "T4U", "FTI_measured", "FTI", "TBG_measured",
                    "TBG")
hyper$target <- ifelse(hyper$target=='negative',0,1)

glimpse(hyper)


##########################################################

# check outcome balance
print(table(hyper$target))
print(prop.table(table(hyper$target)))


#########################################################


# clean up character variables, impute, etc
ind <- sapply(hyper, is.factor)
hyper[ind] <- lapply(hyper[ind], as.character)

hyper[ hyper == "?" ] = NA
hyper[ hyper == "f" ] = 0
hyper[ hyper == "t" ] = 1
hyper[ hyper == "n" ] = 0
hyper[ hyper == "y" ] = 1
hyper[ hyper == "M" ] = 0
hyper[ hyper == "F" ] = 1

hyper[ind] <- lapply(hyper[ind], as.numeric)

repalceNAsWithMean <- function(x) {replace(x, is.na(x), mean(x[!is.na(x)]))}
hyper <- repalceNAsWithMean(hyper)

glimpse(hyper)


###############################################################


# build train/test data
set.seed(1234)
splitIndex <- createDataPartition(hyper$target, p = .50,
                                  list = FALSE,
                                  times = 1)
trainSplit <- hyper[ splitIndex,]
testSplit <- hyper[-splitIndex,]

table(trainSplit$target)
prop.table(table(trainSplit$target))

table(testSplit$target)
prop.table(table(testSplit$target))


###############################################################


# build model
ctrl <- trainControl(method = "cv", number = 5)
tbmodel <- train(factor(target) ~ ., data = trainSplit, method = "rpart",
                 trControl = ctrl)
tbmodel

predictors <- names(trainSplit)[names(trainSplit) != 'target']
pred <- predict(tbmodel$finalModel, testSplit[,predictors])


###############################################################


# evaluate performance
tbmodel_prediction_obj <- prediction(predictions = pred[ , 2], labels = testSplit$target)
tbmodel_performance_obj <- performance(prediction.obj = tbmodel_prediction_obj, measure = "tpr", x.measure = "fpr")
plot(tbmodel_performance_obj)
abline(a=0, b= 1)

tbmodel_auc <- performance(prediction.obj = tbmodel_prediction_obj, measure = "auc")
tbmodel_auc@y.values # .8982034


##############################################################


# use SMOTE to oversample rare positive class, and undersample majority negative class
table(trainSplit$target)

# see help doc on smote for explanation of perc.over and perc.under
# basically, perc.over is intuitive as how many new synthetic samples from minority class
# perc.under is understood in reference to the original count of minority class
# so in this case the original minority/negative class had 79 obs, 
# so perc.under = 100 would return 79 obs for "majority"/positive class
# and perc.under = 200 returns 79x2 = 158 obs for positive class
trainSplit$target <- as.factor(trainSplit$target)
trainSplit <- SMOTE(target ~ ., trainSplit, perc.over = 100, perc.under=200)
trainSplit$target <- as.numeric(trainSplit$target)

table(trainSplit$target)
prop.table(table(trainSplit$target))


##########################################################


# rerun model on smote data
ctrl <- trainControl(method = "cv", number = 5)
tbmodel_smote <- train(factor(target) ~ ., data = trainSplit, method = "rpart",
                 trControl = ctrl)
tbmodel_smote

predictors <- names(trainSplit)[names(trainSplit) != 'target']
pred <- predict(tbmodel_smote$finalModel, testSplit[,predictors])


###############################################################


# evaluate performance
tbmodel_smote_prediction_obj <- prediction(predictions = pred[ , 2], labels = testSplit$target)
tbmodel_smote_performance_obj <- performance(prediction.obj = tbmodel_smote_prediction_obj, measure = "tpr", x.measure = "fpr")
plot(tbmodel_smote_performance_obj)
abline(a=0, b= 1)

tbmodel_smote_auc <- performance(prediction.obj = tbmodel_smote_prediction_obj, measure = "auc")
tbmodel_smote_auc@y.values # 0.9391245











