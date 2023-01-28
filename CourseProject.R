library(dplyr)
library(caret)
library(corrplot)
library(rpart.plot)
library(rattle)
library(randomForest)

############################################################
##################### Load Data ############################
############################################################
dataPath <-paste('/home/ubuntu/Documents/Courses/',
                 'Data Science - Specialization/',
                 'Practical Machine Learning/',
                 '/Practical-Machine-Learning---Course-Project/', sep='')

training <- read.csv(paste(dataPath, 'pml-training.csv', sep=''))
testing <- read.csv(paste(dataPath, 'pml-testing.csv', sep=''))
           
rm(dataPath)

# Rename testing set to validation set
validation <- testing
rm(testing)

############################################################
################ Clean Data ################################
############################################################
# Exclude columns like name, time-stamp, etc.
excludeCols <- c(1:7, 160)
training2 <- training[-excludeCols]
rm(excludeCols)

# Change char columns to numeric columns and NA <- 0
training2 <- training2 %>% mutate_if(is.character, as.numeric)
training2[is.na(training2)]<-0

# Remove columns with minimal variance
training2 <- select(training2,-nearZeroVar(training2))

# Remove columns to reduce pairwise correlation
corTraining <- cor(training2)
corCols <- findCorrelation(corTraining)
training2 <- training2[,-corCols]
rm(corTraining, corCols)

# Add output vector back to training2
training2$classe <- as.factor(training$classe)

############################################################
########### Create Training and Test Sets ##################
############################################################
train <- training2 %>% sample_frac(0.70)
test  <- anti_join(training2, train)
rm(training)
rm(training2)

############################################################
###### Training Control for 4-fold cross-validation ########
############################################################
set.seed(12321)
trCtl <- trainControl(method = 'cv', number = 4)
rpCtl <- rpart.control(minsplit = 1000, xval = 4)

############################################################
################ Decision Tree Model #######################
############################################################
modelDecTree <- rpart(classe ~ ., data = train, 
                      control = rpCtl, method = "class")
fancyRpartPlot(modelDecTree)

predDecTree <- predict(modelDecTree, test, type = "class")
cmDecTree <- confusionMatrix(predDecTree, test$classe)

############################################################
################ Random Forest Model #######################
############################################################

modelRF <- randomForest(classe~., data = train, ntree = 500, 
                        na.action = na.fail, trControl = trCtl)

predRF <- predict(modelRF, test, type='class')
cmRF <- confusionMatrix(predRF, test$classe)

############################################################
######### Generalised Boosted Regression Model #############
############################################################
modelGBM <- train(classe~., data = train, method = 'gbm',
                  trControl = trCtl, verbose = F)

predGBM <- predict(modelGBM, test)
cmGBM <- confusionMatrix(predGBM, test$classe)

rm(trCtl)

############################################################
############### Predict Using Best Model ###################
############################################################
results <- predict(modelRF, validation)