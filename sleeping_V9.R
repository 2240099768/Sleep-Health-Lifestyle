##1 Load data ---------------------
library(readr)
library(caret)
library(dplyr)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(neuralnet)
library(nnet)
library(e1071)
library(class)
library(reshape2)
library(RColorBrewer)
library(ggplot2)
library(corrplot)
library(MASS)
library(xgboost)
library(gains)

sleep_data <- read_csv("Sleep_health_and_lifestyle_dataset.csv")

##2 explore data ---------------------
# basic analysis
names(sleep_data)
head(sleep_data)
str(sleep_data)
summary(sleep_data)

#2.1 explore prediction target-sleep.disorder
unique_values <- unique(sleep_data$Sleep.Disorder)
cat(paste("The outputs from the classification are:", toString(unique_values)))
table(sleep_data$Sleep.Disorder)
#plot
ggplot(sleep_data, aes(x=Sleep.Disorder, fill=Sleep.Disorder)) +
  geom_histogram(stat="count", position=position_dodge(width=10)) +
  scale_fill_manual(values=c('#EBDEF0', '#4A235A', '#C39BD3')) +
  labs(title='Distribution of persons have sleep disorder or not') +
  theme_minimal() +
  theme(plot.background = element_rect(fill='white'),
        panel.background = element_rect(fill='white'),
        plot.title = element_text(size=12, hjust = 0,face="bold"),
        axis.title.x = element_text(size=10),
        axis.title.y = element_text(size=10),
        axis.text.x = element_text(size=10,face="bold"),
        axis.text.y = element_text(size=10,face="bold"),
        legend.title = element_text(size=10,face="bold"))

#2.2 gender
unique_gender <- unique(sleep_data$Gender)
cat(paste("The values of Sex column are:", toString(unique_gender)))
gender_disorder_counts <- sleep_data %>%
  group_by(Sleep.Disorder, Gender) %>%
  summarise(Count = n(),.groups = 'keep')
# Calculate the percentage for each gender within each Sleep.Disorder category
gender_disorder_counts <- sleep_data %>%
  group_by(Sleep.Disorder, Gender) %>%
  summarise(Count = n(), .groups = 'keep') %>%
  group_by(Sleep.Disorder) %>%
  mutate(Total = sum(Count)) %>%
  ungroup() %>%
  mutate(Percentage = Count / Total * 100)
# Plot
ggplot(gender_disorder_counts, aes(x="", y=Percentage, fill=Gender)) +
  geom_bar(stat="identity", width=1) +
  coord_polar("y", start=0) +
  facet_wrap(~ Sleep.Disorder) +
  scale_fill_manual(values=c('#C39BD3', '#EBDEF0')) +
  labs(title="The relationship between Sex and Sleep Disorder") +
  theme_void() +
  theme(legend.position="bottom")  # Adjust legend position if desired


##3 data pre-process ---------------------

##3.1 check missing-values
sum(is.na(sleep_data))
#drop person.ID
sleep_data <- sleep_data[, !names(sleep_data) %in% 'Person.ID']
#drop Heart.Rate because everyone's heart rate is within normal range
sleep_data <- sleep_data[, !names(sleep_data) %in% 'Heart.Rate']
#1:"sleep apnea" or "insomnia," and  0: no sleep problems
sleep_data$Sleep.Disorder <- as.numeric(sleep_data$Sleep.Disorder %in% c("Sleep Apnea", "Insomnia"))

##3.2 transfer Blood.Pressure
unique(sleep_data$Blood.Pressure)
#categorize blood pressure
categorize_blood_pressure <- function(bp) {
  bp_parts <- strsplit(bp, split = "/")
  sys <- as.numeric(bp_parts[[1]][1])
  dias <- as.numeric(bp_parts[[1]][2])
  if (sys < 90 || dias < 60) {
    result <- 'Low'
  } else if (sys < 120 && dias < 80) {
    result <- 'Normal'
  } else if (sys >= 120 && sys < 130 && dias < 80) {
    result <- 'Elevated'
  } else if ((sys >= 130 && sys < 140) || (dias >= 80 && dias < 90)) {
    result <- 'Hypertension Stage 1'
  } else if (sys >= 140 || dias >= 90) {
    result <- 'Hypertension Stage 2'
  } else if (sys > 180 || dias > 120) {
    result <- 'Hypertensive Crisis'
  } else {
    result <- 'Unknown'
  }
  return(result)
}
sleep_data$Blood.Pressure.Category <- sapply(sleep_data$Blood.Pressure, categorize_blood_pressure)
print(unique(sleep_data$Blood.Pressure.Category))

#3.3 trasnfer Age into categoriable variable
# categorize age
sleep_data$Age_Group <- cut(
  x = sleep_data$Age,
  breaks = c(0, 16, 30, 45, 100),
  labels = c('Child', 'Young Adults', 'Middle-aged Adults', 'Old Adults'),
  include.lowest = TRUE # Include the lowest value in the first bin
)
head(sleep_data$Age_Group)

#3.4 BMI.Category data-cleaning
sleep_data$BMI.Category[sleep_data$BMI.Category == "Normal Weight"] <- "Normal"
unique(sleep_data$BMI.Category)

#3.5 cut separate processing
sleep_data$Daily.Steps=cut(sleep_data$Daily.Steps,4)
sleep_data$Sleep.Duration=cut(sleep_data$Sleep.Duration,3)
sleep_data$Physical.Activity.Level=cut(sleep_data$Physical.Activity.Level,4)

#3.6 Use tag encoder
categories <- c('Gender','Occupation','Sleep.Duration',
                'Physical.Activity.Level','BMI.Category',
                'Daily.Steps','Age_Group',
                'Blood.Pressure.Category')
for (label in categories) {
  sleep_data[[label]] <- as.numeric(as.factor(sleep_data[[label]]))
}

#3.7 delete Age and Blood.Pressure column, then all the variable are numeric and classified
sleep_data1 <- sleep_data[,-c(2,9)]

#3.8 corrplot
cor_matrix <- cor(sleep_data1)
color_palette <- colorRampPalette(c("pink",'white', "#4A235A"))(200)
corrplot(cor_matrix, type = "upper", order = "hclust",
         tl.col = "black", tl.srt = 45, col = color_palette,
         tl.cex = 0.9)

##4 split data ---------------------
set.seed(20231124)
train_index <- sample(seq_len(nrow(sleep_data1)), size = floor(0.60 * nrow(sleep_data1)))
train_data <- sleep_data1[train_index, ]
x_train <- as.matrix(train_data[, -which(names(train_data) == "Sleep.Disorder")])
y_train <- train_data$Sleep.Disorder

test_data <- sleep_data1[-train_index, ]
x_test <- as.matrix(test_data[, -which(names(test_data) == "Sleep.Disorder")])
y_test <- as.numeric(test_data$Sleep.Disorder)

##5 Logistic Regression Model ---------------------
#Train
LR_model <- glm(Sleep.Disorder ~ ., data=train_data, family="binomial")
#Test
LR_predictions <- predict(LR_model, newdata = as.data.frame(x_test), type = "response")
LR_class_predictions <- ifelse(LR_predictions > 0.5, 1, 0)
#Evaluate
LR_confusion <- confusionMatrix(as.factor(LR_class_predictions), as.factor(as.numeric(y_test)))
LR_confusion
#the lift chart
LR_gain <- gains(test_data$Sleep.Disorder, LR_predictions, groups=10)
summary(LR_gain)
plot(LR_gain)
LR_preds <- as.numeric(LR_predictions>0.5)
LR_truth <- test_data$Sleep.Disorder
LR_probs <- LR_predictions
LR_preds <- as.numeric(LR_probs > 0.5)
LR_nick <- cbind(LR_truth, LR_probs, LR_preds)
# plot lift chart
plot(c(0,LR_gain$cume.pct.of.total*sum(test_data$Sleep.Disorder))~c(0,LR_gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0,sum(test_data$Sleep.Disorder))~c(0, dim(test_data)[1]), lty=2)

##6 XGBoost Model ---------------------
#Train
xgb_model <- xgboost(data = x_train, label = as.numeric(y_train), nrounds = 100, objective = "binary:logistic")
#Test
xgb_predictions <- predict(xgb_model, newdata = as.matrix(x_test))
xgb_class_predictions <- ifelse(xgb_predictions > 0.5, 1, 0)
# visualize result
importance_matrix <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix)
# adjust params
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta = 0.3,
  max_depth = 6
)
results <- data.frame(eta = numeric(), max_depth = integer(), test_auc_mean = numeric())
for (eta_val in seq(0.01, 0.3, by = 0.05)) {
  for (depth in 3:8) {
    params$eta <- eta_val
    params$max_depth <- depth

    cv_model <- xgb.cv(
      params = params,
      data = x_train,
      label = as.numeric(y_train),
      nrounds = 100,
      nfold = 5,
      metrics = "auc",
      early_stopping_rounds = 10,
      verbose = FALSE
    )

    best_iteration <- cv_model$best_iteration
    best_test_auc_mean <- cv_model$evaluation_log$test_auc_mean[best_iteration]
    results <- rbind(results, data.frame(eta = eta_val,
                                         max_depth = depth,
                                         test_auc_mean = best_test_auc_mean,
                                         nrounds = best_iteration))
  }
}
#plot result
ggplot(results, aes(x = max_depth, y = test_auc_mean, group = eta, color = as.factor(eta))) +
  geom_line() +
  geom_point() +
  labs(title = "Performance of Different max_depth and eta Combinations",
       x = "Max Depth",
       y = "Test AUC Mean",
       color = "Eta Value") +
  theme_minimal()
best_params <- params
best_params$eta <- 0.06
best_params$max_depth <- 5
# retrain best xgboost model
xgboost_model <- xgboost(
  params = best_params,
  data = x_train,
  label = as.numeric(y_train),
  nrounds = 100
)
#Evaluate
xgb_confusion <- confusionMatrix(as.factor(xgb_class_predictions), as.factor(as.numeric(y_test)))
xgb_confusion
#the lift chart
xgb_gain <- gains(test_data$Sleep.Disorder, xgb_predictions, groups=10)
summary(xgb_gain)
plot(xgb_gain)
xgb_preds <- as.numeric(xgb_predictions>0.5)
xgb_truth <- test_data$Sleep.Disorder
xgb_probs <- xgb_predictions
xgb_preds <- as.numeric(xgb_probs > 0.5)
xgb_nick <- cbind(xgb_truth, xgb_probs, xgb_preds)
# plot lift chart
plot(c(0,xgb_gain$cume.pct.of.total*sum(test_data$Sleep.Disorder))~c(0,xgb_gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0,sum(test_data$Sleep.Disorder))~c(0, dim(test_data)[1]), lty=2)
# plot the xgboost tree
xgb.plot.tree(model=xgboost_model, trees = 0)
xgb.plot.tree(model=xgboost_model, trees = 1)


##7 SVC Model ---------------------
#Train
svc_model <- svm(Sleep.Disorder ~ ., data = train_data)
#Test
svc_predictions <- predict(svc_model, newdata = x_test)
svc_class_predictions <- ifelse(svc_predictions > 0.5, 1, 0)
#Evaluate
svc_confusion <- confusionMatrix(as.factor(svc_class_predictions), as.factor(as.numeric(y_test)))
svc_confusion
#the lift chart
svc_gain <- gains(test_data$Sleep.Disorder, svc_predictions, groups=10)
summary(svc_gain)
plot(svc_gain)
svc_preds <- as.numeric(svc_predictions>0.5)
svc_truth <- test_data$Sleep.Disorder
svc_probs <- svc_predictions
svc_preds <- as.numeric(svc_probs > 0.5)
svc_nick <- cbind(svc_truth, svc_probs, svc_preds)
# plot lift chart
plot(c(0,svc_gain$cume.pct.of.total*sum(test_data$Sleep.Disorder))~c(0,xgb_gain$cume.obs),
     xlab="# cases", ylab="Cumulative", main="", type="l")
lines(c(0,sum(test_data$Sleep.Disorder))~c(0, dim(test_data)[1]), lty=2)

##8 Classification Tree ---------------------
#Train
train.ct <- rpart(Sleep.Disorder ~ .,data = train_data,method = "class",cp = 0,minsplit = 1)
#Plot
prp(train.ct,type = 1,extra = 1,split.font = 1,varlen = -10,box.palette = "#EBDEF0",border.col = "#4A235A")
#Inspect the cp table
train.ct$cptable
#Train- Predict
train.ct.pred <- predict(train.ct,train_data,type = "class")
confusionMatrix(train.ct.pred, as.factor(train_data$Sleep.Disorder), positive = "1")
#Test
test.ct.pred <- predict(train.ct,test_data,type = "class")
#Evaluate
confusionMatrix(test.ct.pred, as.factor(test_data$Sleep.Disorder), positive = "1")
# Cross-validation in RPART using xval
cv.ct <- rpart(Sleep.Disorder ~ ., data = sleep_data1, method = "class",
               cp = 0, minsplit = 1, xval = 10)
printcp(cv.ct)

##9 Random Forest ------------------------
#Train
train.rf <- randomForest(as.factor(Sleep.Disorder) ~ .,
                         data = train_data,
                         ntree = 100,
                         importance = TRUE)
# plot classification error
plot(train.rf$err.rate[,1], type = "l", xlab = "Number of Trees", ylab = "Classification Error")
#Train-decide ntree = 60
train.rf <- randomForest(as.factor(Sleep.Disorder) ~ .,
                         data = train_data,
                         ntree = 60,
                         importance = TRUE)
#Variable Importance Plot
varImpPlot(train.rf, type = 1, main="Variable Importance")
#Test
rf.pred <- predict(train.rf, test_data)
rf.pred1 <- predict(train.rf, test_data, type = "prob")
#Evaluate
confusionMatrix(rf.pred, as.factor(test_data$Sleep.Disorder), positive = "1")
#the lift chart
positive_prob <- rf.pred1[,2]
actual_numeric <- as.numeric(as.factor(test_data$Sleep.Disorder)) - 1  # 将因子转换为 0 和 1
rf_gain <- gains(actual_numeric, positive_prob, groups=10)
summary(rf_gain)
plot(c(0,rf_gain$cume.pct.of.total*sum(test_data$Sleep.Disorder)) ~ c(0,rf_gain$cume.obs),
     xlab="# Cases", ylab="Cumulative", main="Lift Chart for Random Forest", type="l")
lines(c(0,sum(test_data$Sleep.Disorder)) ~ c(0, nrow(test_data)), lty=2)



##10 KNN ----------------------
#knn model
knn.pred <- knn(train = train_data, test = test_data, cl = y_train, k = 2)
knn_cm <- confusionMatrix(data = knn.pred, reference = factor(y_test, levels = c("0", "1")))
print(knn_cm$table)
print(knn_cm)
#find best k
accuracy.df <- data.frame(k = seq(1, 20, 1), accuracy = rep(0, 20))
for(i in 1:20) {
  knn.pred <- knn(train = train_data, test = test_data, cl = y_train, k = i)
  accuracy.df[i, 2] <- confusionMatrix(factor(knn.pred , levels = c(0,1)),
                                       factor(factor(y_test, levels = c("0", "1")),
                                              levels = c(0,1)))$overall[1]
}
plot(accuracy.df[,1], accuracy.df[,2], type = "l",
     xlab = "k",
     ylab = "Accuracy")
#best k =3
#knn model- use best k
knn.pred <- knn(train = train_data, test = test_data, cl = y_train, k = 3)
knn_cm <- confusionMatrix(data = knn.pred, reference = factor(y_test, levels = c("0", "1")))
print(knn_cm$table)
print(knn_cm)

results_df <- data.frame(actual = as.numeric(as.factor(y_test)) - 1,
                         predicted = as.numeric(as.factor(knn.pred)) - 1)
results_df <- results_df[order(results_df$predicted, decreasing = TRUE),]
results_df$group <- cut(seq(1, nrow(results_df)), breaks = 10, labels = FALSE)
cumulative_response <- tapply(results_df$actual, results_df$group, function(x) sum(x))
cumulative_response_sum <- cumsum(cumulative_response)
plot(cumulative_response_sum, type = "o", xlab = "Group", ylab = "Cumulative Positive Responses",
     main = "kNN Model Response Chart")


##11 Naive Bayes ----------------------
#Train
nb <- naiveBayes(x_train, y_train)
#Test
nb_predictions <- predict(nb, x_test)
#Evaluate
nb_confusionMatrix <- table(Predicted = nb_predictions, Actual = y_test)
print(nb_confusionMatrix)
summary(nb_predictions)
# calculate sensitivity/specificity and accuracy
calculate_metrics <- function(confusionMatrix) {
  sensitivity <- confusionMatrix[1, 1] / (confusionMatrix[1, 1] + confusionMatrix[2, 1])
  specificity <- confusionMatrix[2, 2] / (confusionMatrix[2, 2] + confusionMatrix[1, 2])
  accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
  return(list(sensitivity = sensitivity, specificity = specificity, accuracy = accuracy))
}
nb_metrics <- calculate_metrics(nb_confusionMatrix)
print(nb_metrics)

##12 Linear Discriminant Analysis ----------------------
#Train
lda <- lda(x_train, grouping = y_train)
#Test
lda_predictions <- predict(lda, x_test)$class
#Evaluate
lda_confusionMatrix <- table(Predicted = lda_predictions, Actual = y_test)
print(lda_confusionMatrix)
lda_metrics <- calculate_metrics(lda_confusionMatrix)
print(lda_metrics)
