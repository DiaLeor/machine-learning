library(titanic)    # loads titanic_train data frame
library(caret)
library(tidyverse)
library(rpart)

# 3 significant digits
options(digits = 3)

# clean the data - `titanic_train` is loaded with the titanic package
titanic_clean <- titanic_train %>%
  mutate(Survived = factor(Survived),
         Embarked = factor(Embarked),
         Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age), # NA age to median age
         FamilySize = SibSp + Parch + 1) %>%    # count family members
  select(Survived,  Sex, Pclass, Age, Fare, SibSp, Parch, FamilySize, Embarked)

head(titanic_clean)
str(titanic_clean)
ls(titanic_clean)

dat <- titanic_clean


# Question 1: Training and test sets --------------------------------------

# Split titanic_clean into test and training sets - after running the setup code, it should have
# 891 rows and 9 variables.

# Set the seed to 42, then use the caret package to create a 20% data partition based on the
# Survived column. Assign the 20% partition to test_set and the remaining 80% partition to train_set.

set.seed(42)
index <- createDataPartition(dat$Survived, times = 1, p = .20, list = FALSE)
test <- dat[index, ]
train <- dat[-index, ]

# How many observations are in the training set?
str(train)

# How many observations are in the test set?
str(test)

# What proportion of individuals in the training set survived?
mean(train$Survived == 1)

#..Answer Code..
set.seed(42) 
test_index <- createDataPartition(titanic_clean$Survived, times = 1, p = 0.2, list = FALSE)
# create a 20% test set
test_set <- titanic_clean[test_index,]
train_set <- titanic_clean[-test_index,]

nrow(train_set)
nrow(test_set)
mean(train_set$Survived == 1)

# Question 2: Baseline prediction by guessing outcome ---------------------

# The simplest prediction method is randomly guessing the outcome without using additional
# predictors. These methods will help us determine whether our machine learning algorithm
# performs better than chance. How accurate are two methods of guessing Titanic passenger survival?

# Set the seed to 3. For each individual in the test set, randomly guess whether that person
# survived or not by sampling from the vector c(0,1) (Note: use the default argument setting of
# prob from the sample function).
set.seed(3)
y_hat <- sample(c(0, 1), length(index), replace = TRUE) %>% 
  factor(levels = levels(test$Survived))

# What is the accuracy of this guessing method?
mean(y_hat == test$Survived)

#..Answer Code..
set.seed(3)
# guess with equal probability of survival
guess <- sample(c(0,1), nrow(test_set), replace = TRUE)
mean(guess == test_set$Survived)

# Question 3a: Predicting survival by sex ---------------------------------

# Use the training set to determine whether members of a given sex were more likely to
# survive or die.

# What proportion of training set females survived?
f_train <- train %>% filter(Sex == "female")
mean(f_train$Survived == 1)

# What proportion of training set males survived?
m_train <- train %>% filter(Sex == "male")
mean(m_train$Survived == 1)

#..Answer Code..
train_set %>%
  group_by(Sex) %>%
  summarize(Survived = mean(Survived == 1)) %>%
  filter(Sex == "female") %>%
  pull(Survived)

train_set %>%
  group_by(Sex) %>%
  summarize(Survived = mean(Survived == 1)) %>%
  filter(Sex == "male") %>%
  pull(Survived)

# Question 3b: Predicting survival by sex ---------------------------------

# Predict survival using sex on the test set: if the survival rate for a sex is over 0.5, 
# predict survival for all individuals of that sex, and predict death if the survival rate 
# for a sex is under 0.5.

# What is the accuracy of this sex-based prediction method on the test set?
y_hat <- ifelse(test$Sex == "female", 1, 0) %>% 
  factor(levels = levels(test$Survived))
mean(y_hat == test$Survived)

#..Answer Code..
sex_model <- ifelse(test_set$Sex == "female", 1, 0)    # predict Survived=1 if female, 0 if male
mean(sex_model == test_set$Survived)    # calculate accuracy

# Question 4a: Predicting survival by passenger class ---------------------

# In the training set, which class(es) (Pclass) were passengers more likely to survive than die?
# Note that "more likely to survive than die" (probability > 50%) is distinct from "equally likely
# to survive or die" (probability = 50%).

p1_train <- train %>% filter(Pclass == 1)
mean(p1_train$Survived == 1)

p2_train <- train %>% filter(Pclass == 2)
mean(p2_train$Survived == 1)

p3_train <- train %>% filter(Pclass == 3)
mean(p3_train$Survived == 1)

#..Answer Code..
train_set %>%
  group_by(Pclass) %>%
  summarize(Survived = mean(Survived == 1))


# Question 4b: Predicting survival by passenger class ---------------------

# Predict survival using passenger class on the test set: predict survival if the survival rate for
# a class is over 0.5, otherwise predict death.

# What is the accuracy of this class-based prediction method on the test set?
y_hat <- ifelse(test$Pclass == 1, 1, 0) %>% 
  factor(levels = levels(test$Survived))
mean(y_hat == test$Survived)

#..Answer Code..
class_model <- ifelse(test_set$Pclass == 1, 1, 0)    # predict survival only if first class
mean(class_model == test_set$Survived)    # calculate accuracy


# Question 4c: Predicting survival by passenger class ---------------------

# Use the training set to group passengers by both sex and passenger class.

# Which sex and class combinations were more likely to survive than die (i.e. >50% survival)?
train %>%
  group_by(Sex,Pclass) %>%
  summarize(Survived = mean(Survived == 1))

#..Answer Code..
train_set %>%
  group_by(Sex, Pclass) %>%
  summarize(Survived = mean(Survived == 1)) %>%
  filter(Survived > 0.5)

# Question 4d: Predicting survival by passenger class ---------------------
# Predict survival using both sex and passenger class on the test set. Predict survival if the survival
# rate for a sex/class combination is over 0.5, otherwise predict death.

# What is the accuracy of this sex- and class-based prediction method on the test set?
y_hat <- ifelse(test$Sex == "female" & test$Pclass != 3, 1, 0)
mean(y_hat == test$Survived)

#..Answer Code..
sex_class_model <- ifelse(test_set$Sex == "female" & test_set$Pclass != 3, 1, 0)
mean(sex_class_model == test_set$Survived)


# Question 5a: Confusion matrix -------------------------------------------

# Use the confusionMatrix() function to create confusion matrices for the sex model, class model,
# and combined sex and class model. You will need to convert predictions and survival status to
# factors to use this function.

y_hat_sex <- ifelse(test$Sex == "female", 1, 0) %>% 
  factor(levels = levels(test$Survived))
cm_sex <- confusionMatrix(data = y_hat_sex, reference = test$Survived)

y_hat_class <- ifelse(test$Pclass == 1, 1, 0) %>% 
  factor(levels = levels(test$Survived))
cm_class <- confusionMatrix(data = y_hat_class, reference = test$Survived)

y_hat_combo <- ifelse(test$Sex == "female" & test$Pclass != 3, 1, 0) %>% 
  factor(levels = levels(test$Survived))
cm_combo <- confusionMatrix(data = y_hat_combo, reference = test$Survived)

# What is the "positive" class used to calculate confusion matrix metrics?
# Which model has the highest sensitivity?
# Which model has the highest specificity?
# Which model has the highest balanced accuracy?

cm_sex
cm_class
cm_combo

#..Answer Code..
confusionMatrix(data = factor(sex_model), reference = factor(test_set$Survived))
confusionMatrix(data = factor(class_model), reference = factor(test_set$Survived))
confusionMatrix(data = factor(sex_class_model), reference = factor(test_set$Survived))


# Question 5b: Confusion matrix -------------------------------------------

# What is the maximum value of balanced accuracy from Q5a?

#..Answer Code..
# see Q5a


# Question 6: F1 scores ---------------------------------------------------

# Use the F_meas() function to calculate F_1 scores for the sex model, class model, and combined sex and
# class model. You will need to convert predictions to factors to use this function.

# Which model has the highest F_1 score?
F_meas(data = y_hat_sex, reference = factor(test$Survived))
F_meas(data = y_hat_class, reference = factor(test$Survived))
F_meas(data = y_hat_combo, reference = factor(test$Survived))

#..Answer Code..
F_meas(data = factor(sex_model), reference = test_set$Survived)
F_meas(data = factor(class_model), reference = test_set$Survived)
F_meas(data = factor(sex_class_model), reference = test_set$Survived)       