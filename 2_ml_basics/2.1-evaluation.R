## Section 2.1 - Basics of Evaluation Machine Learning Algorithms

# Evaluation Metrics ------------------------------------------------------

# Note: the set.seed() function is used to obtain reproducible results. This course requires a R
# version of 3.6 or newer to obtain the same results when setting the seed.

# To mimic the ultimate evaluation process, we randomly split our data into two — a training set and
# a test set — and act as if we don’t know the outcome of the test set. We develop algorithms using
# only the training set; the test set is used only for evaluation.

# The createDataPartition() function from the caret package can be used to generate indexes for
# randomly splitting data.

# Note: contrary to what the documentation says, this course will use the argument p as the
# percentage of data that goes to testing. The indexes made from createDataPartition() should
# be used to create the test set. Indexes should be created on the outcome and not a predictor.

# The simplest evaluation metric for categorical outcomes is overall accuracy: the proportion of
# cases that were correctly predicted in the test set.

# ..Code..
library(tidyverse)
library(caret)
library(dslabs)
data(heights)

# define the outcome and predictors
y <- heights$sex
x <- heights$height

# generate training and test sets
set.seed(2007)
test_index <- createDataPartition(y, times = 1, p = 0.5, list = FALSE)
test_set <- heights[test_index, ]
train_set <- heights[-test_index, ]

# guess the outcome
y_hat <- sample(c("Male", "Female"), length(test_index), replace = TRUE) %>% 
  factor(levels = levels(test_set$sex))

# compute accuracy
mean(y_hat == test_set$sex)

# compare heights in males and females in our data set
heights %>% group_by(sex) %>% summarize(mean(height), sd(height))

# now try predicting "male" if the height is within 2 SD of the average male
y_hat <- ifelse(x > 62, "Male", "Female") %>% factor(levels = levels(test_set$sex))
mean(y == y_hat)

# examine the accuracy of 10 cutoffs
cutoff <- seq(61, 70)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train_set$height > x, "Male", "Female") %>% 
    factor(levels = levels(test_set$sex))
  mean(y_hat == train_set$sex)
})
data.frame(cutoff, accuracy) %>% 
  ggplot(aes(cutoff, accuracy)) + 
  geom_point() + 
  geom_line() 
max(accuracy)

best_cutoff <- cutoff[which.max(accuracy)]
best_cutoff

y_hat <- ifelse(test_set$height > best_cutoff, "Male", "Female") %>% 
  factor(levels = levels(test_set$sex))
y_hat <- factor(y_hat)
mean(y_hat == test_set$sex)

# Confusion Matrix --------------------------------------------------------

# Overall accuracy can sometimes be a deceptive measure because of unbalanced classes.

# A general improvement to using overall accuracy is to study sensitivity and specificity separately.

# A confusion matrix tabulates each combination of prediction and actual value. You can create a
# confusion matrix in R using the table() function or the confusionMatrix() function from the caret
# package. The confusionMatrix() function will be covered in more detail later.

# If your training data is biased in some way, you are likely to develop algorithms that are biased
# as well. The problem of biased training sets is so commom that there are groups dedicated to
# study it.

# ..Code..
# tabulate each combination of prediction and actual value
table(predicted = y_hat, actual = test_set$sex)
test_set %>% 
  mutate(y_hat = y_hat) %>%
  group_by(sex) %>% 
  summarize(accuracy = mean(y_hat == sex))
prev <- mean(y == "Male")

confusionMatrix(data = y_hat, reference = test_set$sex)

# Sensitivity, Specificity, and Prevalence --------------------------------

# Sensitivity, also known as the true positive rate or recall, is the proportion of actual positive
# outcomes correctly identified as such: Y_hat = 1 when Y = 1. High sensitivity  means that
# Y = 1 ⟹ Y_hat = 1.

#Specificity, also known as the true negative rate, is the proportion of actual negative outcomes
#that are correctly identified as such: Y_hat = 0 when Y = 0. High specificity means that
#Y = 0 ⟹ Y_hat = 0.

# Specificity can also be thought of as the proportion of positive calls that are actually
# positive: that is, high specificity means that Y_hat = 1 ⟹ Y = 1.

# Sensitivity is typically quantified by (TP)/(TP + FN), the proportion of actual positives
# (TP + FN) that are called positives (TP). This quantity is also called the true positive
# rate (TPR) or recall.

# Specificity is typically quantified by (TN)/(TN + FP), the proportion of actual negatives
# (TN + FP) that are called negatives (TN). This quantity is also called the true negative
# rate (TNR).

# Specificity can also be quantified by (TP)/(TP + FP), the proportion of outcomes called
# positives (TP + FP) that are actually positives (TP). This quantity is called the positive
# predicitve value (PPV) or precision.

# Prevalence is defined as the proportion of positives.

# .. Code..
# get the metrics
cm <- confusionMatrix(data = y_hat, reference = test_set$sex)

# access specific metrics
cm$overall["Accuracy"]

cm$byClass[c("Sensitivity","Specificity", "Prevalence")]

# Balanced Accuracy and F1 Score ------------------------------------------

# For optimization purposes, sometimes it is more useful to have a one number summary than
# studying both specificity and sensitivity. One preferred metric is balanced accuracy. Because
# specificity and sensitivity are rates, it is more appropriate to compute the harmonic average.
# In fact, the F1-score, a widely used one-number summary, is the harmonic average of precision
# and recall. 

# Depending on the context, some type of errors are more costly than others. The F1-score can
# be adapted to weigh specificity and sensitivity differently. 

# You can compute the F1-score using the F_meas() function in the caret package.

# ..Code..
# maximize F-score
cutoff <- seq(61, 70)
F_1 <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train_set$height > x, "Male", "Female") %>% 
    factor(levels = levels(test_set$sex))
  F_meas(data = y_hat, reference = factor(train_set$sex))
})

data.frame(cutoff, F_1) %>% 
  ggplot(aes(cutoff, F_1)) + 
  geom_point() + 
  geom_line()

max(F_1)

best_cutoff_2 <- cutoff[which.max(F_1)]
best_cutoff_2

y_hat <- ifelse(test_set$height > best_cutoff_2, "Male", "Female") %>% 
  factor(levels = levels(test_set$sex))
sensitivity(data = y_hat, reference = test_set$sex)
specificity(data = y_hat, reference = test_set$sex)

# Prevalence Matters in Practice ------------------------------------------

# A machine learning algorithm with very high sensitivity and specificity may not be useful
# in practice when prevalence is close to either 0 or 1. For example, if you develop an
# algorithm for disease diagnosis with very high sensitivity, but the prevalence of the disease
# is pretty low, then the precision of your algorithm is probably very low based on Bayes'
# theorem.

# ROC and Precision-Recall Curves -----------------------------------------

# A very common approach to evaluating accuracy and F1-score is to compare them graphically
# by plotting both. A widely used plot that does this is the receiver operating characteristic
# (ROC) curve. The ROC curve plots sensitivity (TPR) versus 1 - specificity, also known as the
# false positive rate (FPR).

# However, ROC curves have one weakness and it is that neither of the measures plotted depend on
# prevalence. In cases in which prevalence matters, we may instead make a precision-recall plot,
# which has a similar idea with ROC curve.

# ..Code..
# Note: seed is not set so your results may slightly vary from those shown in the video.
p <- 0.9
n <- length(test_index)
y_hat <- sample(c("Male", "Female"), n, replace = TRUE, prob=c(p, 1-p)) %>% 
  factor(levels = levels(test_set$sex))
mean(y_hat == test_set$sex)

# ROC curve
probs <- seq(0, 1, length.out = 10)
guessing <- map_df(probs, function(p){
  y_hat <- 
    sample(c("Male", "Female"), n, replace = TRUE, prob=c(p, 1-p)) %>% 
    factor(levels = c("Female", "Male"))
  list(method = "Guessing",
       FPR = 1 - specificity(y_hat, test_set$sex),
       TPR = sensitivity(y_hat, test_set$sex))
})
guessing %>% qplot(FPR, TPR, data =., xlab = "1 - Specificity", ylab = "Sensitivity")

cutoffs <- c(50, seq(60, 75), 80)
height_cutoff <- map_df(cutoffs, function(x){
  y_hat <- ifelse(test_set$height > x, "Male", "Female") %>% 
    factor(levels = c("Female", "Male"))
  list(method = "Height cutoff",
       FPR = 1-specificity(y_hat, test_set$sex),
       TPR = sensitivity(y_hat, test_set$sex))
})

# plot both curves together
bind_rows(guessing, height_cutoff) %>%
  ggplot(aes(FPR, TPR, color = method)) +
  geom_line() +
  geom_point() +
  xlab("1 - Specificity") +
  ylab("Sensitivity")

library(ggrepel)
map_df(cutoffs, function(x){
  y_hat <- ifelse(test_set$height > x, "Male", "Female") %>% 
    factor(levels = c("Female", "Male"))
  list(method = "Height cutoff",
       cutoff = x, 
       FPR = 1-specificity(y_hat, test_set$sex),
       TPR = sensitivity(y_hat, test_set$sex))
}) %>%
  ggplot(aes(FPR, TPR, label = cutoff)) +
  geom_line() +
  geom_point() +
  geom_text_repel(nudge_x = 0.01, nudge_y = -0.01)

# plot precision against recall
guessing <- map_df(probs, function(p){
  y_hat <- sample(c("Male", "Female"), length(test_index), 
                  replace = TRUE, prob=c(p, 1-p)) %>% 
    factor(levels = c("Female", "Male"))
  list(method = "Guess",
       recall = sensitivity(y_hat, test_set$sex),
       precision = precision(y_hat, test_set$sex))
})

height_cutoff <- map_df(cutoffs, function(x){
  y_hat <- ifelse(test_set$height > x, "Male", "Female") %>% 
    factor(levels = c("Female", "Male"))
  list(method = "Height cutoff",
       recall = sensitivity(y_hat, test_set$sex),
       precision = precision(y_hat, test_set$sex))
})

bind_rows(guessing, height_cutoff) %>%
  ggplot(aes(recall, precision, color = method)) +
  geom_line() +
  geom_point()
guessing <- map_df(probs, function(p){
  y_hat <- sample(c("Male", "Female"), length(test_index), replace = TRUE, 
                  prob=c(p, 1-p)) %>% 
    factor(levels = c("Male", "Female"))
  list(method = "Guess",
       recall = sensitivity(y_hat, relevel(test_set$sex, "Male", "Female")),
       precision = precision(y_hat, relevel(test_set$sex, "Male", "Female")))
})

height_cutoff <- map_df(cutoffs, function(x){
  y_hat <- ifelse(test_set$height > x, "Male", "Female") %>% 
    factor(levels = c("Male", "Female"))
  list(method = "Height cutoff",
       recall = sensitivity(y_hat, relevel(test_set$sex, "Male", "Female")),
       precision = precision(y_hat, relevel(test_set$sex, "Male", "Female")))
})
bind_rows(guessing, height_cutoff) %>%
  ggplot(aes(recall, precision, color = method)) +
  geom_line() +
  geom_point()

# Loss Function -----------------------------------------------------------

# The most commonly used loss function is the squared loss function. Because we often have a
# test set with many observations, say N, we use the mean squared error (MSE). In practice, we
# often report the root mean squared error (RMSE), which is the square root of MSE, because it
# is in the same units as the outcomes.

# If the outcomes are binary, both RMSE and MSE are equivalent to one minus accuracy.

# Note that there are loss functions other than the squared loss. For example, the Mean Absolute
# Error uses absolute values instead of squaring the errors. However, we focus on minimizing
# square loss since it is the most widely used.

