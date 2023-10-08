## Section 2.1 - Basics of Evaluation Machine Learning Algorithms

# Evaluation Metrics ------------------------------------------------------

# Before describing approaches to optimize the way we build algorithms, we first need to
# mathematically define what we mean when we say one approach is better than another.
# Specifically, we need to quantify what we mean by "better." For our first intro to machine
# learning (ml) concepts, we'll do a boring and simple example — how to predict sex using
# height.

# We use the height data from the dslabs package. We start by defining the outcomes and
# predictors.
 y <- heights$sex # only one predictor, a categorical outcome (male or female)
 x <- heights$height
# We know that we will not be able to predict y very accurately based on just x because male
# and female average heights are not that different relative to within group variability. But
# can we do better than guessing? Again, to answer this question, we need a quantitative
# definition of "better."

# Ultimately, a ml algorithm is evaluated on how it performs in the real world with
# completely new data sets. However, when developing an algorithm, we usually have a data set
# for which we know the outcomes (as we do with the heights data set).
 
# To mimic the ultimate evaluation process, we randomly split our data into two — a training set and
# a test set — and act as if we don't know the outcome of the test set. We develop algorithms using
# only the training set; the test set is used only for evaluation.
  # data used to develop algorithm: training set
  # data used to pretend we don't know the outcome: test set  

# The createDataPartition() function from the caret package can be used to generate indices for
# randomly splitting data into training and test sets.
set.seed(2007)
test_index <- createDataPartition(y, times = 1, p= 0.5, list = FALSE)
# Note: the set.seed() function is used to obtain reproducible results. This course requires a R
# version of 3.6 or newer to obtain the same results when setting the seed.
# The argument times is used to define how many random samples of indices to return. The
# argument p is used to define what proportion of the data is represented by the index. And the
# argument list is used to decide if we want the indices to return as a list or not.
# Note: contrary to what the documentation says, this course will use the argument p as the
# percentage of data that goes to testing. 

# The results of the createDataPartition() call can be used to definte the training and test set.
# Indices should be created on the outcome and not a predictor:
test_set <- heights[test_index, ]
train_set <- heights[-test_index, ]

# We then develop an algorithm using only the training set. And once we're done developing the
# algorithm, we will freeze it and evaluate it on the test set.

# The simplest evaluation metric for categorical outcomes is overall accuracy: the proportion of
# cases that were correctly predicted in the test set. This metric is usually referred to as
# overall accuracy.

# To demonstrate the use of overall accuracy, we can build two competing algorithms and compare
# them. We start by developing the simplest possible ml algorithm — guessing the outcome:
y_hat <- sample(c("Male", "Female"), length(test_index), replace = TRUE) %>% 
  factor(levels = levels(test_set$sex))
# Note that we're completely ignoring the predictor and simply guessing the sex. The overall
# accuracy is simply defined as the overall proportion that is predicted correctly, which we can
# calculate like this:
mean(y_hat == test_set$sex)
# Not surprisingly, our accuracy is about 50% — we're guessing. Now, can we do better?

# Exploratory data analysis suggest we can because, on average, males are slightly taller than
# females. We can see that by just typing this code:
heights %>% group_by(sex) %>% summarize(mean(height), sd(height))
# But how do we make use of this insight?

# Another simple approach. Predict male if height is within two sd from the average male height.
# So if height > 62, we will predict male, otherwise female. We can use this code to do that:
y_hat <- ifelse(x > 62, "Male", "Female") %>% factor(levels = levels(test_set$sex))
# The accuracy goes up from 0.5 with our previous guessing algorithm to about 80%, as you can
# see here:
mean(y == y_hat)
# But can we do even better? In the example that we just saw, we use a cutoff of 62. But can we
# examine the accuracy obtained for other cutoffs and then pick the value that provides the best
# result?

# Remember, it is important that we optimize the cutoff using only the training set. The test set
# is only for evaluation (although for this simplistic example it doesn't make much of a
# difference)! Later, we will learn that evaluating an algorithm on the training set can lead
# to overfitting for more complex approaches. This often leads to overoptimistic assessments.

# So let's try to build this algorithm. We will optimize our approach by examining the
# accuracy of 10 different cutoffs and picking the one yielding the best result. You can write
# this code to do that:
cutoff <- seq(61, 70)
accuracy <- map_dbl(cutoff, function(x){
  y_hat <- ifelse(train_set$height > x, "Male", "Female") %>% 
    factor(levels = levels(test_set$sex))
  mean(y_hat == train_set$sex)
})

# We can then make a plot showing the accuracy obtained for the different cutoffs.
data.frame(cutoff, accuracy) %>% 
  ggplot(aes(cutoff, accuracy)) + 
  geom_point() + 
  geom_line() 

# We see that the maximum accuracy achieved is about 85%, which is much higher than 50%.
max(accuracy)

# The selected cutoff that gave us this accuracy was 64. We can find it this way:
best_cutoff <- cutoff[which.max(accuracy)]
best_cutoff

# We can now test this cutoff on our test set to make sure accuracy is not overly optimistic.
y_hat <- ifelse(test_set$height > best_cutoff, "Male", "Female") %>% 
  factor(levels = levels(test_set$sex))
y_hat <- factor(y_hat)
mean(y_hat == test_set$sex)
# We find that we get an 80% accuracy. We see that our accuracy is a bit lower than the
# accuracy observed on the training set, but it's still better than guessing.

# Now because we tested on a data set that we did not train on, we know that our result, our
# 80% result, is not due to cherry picking a good result. This is what we refer to as
# overtraining. We will learn more about this in future videos.

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

# The prediction rule we developed predicts male if the student is taller than 64 inches. Given that
# the average female is about 64 inches, this prediction rule seems wrong. What happened? If a student
# is the height of the average female, shouldn't we predict female? Generally speaking, overall
# accuracy can sometimes be a deceptive measure because of unbalanced classes. To see this, we will
# start by constructing what is referred to as the confusion matrix, which basically tabulates each
# combination of prediction and actual value. You can create a confusion matrix in R using the table()
# function (or the confusionMatrix() function from the caret package — the confusionMatrix() function
# will be covered in more detail later):
table(predicted = y_hat, actual = test_set$sex)
# A problem arises: if we compute the accuracy separately for each sex we get the following results:
test_set %>%
  mutate(y_hat = y_hat) %>% 
  group_by(sex) %>% 
  summarize(accuracy = mean(y_hat == sex))
# There is an imbalance in the accuracy for males and females. In other words, too many females are
# predicted to be male. So how can our overall accuracy be so high then? The reason is that these
# heights were collected from three data science courses, two of which had more males enrolled.
# You can see that there's only 22% females in the aggregated data.
prev <- mean(y == "Female")
prev
# So when computing overall accuracy, the high percentage of mistakes made for females is outweighed
# by the gains in correct calls for males. Think of an extreme example where 99% of the data are males.
# Then a strategy that always guesses males no matter what will obviously give us a high accuracy.
# But with a more representative data set, the strategy will be as bad as guessing. This can actually
# be a big problem in ml.

# If your training data is biased in some way, you are likely to develop algorithms that are biased
# as well. The problem of biased training sets is so common that there are groups dedicated to
# study it. A first step to discovering these types of problems is to look at metrics other than
# overall accuracy when evaluating a ml algorithm. There are several metrics that we can use to
# evaluate an algorithm in a way that prevalence does no cloud our assessment. And these can all be
# derived from the confusion matrix. A general improvement to using overall accuracy is to study
# sensitivity and specificity separately.

# ..Code..
# tabulate each combination of prediction and actual value
table(predicted = y_hat, actual = test_set$sex)
test_set %>% 
  mutate(y_hat = y_hat) %>%
  group_by(sex) %>% 
  summarize(accuracy = mean(y_hat == sex))

# calculate the percentage of females in the aggregated data
prev <- mean(y == "Female")
prev

# Sensitivity, Specificity, and Prevalence --------------------------------

# We define sensitivity and specificity for binary outcomes. When the outcomes are categorical, we can
# define these terms for a specific category. For example, in the digits problem, we can ask for the
# specificity in the case of correctly predicting a 2 as opposed to some other digit. Now, once we
# specify a category of interest, then we can talk about
# positive outcomes: Y = 1, and
# negative outcomes: Y = 0.
# Note that the words positive and negative are not to be interpreted as in the English language.
# Negative doesn't necessarily imply bad. For example, in medical testing, a negative test is usually
# a good outcome.

# In general, sensitivity is defined as the ability of an algorithm to predict a positive outcome when
# the actual outcome is positive. Sensitivity, also known as the true positive rate or recall, is the
# proportion of actual positive outcomes correctly identified as such: Y_hat = 1 when Y = 1. High
# sensitivity  implies that Y = 1 ⟹ Y_hat = 1 (read as Y = 1 implies Y_hat will be 1).
# Because an algorithm that calls everything positive has perfect sensitivity, this metric on its own
# is not enough to judge an algorithm. For this reason, we also examine specificity.

# In general, specificity is defined as the ability of an algorithm to predict a negative outcome when
# the actual outcome is negative. Specificity, also known as the true negative rate, is the proportion
# of actual negative outcomes that are correctly identified as such: Y_hat = 0 when Y = 0. High
# specificity implies that Y = 0 ⟹ Y_hat = 0 (read as Y = 0 implies Y_hat will be 0).
# Specificity can ALSO be thought of as the proportion of positive calls that are actually
# positive: that is, high specificity also implies that Y = 1 ⟹ Y_hat = 1 (read as Y = 1 
# implies Y_hat will be 1).

# To provide precise definitions, we start by naming the four entries of the confusion matrix like
# this (there is a small table provided in the video which is not included here):
# Actually positive & Predicted positive = True positive (TP)
# Actually negative & Predicted positive = False positive (FP)
# Actually positive & Predicted negative = False negative (FN)
# Actually negative & Predicted negative = True negative (TN)

# Sensitivity is typically quantified by (TP)/(TP + FN), in other words: the proportion of actual
# positives (TP + FN) that are called positives (TP). This quantity is also called the true positive
# rate (TPR) or recall.

# Specificity is typically quantified by (TN)/(TN + FP), in other words: the proportion of actual
# negatives (TN + FP) that are called negatives (TN). This quantity is also called the true negative
# rate (TNR).

# However, there's another way of quantifying specificity: (TP)/(TP + FP), the proportion of outcomes
# called positives (TP + FP) that are actually positives (TP). This quantity is called the positive
# predictive value (PPV) or precision.

# Another important summary that can be extracted from the confusion matrix is the prevalence.
# Prevalence is defined as the proportion of positives. In our sex and height example, when we use
# overall accuracy, the fact that our prevalence, the proportion of females, was too low turned out
# to be a problem.
  ## NOTE that unlike the TPR and TNR, precision depends on prevalence, since higher
  ## # prevalence implies you can get higher precision even when guessing.

# There is a useful table included in the official course notes that is useful for remembering the
# terms. It includes a column that shows the definition if we think of the proportions as
# probabilities.

# The caret package function confusionMatrix() computes all these metrics for us once we define which
# category is considered the positives. The function expects factors as inputs. And the first level
# is considered the positive outcome. In our example, female is the first level because it comes before
# males alphabetically. Given this choice, using a new terminology, females are Y = 1 (positives) and
# males are Y = 0 (negatives).

# If we type the following code into R, we see several metrics including accuracy, sensitivity,
# specificity, and PPV. You can access this directly like this:
cm <- confusionMatrix(data = y_hat, reference = test_set$sex)
cm
# or like this:
cm$overall["Accuracy"]
cm$byClass[c("Sensitivity", "Specificity", "Prevalence")]
# In our example, we can see that the high overall accuracy is possible despite relatively low
# sensitivity. As we hinted at before, the reason this happens is because of low prevalence. It's
# only 23%. The proportion of females is low. Because prevalence is low, failing to predict actual
# females as females (low sensitivity) doesn't lower the accuracy as much as failing to predict
# actual males as males (low specificity). This is an example of why it is important to examine
# sensitivity and specificity and not just accuracy.
 
# Before applying this algorithm to general data sets, we need to ask ourselves if prevalence will be
# the same in new data sets out in the wild. 

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

