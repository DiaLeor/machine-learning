## Section 6.1 - Case Study: MNIST

# MNIST Case Study: Preprocessing -----------------------------------------

# Now that we have learned several methods and explore them with illustrative examples
# that weren't that interesting, we will apply what we have learned in the course on the
# Modified National Institute of Standards and Technology database (MNIST) digits, a
# popular dataset used in machine learning competitions. We can load this data using
# the following dslabs function like this.
library(tidyverse)
library(dslabs)
mnist <- read_mnist()

# The data set includes two components: a training set and a test set.
names(mnist)

# Each of these components includes a matrix with features in the columns and a vector with the class
# as integers.
dim(mnist$train$images)
table(mnist$train$labels)

# Because we want this example to run on a small laptop and in less than one hour, we will
# consider a subset of the dataset. We will sample 10,000 random rows from the training
# set and 1,000 random rows from the test set. You can use this code to do this:
set.seed(1990)
index <- sample(nrow(mnist$train$images), 10000)
x <- mnist$train$images[index,]
y <- factor(mnist$train$labels[index])

index <- sample(nrow(mnist$test$images), 1000)
x_test <- mnist$test$images[index,]
y_test <- factor(mnist$test$labels[index])

# Before running the machine learning algorithms, it's very common to first do what is
# called preprocessing. Common preprocessing steps include:
  # - standardizing or transforming predictors and
  # - removing predictors that are not useful, are highly correlated with others, have
  # very few non-unique values, or have close to zero variation.
  # - Other examples include taking the log transform of some predictors...

# As an example, we can run the nearZero() function from the caret package to see that
# several features do not vary much from observation to observation.
library(matrixStats)
sds <- colSds(x)
qplot(sds, bins = 256, color = I("black"))

# We can see that there's a large number of features with almost zero variability. This is
# expected because there are parts of the image that rarely contain writing. There's no dark
# pixels. The caret package includes a function that recommends features to be removed due
# to near-zero variance. You can type this code:
library(caret)
nzv <- nearZeroVar(x)

# We can see the columns recommended for removal by typing this command.
image(matrix(1:784 %in% nzv, 28, 28))

# After doing this, we only end up keeping 252 features.
col_index <- setdiff(1:ncol(x), nzv)
length(col_index)

# This has two advantages. First, we remove features that are not informative. Second,
# the algorithm will run faster.

# ..Code..
library(dslabs)
mnist <- read_mnist()

names(mnist)
dim(mnist$train$images)

class(mnist$train$labels)
table(mnist$train$labels)

# sample 10k rows from training set, 1k rows from test set
set.seed(1990)
index <- sample(nrow(mnist$train$images), 10000)
x <- mnist$train$images[index,]
y <- factor(mnist$train$labels[index])

index <- sample(nrow(mnist$test$images), 1000)
x_test <- mnist$test$images[index,]
y_test <- factor(mnist$test$labels[index])

library(matrixStats)
sds <- colSds(x)
qplot(sds, bins = 256, color = I("black"))

library(caret)
nzv <- nearZeroVar(x)
image(matrix(1:784 %in% nzv, 28, 28))

col_index <- setdiff(1:ncol(x), nzv)
length(col_index)

# MNIST Case Study: kNN ---------------------------------------------------

# After the preprocessing, we're ready to fit some models. Before we start, the caret package
# requires that we add column names to the feature matrices. You can do it like this:
colnames(x) <- 1:ncol(mnist$train$images)
colnames(x_test) <- colnames(x)

# Now let's start with kNN. The first step is to optimize for k. Keep in mind that when we run
# the algorithm, we'll have to compute a distance between each observation in the test set and
# each observation in the training set. And we need a lot of computations to do this. We will
# therefore use k-fold cross-validation to improve speed. If we run the following code, the
# computation time on a standard laptop will be several minutes:
control <- trainControl(method = "cv", number = 10, p = .9)
train_knn <- train(x[,col_index], y,
                   method = "knn", 
                   tuneGrid = data.frame(k = c(3,5,7)),
                   trControl = control)

# In general, it is a good idea to test out a small subset of the data first to get an idea of
# timing before we start running code that might take hours or days to complete. We can do this
# like this:
n <- 1000
b <- 2
index <- sample(nrow(x), n)
control <- trainControl(method = "cv", number = b, p = .9)
train_knn <- train(x[index ,col_index], y[index],
                   method = "knn",
                   tuneGrid = data.frame(k = c(3,5,7)),
                   trControl = control)

# This is an example of how we can test it out first. We can then increase n and b in the code
# we just showed to try to establish a pattern of how they affect computing time. And this will
# give us an idea of how long the fitting process will take for larger values of these two, n and b.
# You want to know if a function is going to take hours or even days before you run it.

# Once we know 3 is the optimal value for k, we can train our kNN algorithm using this code.
fit_knn <- knn3(x[ ,col_index], y,  k = 3)

# Note that we achieve very high accuracy.
y_hat_knn <- predict(fit_knn,
                     x_test[, col_index],
                     type="class")
cm <- confusionMatrix(y_hat_knn, factor(y_test))
cm$overall["Accuracy"]

# Now lets look into the results a little more. From the specificity and sensitivity, we also see
# that 8's are the hardest to detect. And the most common incorrectly predicted digit is 7. You
# can see that by typing this code:
cm$byClass[,1:2]

# ..Code..
# Note that your results may vary from those seen in the video as the seed was not set here.
colnames(x) <- 1:ncol(mnist$train$images)
colnames(x_test) <- colnames(x)

control <- trainControl(method = "cv", number = 10, p = .9)
train_knn <- train(x[,col_index], y,
                   method = "knn", 
                   tuneGrid = data.frame(k = c(3,5,7)),
                   trControl = control)

n <- 1000
b <- 2
index <- sample(nrow(x), n)
control <- trainControl(method = "cv", number = b, p = .9)
train_knn <- train(x[index ,col_index], y[index],
                   method = "knn",
                   tuneGrid = data.frame(k = c(3,5,7)),
                   trControl = control)
fit_knn <- knn3(x[ ,col_index], y,  k = 3)

y_hat_knn <- predict(fit_knn,
                     x_test[, col_index],
                     type="class")
cm <- confusionMatrix(y_hat_knn, factor(y_test))
cm$overall["Accuracy"]

cm$byClass[,1:2]

# MNIST Case Study: Random Forest -----------------------------------------

# Now let's see if we can do even better with an algorithm called random forest. You can learn
# about this algorithm in the course notes provided. With random forest, computation time is
# a challenge. We also have  several parameters we can tune. Because with random forest, the
# fitting is the slowest part of the procedure rather than the predicting, as with kNN, we
# will use only 5-fold cross-validation. We will change several other aspects of the algorithm
# just to make it run faster. But if you have time, you can run the full version of it. Here's
# the code implementing the version of the algorithm that finishes in a reasonable amount of time:
library(randomForest)
control <- trainControl(method="cv", number = 5)
grid <- data.frame(mtry = c(1, 5, 10, 25, 50, 100))
train_rf <-  train(x[, col_index], y,
                   method = "rf",
                   nTree = 150,
                   trControl = control,
                   tuneGrid = grid,
                   nSamp = 5000)

# Once that code is done running, we have optimized our algorithm and we are ready to fit our
# final model. We can do that like this:
fit_rf <- randomForest(x[, col_index], y,
                       minNode = train_rf$bestTune$mtry)

# We can see that with random forest, we achieve an even higher accuracy than we did with kNN.
# You can see the accuracy by typing this code:
y_hat_rf <- predict(fit_rf, x_test[ ,col_index])
cm <- confusionMatrix(y_hat_rf, y_test)
cm$overall["Accuracy"]

# With further tuning not discussed here, we can get an even higher accuracy.

# ..Code..
library(randomForest)
control <- trainControl(method="cv", number = 5)
grid <- data.frame(mtry = c(1, 5, 10, 25, 50, 100))
train_rf <-  train(x[, col_index], y,
                   method = "rf",
                   nTree = 150,
                   trControl = control,
                   tuneGrid = grid,
                   nSamp = 5000)

fit_rf <- randomForest(x[, col_index], y,
                       minNode = train_rf$bestTune$mtry)

y_hat_rf <- predict(fit_rf, x_test[ ,col_index])
cm <- confusionMatrix(y_hat_rf, y_test)
cm$overall["Accuracy"]


# MNIST Case Study: Variable Importance ------------------------------------

# Random forest is an example of what we call a black box model. Unlike methods such as
# linear regression, with random forest, it is hard or impossible to write down mathematical
# formulae to decipher how the features are used to predict. However, many of these methods,
# many of these algorithms provide approaches that summarize the influence of each feature.
# 
# The randomForest package supports variable importance calculations. The following code will
# give you the importance of each feature:
imp <- importance(fit_rf)
imp

# We can see which features are being used most by plotting an image of the result of this
# funciton, like this:
mat <- rep(0, ncol(x))
mat[col_index] <- imp
image(matrix(imp, 28, 28))

# We can see the locations on the image of the features that are being most. We can see that the
# outside edges of the image are not used as much, which makes sense because there is hardly ever
# any writing out there.

# We finish our lesson on how to use machine learning in practive by demonstrating the utility of
# using visual assessments to improve your algorithm. How we do this depends on the application.
# Here, we're going to show images of digits for which we made an incorrect prediction. We can
# compare what we get with kNN to random forests. This will help us realize if there's a mistake
# we're making that can maybe tell us how we can improve how we apply the algorithm.

# By examining errors like this, we often find specific weaknesses to algorithms or parameter
# choices, and we can try to correct them. It's important to note that machine learning algorithms
# rarely work out of the box. Preprocessing and fine-tuning are almost always necessary and highly
# important. Data visualization is often a powerful technique for evaluating and finding ways to
# improve.

# ..Code..
imp <- importance(fit_rf)
imp

mat <- rep(0, ncol(x))
mat[col_index] <- imp
image(matrix(imp, 28, 28))


p_max <- predict(fit_rf, x_test[,col_index], type = "prob") 
p_max <- p_max / rowSums(p_max)
p_max <- apply(p_max, 1, max)

ind  <- which(y_hat_rf != y_test)
ind <- ind[order(p_max[ind], decreasing = TRUE)]

rafalib::mypar(1,4)
for(i in ind[1:4]){
  image(matrix(x_test[i,], 28, 28)[, 28:1], 
        main = paste0("Pr(",y_hat_rf[i],")=",round(p_max[i], 2),
                      " but is a ",y_test[i]),
        xaxt="n", yaxt="n")
}

# Ensembles ---------------------------------------------------------------

# A very powerful approach in machine learning is the idea of ensembling different machine
# algorithms into one model to improve predictions. The idea of an ensemble is similar to
# the idea of combining data from different pollsters to obtain a better estimate of the
# true support for different candidates. In machine learning, one can usually greatly improve
# the final result of our predictions by combining the results of different algorithms. Here,
# we present a very simple example, where we compute new class probabilities by taking the
# average of the class probabilities provided by random forest and k-nearest neighbors.

# We can use the code below to simply average these probabilities. And we can see that once we
# do this, when we form the prediction, we actually improve the accuracy over both k-nearest
# neighbors and random forest.

# Now notice in this very simple example, we ensemble just two methods. In practice, we might
# ensemble dozens or even hundreds of different methods. And this really provides substantial
# improvements.

# ..Code..
p_rf <- predict(fit_rf, x_test[,col_index], type = "prob")
p_rf <- p_rf / rowSums(p_rf)
p_knn <- predict(fit_knn, x_test[,col_index])
p <- (p_rf + p_knn)/2
y_pred <- factor(apply(p, 1, which.max)-1)

confusionMatrix(y_pred, y_test)$overall["Accuracy"]