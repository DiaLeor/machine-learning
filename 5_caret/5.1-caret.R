## Section 5.1 - The Caret Package

# Caret Package -----------------------------------------------------------

# We have already learned about regression and k-nearest neighbors as machine learning
# algorithms. In the provided reading materials, you can learn several other algorithms.
# Many of these algorithms are implemented in R. However, they are distributed via different
# packages developed by different authors and often use different syntax. The caret package
# helps provide a uniform interface and standardized syntax for the many different machine
# learning packages in R. Note that caret does not automatically install the packages needed. The
# required packages for each method are described in the package manual.

# Here, we provide some simple examples showing how we use this incredibly helpful package. We will
# use the 2 or 7 example to illustrate. We load the data like this:
library(tidyverse)
library(dslabs)
data("mnist_27")

# The caret train() function lets us train different algorithms using simple syntax. For example,
# we're going to fit two algorithms typing the following:
library(caret)
train_glm <- train(y ~ ., method = "glm", data = mnist_27$train)
train_knn <- train(y ~ ., method = "knn", data = mnist_27$train)

# To make predictions, we can use the output of this function directly without needing to look at
# the specifics of the specific packages that implement these algorithms. Instead, we can learn
# how to obtain predictions using predict.train. The code looks the same for both methods:
y_hat_glm <- predict(train_glm, mnist_27$test, type = "raw")
y_hat_knn <- predict(train_knn, mnist_27$test, type = "raw")

# This permits us to quickly compare the algorithms. For example, we can compare the accuracy:
confusionMatrix(y_hat_glm, mnist_27$test$y)$overall[["Accuracy"]]
confusionMatrix(y_hat_knn, mnist_27$test$y)$overall[["Accuracy"]]

# When an algorithm includes a tuning parameter, train() automatically uses cross-validation
# to decide among a few default values that it tries. To find out what parameter or parameters
# are optimized, you can read the manual or study the output of the the getModelInfo() and
# modelLookup() functions:
getModelInfo("knn")
modelLookup("knn")

# If you run it with default values, you can quickly see the result of the cross-validation using
# the ggplot() function.
ggplot(train_knn, highlight = TRUE)
  # the argument 'highlight' highlights the max

# By default, the cross-validation is performed taking 25 bootstrap samples comprised of 25% of
# the observations. For the kNN method, the default is to try the parameters k = 5, 7, and 9. We
# can change this using the tune grid parameter in the train() function to select a grid of values to
# be compared.

# The grid of values must be supplied by a data frame with the parameter names as specified in the
# modelLookup() output. Here, we present an example where we try out 30 values between 9 and 67. To
# do this with caret, we need to define a column named k:
data.frame(k = seq(9, 67, 2))
  # NOTE that when running this code, we're fitting 30 versions of kNN to 25 bootstrap samples.
  # Since we're filling 30*25, or 750, kNN models, running the following code (which implements all
  # this) will take several seconds.
set.seed(2008)
train_knn <- train(y ~ ., method = "knn",
                   data = mnist_27$train,
                   tuneGrid = data.frame(k = seq(9, 71, 2)))
ggplot(train_knn, highlight = TRUE)

# The ggplot() function shows us the accuracy for all the parameters we tried. To access the parameter
# that maximizes the accuracy, you can access it using the bestTune component of the output of the
# train() function:
train_knn$bestTune
# and the best performing model like this:
train_knn$finalModel

# The function predict() will use this best performing model. Here is the accuracy for the best
# model when applied to the test set, which we have not used at all yet because the cross-validation
# was done on the training set.
confusionMatrix(predict(train_knn, mnist_27$test, type = "raw"),
                mnist_27$test$y)$overall["Accuracy"]

# If we want to change how we perform cross-validation, we can use the trControl parameter and the
# trainControl() function. We can make the code we just saw a bit faster by using, for example,
# 10-fold cross-validation. This means we have 10 samples using 10% of the observations each.
control <- trainControl(method = "cv", number = 10, p = .9)
train_knn_cv <- train(y ~ ., method = "knn",
                      data = mnist_27$train,
                      tuneGrid = data.frame(k = seq(9, 71, 2)),
                      trControl = control)
ggplot(train_knn_cv, highlight = TRUE)
  # We notice that the accuracy estimates are more variable, which is expected since we changed
  # the number of samples used to estimate accuracy. NOTE that the results component of the 
  # train output includes several summary statistics related to the variability of the cross-
  # validation estimate.
names(train_knn_cv$results)

# Now, let's look at the best fitting kNN model.
plot_cond_prob <- function(p_hat=NULL){
  tmp <- mnist_27$true_p
  if(!is.null(p_hat)){
    tmp <- mutate(tmp, p=p_hat)
  }
  tmp %>% ggplot(aes(x_1, x_2, z=p, fill=p)) +
    geom_raster(show.legend = FALSE) +
    scale_fill_gradientn(colors=c("#F8766D","white","#00BFC4")) +
    stat_contour(breaks=c(0.5),color="black")
}

plot_cond_prob(predict(train_knn, mnist_27$true_p, type = "prob")[,2])

# The best fitting kNN model approximates the true conditional probability. However, we do see
# that the boundary is somewhat wiggly. This is because kNN, like the basic bin smoother, does
# not use a kernel. To improve this, we could use another algorithm called loess.

# Note that not all parameters in machine learning algorithms are tuned. We use the
# train() function to only optimize parameters that are tunable.

# ..Code..
library(tidyverse)
library(dslabs)
data("mnist_27")

library(caret)
train_glm <- train(y ~ ., method = "glm", data = mnist_27$train)
train_knn <- train(y ~ ., method = "knn", data = mnist_27$train)

y_hat_glm <- predict(train_glm, mnist_27$test, type = "raw")
y_hat_knn <- predict(train_knn, mnist_27$test, type = "raw")

confusionMatrix(y_hat_glm, mnist_27$test$y)$overall[["Accuracy"]]
confusionMatrix(y_hat_knn, mnist_27$test$y)$overall[["Accuracy"]]

getModelInfo("knn")
modelLookup("knn")

ggplot(train_knn, highlight = TRUE)

data.frame(k = seq(9, 67, 2))

set.seed(2008)
train_knn <- train(y ~ ., method = "knn",
                   data = mnist_27$train,
                   tuneGrid = data.frame(k = seq(9, 71, 2)))

ggplot(train_knn, highlight = TRUE)

train_knn$bestTune
train_knn$finalModel

confusionMatrix(predict(train_knn, mnist_27$test, type = "raw"),
                mnist_27$test$y)$overall["Accuracy"]

control <- trainControl(method = "cv", number = 10, p = .9)
train_knn_cv <- train(y ~ ., method = "knn",
                      data = mnist_27$train,
                      tuneGrid = data.frame(k = seq(9, 71, 2)),
                      trControl = control)
ggplot(train_knn_cv, highlight = TRUE)

names(train_knn_cv$results)

plot_cond_prob <- function(p_hat=NULL){
  tmp <- mnist_27$true_p
  if(!is.null(p_hat)){
    tmp <- mutate(tmp, p=p_hat)
  }
  tmp %>% ggplot(aes(x_1, x_2, z=p, fill=p)) +
    geom_raster(show.legend = FALSE) +
    scale_fill_gradientn(colors=c("#F8766D","white","#00BFC4")) +
    stat_contour(breaks=c(0.5),color="black")
}

plot_cond_prob(predict(train_knn, mnist_27$true_p, type = "prob")[,2])

# Fitting with Loess ------------------------------------------------------

# By reading through the available models part of the manual, we see that we can use
# the gamLoess method. In this manual, we also see that we need to install the gam package
# if we have not done so already.
install.packages("gam")

# The "gam" package allows us to fit a model using the gamLoess method in caret. Then we
# see that we have two parameters to optimize. 
modelLookup("gamLoess")

# We will stick to a degree of one to keep it simple. But to try out different values of the
# span, we'll have to include a column in the table with the name degree so we can do this. So
# we'll define a grid like this:
grid <- expand.grid(span = seq(0.15, 0.65, len = 10), degree = 1)

# We will use the default cross-validation control parameters.
train_loess <- train(y ~ ., 
                     method = "gamLoess",
                     tuneGrid=grid,
                     data = mnist_27$train)
ggplot(train_loess, highlight = TRUE)

# We can see that this method performs similar to kNN. And we get very similar accuracy.
confusionMatrix(data = predict(train_loess, mnist_27$test), 
                reference = mnist_27$test$y)$overall["Accuracy"]

# This method also produces a smoother estimate of the conditional probability than kNN.
p1 <- plot_cond_prob(predict(train_loess, mnist_27$true_p, type = "prob")[,2])
p1


# ..Code..
install.packages("gam")
modelLookup("gamLoess")

grid <- expand.grid(span = seq(0.15, 0.65, len = 10), degree = 1)

train_loess <- train(y ~ ., 
                     method = "gamLoess",
                     tuneGrid=grid,
                     data = mnist_27$train)
ggplot(train_loess, highlight = TRUE)

confusionMatrix(data = predict(train_loess, mnist_27$test), 
                reference = mnist_27$test$y)$overall["Accuracy"]

p1 <- plot_cond_prob(predict(train_loess, mnist_27$true_p, type = "prob")[,2])
p1

# Trees and Random Forests ------------------------------------------------

# While we have explored machine learning algorithms such as kNN and Loess in the course thus
# far, tree-based methods such as classification and regression trees (CART) and random
# forest are other commonly-used machine learning methods you may encounter. Here, we
# present key points to introduce these methods along with a comprehension check to assess
# understanding of these methods.

# Some methods, such as LDA and QDA, are not meant to be used with many predictors p because
# the number of parameters needed to be estimated becomes too large.

# Curse of dimensionality: For kernel methods such as kNN or local regression, when they
# have multiple predictors used,  the span/neighborhood/window made to include a given
# percentage of the data become large. With larger neighborhoods, our methods lose flexibility.
# The dimension here refers to the fact that when we have p predictors, the distance between
# two observations is computed in p-dimensional space.

# A tree is basically a flow chart of yes or no questions. The general idea of the methods we
# are describing is to define an algorithm that uses data to create these trees with
# predictions at the ends, referred to as nodes.

# When the outcome is continuous, we call the decision tree a regression tree.

# When the outcome is categorical, we call the decision tree a classification tree.

# Decision trees operate by predicting an outcome variable Y by partitioning the predictors.

# The general idea here is to build a decision tree and, at end of each node, obtain a
# predictor y_hat. Mathematically, we are partitioning the predictor space into J
# non-overlapping regions, R_1, R_2,..., R_J and then for any predictor x that falls
# within region R_j, estimate f(x) with the average of the training observations y_i
# for which the associated predictor x_i is also in R_j.

# Two common parameters used for partition decision are the complexity parameter (cp) and
# the minimum number of observations required in a partition before partitioning it further
# (minsplit in the rpart package). 

# Classification trees form predictions by calculating which class is the most common among
# the training set observations within the partition, rather than taking the average in
# each partition as with regression trees.

# Two of the more popular metrics to choose the partitions are the Gini index and entropy.

  # Gini(j) = the sum of k = 1 to K of p_hat_j,k (1 - p_hat_j,k)

  # entropy(j) = negative, the sum of k = 1 to K of p_hat_j,k log(p_hat_j,k), with 0

# We can use the geom_step() function in ggplot to visualize the fit of our model when using
# only a single predictor. We can use the functions plot() and text() to display the
# decision process used by our tree whether including one or many predictors.

# Pros: Decision trees are highly interpretable and easy to visualize. They can model human
# decision processes and don’t require use of dummy predictors for categorical variables.

# Cons: The approach via recursive partitioning can easily over-train and is therefore a
# bit harder to train than kNN. Furthermore, in terms of accuracy, it is rarely the best
# performing method since it is not very flexible and is highly unstable to changes in
# training data. 

# Random forests are a very popular machine learning approach that addresses the
# shortcomings of decision trees. The goal is to improve prediction performance and reduce
# instability by averaging multiple decision trees (a forest of trees constructed with
# randomness).

# The general idea of random forests is to generate many predictors, each using regression
# or classification trees, and then forming a final prediction based on the average
# prediction of all these trees. To assure that the individual trees are not the same,
# we use the bootstrap to induce randomness. 

# A disadvantage of random forests is that we lose interpretability.

# An approach that helps with interpretability is to examine variable importance. To define
# variable importance we count how often a predictor is used in the individual trees. The
# caret package includes the function varImp that extracts variable importance from any
# model in which the calculation is implemented.

# ..Code..
# Load tidyverse
library(tidyverse)

# load package for decision tree
library(rpart)

# load the dslabs package
library(dslabs)

# fit a decision tree using the polls_2008 dataset, 
# which contains only one predictor (day)
# and the outcome (margin)
fit <- rpart(margin ~ ., data = polls_2008)

# display the decision tree
plot(fit, margin = 0.1)
text(fit, cex = 0.75)

# examine the fit from the decision tree model
polls_2008 %>%  
  mutate(y_hat = predict(fit)) %>% 
  ggplot() +
  geom_point(aes(day, margin)) +
  geom_step(aes(day, y_hat), col="red")

# fit a decision tree on the mnist data using cross validation
train_rpart <- train(y ~ .,
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.1, len = 25)),
                     data = mnist_27$train)
# and plot it
plot(train_rpart)

# compute accuracy
confusionMatrix(predict(train_rpart, mnist_27$test), mnist_27$test$y)$overall["Accuracy"]

# view the final decision tree
plot(train_rpart$finalModel, margin = 0.1) # plot tree structure
text(train_rpart$finalModel) # add text labels

# load library for random forest
library(randomForest)
train_rf <- randomForest(y ~ ., data=mnist_27$train)
confusionMatrix(predict(train_rf, mnist_27$test), mnist_27$test$y)$overall["Accuracy"]

# use cross validation to choose parameter
train_rf_2 <- train(y ~ .,
                    method = "Rborist",
                    tuneGrid = data.frame(predFixed = 2, minNode = c(3, 50)),
                    data = mnist_27$train)
confusionMatrix(predict(train_rf_2, mnist_27$test), mnist_27$test$y)$overall["Accuracy"]