## Section 4.1 - k-Nearest Neighbors

# k-Nearest-Neighbors (kNN) -----------------------------------------------------

# Cross-validation is one of the most important ideas in machine learning. We focus on 
# conceptual and mathematical aspects and later describe how to implement it. To motivate
# the concept, we will use the two digit data presented earlier. And we introduce for the
# first time an actual machine learning algorithm, k-nearest neighbors.

# We start by loading the data and showing a plot of the predictors with the outcome
# represented with color. We can do it using this code:
library(tidyverse)
library(dslabs)
data("mnist_27")
mnist_27$test %>% 
  ggplot(aes(x_1, x_2, color = y)) +
  geom_point()

# We will use these data to estimate the conditional probability function as previously defined:
# p(x_1,x_2) = Pr(Y = 1 | X_1 = x_1, X_2 = x_2)
# K-nearest neighbors (kNN) estimates the conditional probabilities in a similar way to bin
# smoothing. However, kNN is easier to adapt to multiple dimensions. Using kNN, we define the
# distance between all observations based on the features. Then, for any point (x_1,x_2) for
# which we want an estimate of p(x_1,x_2), we look for the k nearest points to (x_1,x_2) and
# take an average of the 0s and 1s associated with these points. We refer to the set of points
# used to compute the average as the neighborhood.

# Due to the connection we described earlier between conditional expectation and conditional
# probabilities, this gives us an estimate of p(x_1,x_2) just like bin smoother gave us an
# estimate of a trend. As with bin smoothers, we can control the flexibility of our estimate,
# in this case, through the k parameter. Larger values of k result in smoother estimates, while
# smaller values of k result in more flexible and more wiggly estimates.

# To implement the algorithm, we can use the knn3() function from the caret package. There
# are two ways to call this function:

# 1. We can call the function with the first argument being the matrix predictors and
# the second a vector of outcomes, like this:
x <- as.matrix(mnist_27$train)
y <- mnist_27$train$y
knn_fit <- knn3(x,y)

# 2. We can specify a formula and a data frame.
# The data frame contains all the data to be used.
# The formula has the following form: outcome ~ predictor_1 + predictor_2 + predictor_3.
# Therefore, we would type y ~ x_1 + x_2.
# If we're going to use all the predictors, we can actually use a dot like this: y ~ .
# We also need to pick a parameter, the number of neighbors to include. Let's start by
# estimating the function with default k = 5.
library(caret)
knn_fit <- knn3(y ~ ., data = mnist_27$train, k = 5)

# In this case, since our data set is balanced and we care just as much about sensitivity
# as we do about specificity, we will use accuracy to quantify performance. The predict()
# function for knn3 produces a probability for each class. We keep the probabilities of being
# a 7 as the estimate of p_hat(x_1,x_2). You can get it like this:
y_hat_knn <- predict(knn_fit, mnist_27$test, type = "class")

# We also see by using the confusionMatrix() function that we get an accuracy of 81-82%.
confusionMatrix(y_hat_knn, mnist_27$test$y)$overall["Accuracy"]

# Previously, we used linear regression to generate an estimate for this same data set. The
# accuracy was 75%:
fit_lm <- mnist_27$train %>% 
  mutate(y = ifelse(y == 7, 1, 0)) %>% 
  lm(y ~ x_1 + x_2, data = .)
p_hat_lm <- predict(fit_lm, mnist_27$test)
y_hat_lm <-  factor(ifelse(p_hat_lm > 0.5, 7, 2))
confusionMatrix(y_hat_lm, mnist_27$test$y)$overall["Accuracy"]

# We see that kNN with just the default parameter already beats regression. To see why this
# is the case , we plot the estimates p_hat(x_1,x_2) and compare it to the true conditional
# probability, p(x_1,x_2), (see code below). And we see that kNN better adapts to the nonlinear shape of p(x_1,x_2).
# However, our estimate has some islands of blue in the red areas, which intuitively doesn't make
# sense. This is due to what we call over-training.

# Over-training is a reason that we have higher accuracy in the training set compared to the test
# set. We can see that by typing these commands:
y_hat_knn <- predict(knn_fit, mnist_27$train, type = "class")
confusionMatrix(y_hat_knn, mnist_27$train$y)$overall["Accuracy"]

y_hat_knn <- predict(knn_fit, mnist_27$test, type = "class")
confusionMatrix(y_hat_knn, mnist_27$test$y)$overall["Accuracy"]

# Over-training is when a machine learning model can make good predictions on the training
# set but cannot generalize well to new data. Over-training is a reason why we have higher
# accuracy in the training set compared to the test set.

# ..Code..
library(tidyverse)
library(caret)
library(dslabs)
library(gridExtra)

data("mnist_27")

mnist_27$test %>%
  ggplot(aes(x_1, x_2, color = y)) +
  geom_point()

knn_fit <- knn3(y ~ ., data = mnist_27$train)

y_hat_knn <- predict(knn_fit, mnist_27$test, type = "class")
confusionMatrix(y_hat_knn, mnist_27$test$y)$overall["Accuracy"]

fit_lm <- mnist_27$train %>% 
  mutate(y = ifelse(y == 7, 1, 0)) %>% 
  lm(y ~ x_1 + x_2, data = .)
p_hat_lm <- predict(fit_lm, mnist_27$test)
y_hat_lm <- factor(ifelse(p_hat_lm > 0.5, 7, 2))
confusionMatrix(y_hat_lm, mnist_27$test$y)$overall["Accuracy"]

# plot the estimates p_hat(x_1,x_2) and compare to p(x_1,x_2)
plot_cond_prob <- function(p_hat=NULL){
  tmp <- mnist_27$true_p
  if(!is.null(p_hat)){
    tmp <- mutate(tmp, p=p_hat)
  }
  tmp %>% ggplot(aes(x_1, x_2, z=p, fill=p)) +
    geom_raster(show.legend = FALSE) +
    scale_fill_gradientn(colors=c("#F8766D", "white", "#00BFC4")) +
    stat_contour(breaks=c(0.5), color="black")
}
p1 <- plot_cond_prob() +
  ggtitle("True conditional probability")
p2 <- plot_cond_prob(predict(knn_fit, mnist_27$true_p)[,2]) +
  ggtitle("kNN-5 estimate")
grid.arrange(p2, p1, nrow=1)

y_hat_knn <- predict(knn_fit, mnist_27$train, type = "class")
confusionMatrix(y_hat_knn, mnist_27$train$y)$overall["Accuracy"]

y_hat_knn <- predict(knn_fit, mnist_27$test, type = "class")
confusionMatrix(y_hat_knn, mnist_27$test$y)$overall["Accuracy"]

# Over-training and Over-smoothing ------------------------------------------

# With k-nearest neighbors (kNN), over-training is at its worst when we set k = 1. With k = 1,
# the estimate for each (x_1,x_2) in the training set is obtained with just the y corresponding
# to that point. In this case, if the x_1, x_2 are unique, we will obtain perfect accuracy in
# the training set because each point is used to predict itself. Remember that if the predictors
# are not unique and we have different outcomes for at least one set of predictors, then it is
# impossible to predict perfectly. Here we fit a kNN model with k = 1.
knn_fit_1 <- knn3(y ~ ., data = mnist_27$train, k = 1)
y_hat_knn_1 <- predict(knn_fit_1, mnist_27$train, type = "class")
confusionMatrix(y_hat_knn_1, mnist_27$train$y)$overall["Accuracy"]
  # NOTE: The accuracy on the training set is over 99%. However, the test set accuracy is
  # actually worse than regression:
y_hat_knn_1 <- predict(knn_fit_1, mnist_27$test, type = "class")
confusionMatrix(y_hat_knn_1, mnist_27$test$y)$overall["Accuracy"]

# We can see the over-training problem by plotting the estimates (see code below). The black
# curve denotes the decision rule boundaries. The estimate p_hat(x_1,x_2) follows the training
# data too closely. You can see that in the training set, boundaries have been drawn to perfectly
# surround each single red opint in a sea of blue. Because most points - x_1, x_2 - are unique,
# the prediction is either 1 or 0, and the prediction for that point is the associated label.
# However, once we introduce the test set, we see that many of the small islands now have the
# opposite color, and we end up making several incorrect predictions.

# Although not as bad with this example, earlier, we saw that with k = 5, we also have some over-
# training. Hence, we should consider larger values of k. Let's try, as an example, a much larger
# number with k = 401. With this, we get an accuracy in the test set of 79%.
knn_fit_401 <- knn3(y ~ ., data = mnist_27$train, k = 401)
y_hat_knn_401 <- predict(knn_fit_401, mnist_27$test, type = "class")
confusionMatrix(y_hat_knn_401, mnist_27$test$y)$overall["Accuracy"]

# It turns out that this is similar to regression. This is called over-smoothing. We can see it
# by comparing the estimates of p(x_1,x_2) with the two models. Over-smoothing can occur when k
# is too large and does not permit enough flexibility.


# So how do we pick k? In principle, we want to pick the k that maximizes accuracy or minimizes
# the expected MSE as we defined earlier. The goal of cross validation is to estimate quantities
# such as accuracy or expected MSE so we can pick the best set of tuning parameters for any given
# algorithm, such as the k for kNN.

# ..Code..
knn_fit_1 <- knn3(y ~ ., data = mnist_27$train, k = 1)
y_hat_knn_1 <- predict(knn_fit_1, mnist_27$train, type = "class")
confusionMatrix(y_hat_knn_1, mnist_27$train$y)$overall["Accuracy"]

y_hat_knn_1 <- predict(knn_fit_1, mnist_27$test, type = "class")
confusionMatrix(y_hat_knn_1, mnist_27$test$y)$overall["Accuracy"]

# plot the estimates p_hat(x_1,x_2) and compare to p(x_1,x_2)
p1 <- mnist_27$true_p %>% 
  mutate(knn = predict(knn_fit_1, newdata = .)[,2]) %>%
  ggplot() +
  geom_point(data = mnist_27$train, aes(x_1, x_2, color= y), pch=21) +
  scale_fill_gradientn(colors=c("#F8766D", "white", "#00BFC4")) +
  stat_contour(aes(x_1, x_2, z = knn), breaks=c(0.5), color="black") +
  ggtitle("Train set")
p2 <- mnist_27$true_p %>% 
  mutate(knn = predict(knn_fit_1, newdata = .)[,2]) %>%
  ggplot() +
  geom_point(data = mnist_27$test, aes(x_1, x_2, color= y), 
             pch=21, show.legend = FALSE) +
  scale_fill_gradientn(colors=c("#F8766D", "white", "#00BFC4")) +
  stat_contour(aes(x_1, x_2, z = knn), breaks=c(0.5), color="black") +
  ggtitle("Test set")
grid.arrange(p1, p2, nrow=1)

knn_fit_401 <- knn3(y ~ ., data = mnist_27$train, k = 401)
y_hat_knn_401 <- predict(knn_fit_401, mnist_27$test, type = "class")
confusionMatrix(y_hat_knn_401, mnist_27$test$y)$overall["Accuracy"]

# plot the estimates p_hat(x_1,x_2) and compare to p(x_1,x_2)
fit_glm <- glm(y ~ x_1 + x_2, data=mnist_27$train, family="binomial")
p1 <- plot_cond_prob(predict(fit_glm, mnist_27$true_p)) +
  ggtitle("Regression")
p2 <- plot_cond_prob(predict(knn_fit_401, mnist_27$true_p)[,2]) +
  ggtitle("kNN-401")
grid.arrange(p1, p2, nrow=1)
