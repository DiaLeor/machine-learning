## Section 4.1 - k-Nearest Neighbors

# k-Nearest-Neighbors (kNN) -----------------------------------------------------

# K-nearest neighbors (kNN) estimates the conditional probabilities in a similar way to bin
# smoothing. However, kNN is easier to adapt to multiple dimensions.

# Using kNN, for any point (x_1,x_2) for which we want an estimate of p(x_1,x_2), we look for
# the nearest points to (x_1,x_2) and take an average of the 0s and 1s associated with these
# points. We refer to the set of points used to compute the average as the neighborhood.
# Larger values of k result in smoother estimates, while smaller values of k result in more
# flexible and more wiggly estimates. 

# To implement the algorithm, we can use the knn3() function from the caret package. There are
# two ways to call this function:
  
  # 1. We can specify a formula and a data frame. The formula looks like
  # y ~ x_1 + x_2 or y ~.
  
  # the predict() function for knn3 produces a probability for each class.
  # 2. We can also call the function with the first argument being the matrix predictors and
  # the second a vector of outcomes, like this:

x <- as.matrix(mnist_27$train)
y <- mnist_27$train$y
knn_fit <- knn3(x,y)

# Overtraining is when a machine learning model can make good predictions on the training set
# but cannot generalize well to new data. Overtraining is a reason why we have higher
# accuracy in the training set compared to the test set.

# ..Code..
library(tidyverse)
library(caret)
library(dslabs)
library(gridExtra)
library(tidyverse)

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

# Overtraining and Oversmoothing ----------------------------------------

# With k-nearest neighbors (kNN), overtraining is at its worst when we set k = 1.
# With k = 1, the estimate for each (x_1,x_2) in the training set is obtained with just
# the y corresponding to that point.

# Oversmoothing can occur when when k is too large and does not permit enough flexibility.

# The goal of cross validation is to estimate quantities such as accuracy or expected MSE so we
# can pick the best set of tuning parameters for any given algorithm, such as the k for kNN.

# ..Code..
knn_fit_1 <- knn3(y ~ ., data = mnist_27$train, k = 1)
y_hat_knn_1 <- predict(knn_fit_1, mnist_27$train, type = "class")
confusionMatrix(y_hat_knn_1, mnist_27$train$y)$overall["Accuracy"]

y_hat_knn_1 <- predict(knn_fit_1, mnist_27$test, type = "class")
confusionMatrix(y_hat_knn_1, mnist_27$test$y)$overall["Accuracy"]

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

fit_glm <- glm(y ~ x_1 + x_2, data=mnist_27$train, family="binomial")
p1 <- plot_cond_prob(predict(fit_glm, mnist_27$true_p)) +
  ggtitle("Regression")
p2 <- plot_cond_prob(predict(knn_fit_401, mnist_27$true_p)[,2]) +
  ggtitle("kNN-401")
grid.arrange(p1, p2, nrow=1)