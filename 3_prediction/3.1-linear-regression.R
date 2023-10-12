## Section 3.1 - Linear Regression for Prediction

# Case Study: 2 or 7? -----------------------------------------------------


# In the simple example we've discussed, we only had one predictor. Technically, we can't even consider
# this a machine learning (ml) challenge, which are characterized by cases with many predictors. Let's
# re-examine the digits problem, in which we had 784 predictors. For illustrative purposes, we start by simplifying
# this data set to one with two predictors and two cases. We will define the challenge as building an algorithm
# that can distinguish the digit 2 from the digit 7. We picked 2 and 7 because they can look quite similar, for
# example, if the 2 loses its base.

# Because we're not quite ready to build an algorithm for 784 predictors, we will extract two simple predictors
# from these 784. These will be the proportion of dark pixels that are in the upper left quadrant (x_1) and the
# proportion of dark pixels in the lower right quadrant (x_2). Provided is a random sample of 1000 digits (800 in
# the training set & 200 in the test set) in the dslabs package.

library(tidyverse)
library(dslabs)
data("mnist_27")

# We can explore the data by plotting the two predictors and using color to denote the two labels, 2 or 7.
mnist_27$train %>% ggplot(aes(x_1, x_2, color = y)) + geom_point() 
# We can immediately see some patterns, for example, if x_1 (upper left) is very large, then the digit is probably
# a 7. Also, for smaller values of x_1, the 2's appear to be in the mid-range values of x_2 (lower right).
# To illustrate how to interpret x_1 and x_2, the lesson has provided 4 images of the two digits with the largest
# and smallest values for x_1 and the largest and smallest values of x_2:
if(!exists("mnist")) mnist <- read_mnist()
is <- mnist_27$index_train[c(which.min(mnist_27$train$x_1), which.max(mnist_27$train$x_1))]
titles <- c("smallest","largest")
tmp <- lapply(1:2, function(i){
  expand.grid(Row=1:28, Column=1:28) %>%  
    mutate(label=titles[i],  
           value = mnist$train$images[is[i],])
})
tmp <- Reduce(rbind, tmp)
p1 <- tmp %>% ggplot(aes(Row, Column, fill=value)) + 
  geom_raster(show.legend = FALSE) + 
  scale_y_reverse() +
  scale_fill_gradient(low="white", high="black") +
  facet_grid(.~label) + 
  geom_vline(xintercept = 14.5) +
  geom_hline(yintercept = 14.5) +
  ggtitle("Largest and smallest x_1")

is <- mnist_27$index_train[c(which.min(mnist_27$train$x_2), which.max(mnist_27$train$x_2))]
titles <- c("smallest","largest")
tmp <- lapply(1:2, function(i){
  expand.grid(Row=1:28, Column=1:28) %>%  
    mutate(label=titles[i],  
           value = mnist$train$images[is[i],])
})
tmp <- Reduce(rbind, tmp)
p2 <- tmp %>% ggplot(aes(Row, Column, fill=value)) + 
  geom_raster(show.legend = FALSE) + 
  scale_y_reverse() +
  scale_fill_gradient(low="white", high="black") +
  facet_grid(.~label) + 
  geom_vline(xintercept = 14.5) +
  geom_hline(yintercept = 14.5) +
  ggtitle("Largest and smallest x_2")
gridExtra::grid.arrange(p1, p2, ncol = 2)
# We can start getting a sense for why these
# predictors are useful but also why the problem will be somewhat challenging.

# Let's try building an algorithm using linear regression. The model is simply this:
# In this case study we apply logistic regression to classify whether a digit is two or seven.
# We are interested in estimating a conditional probability that depends on two variables
# p(x_1,x_2) = Pr(Y = 1 | X_1 = x_1, X_2 = x_2) = β_0 + β_1*x_1 + β_2*x_2
# We can fit it using what we already learned in the regression course:
fit <- mnist_27$train %>%
  mutate(y = ifelse(y == 7, 1, 0)) %>%
  lm(y ~ x_1 + x_2, data = .)

# We can now build a decision rule based on the estimate of p of x_1, x_2 (p_hat(x_1,x_2))
library(caret)

p_hat <- predict(fit, newdata = mnist_27$test, type = "response")

p_hat <- predict(fit, newdata = mnist_27$test)
y_hat <- factor(ifelse(p_hat > 0.5, 7, 2))

confusionMatrix(y_hat, mnist_27$test$y)$overall[["Accuracy"]] # We get an accuracy well above 50%
# Not bad. But can we do better?

# Because we constructed the mnist27 example, and we had at our disposal 60,000 digits in the mnist data set, we
# use this to build the true conditional distribution — p(x_1,x_2).
  ## NOTE this is something we don't have access to in practice, but we include it in this example because it
  ## permits the comparison of estimates of p to the true p. This comparison teaches us the limitations of
  ## different algorithms.

# This is a plot of the true p(x_1,x_2), a function we don't get to see in practice:
mnist_27$true_p %>% ggplot(aes(x_1, x_2, z = p, fill = p)) +
  geom_raster() +
  scale_fill_gradientn(colors=c("#F8766D", "white", "#00BFC4")) +
  stat_contour(breaks=c(0.5), color="black")
# To start understanding the limitations of regression, NOTE that, with regression, the estimate has to be a plane.
# And as a result, the boundary defined by the decision rule given by p_hat(x,y) = 0.5, which implies — we can show
# this mathematically — that the boundary can't be anything other than a straight line:
# β_hat_0 + β_hat_1*x_1 + β_hat_2*x_2 = 0.5 ⟹
# x_2 = (0.5 - β_hat_0 - β_hat_1*x_1)/β_hat_2
  ## NOTE for the boundary, x_2 is a linear function of x_1. This implies that our regression approach has no
  ## chance of capturing the non-linear nature of the true p(x_1,x_2).
#Here is a visual representation of our estimate using regression:
p_hat <- predict(fit, newdata = mnist_27$true_p)
p_hat <- scales::squish(p_hat, c(0, 1))
p1 <- mnist_27$true_p %>% mutate(p_hat = p_hat) %>%
  ggplot(aes(x_1, x_2,  z=p_hat, fill=p_hat)) +
  geom_raster() +
  scale_fill_gradientn(colors=c("#F8766D","white","#00BFC4")) +
  stat_contour(breaks=c(0.5), color="black") 

p2 <- mnist_27$true_p %>% mutate(p_hat = p_hat) %>%
  ggplot() +
  stat_contour(aes(x_1, x_2, z=p_hat), breaks=c(0.5), color="black") + 
  geom_point(mapping = aes(x_1, x_2, color=y), data = mnist_27$test) 
gridExtra::grid.arrange(p1, p2, ncol = 2)

# Through this case, we know that logistic regression forces our estimates to be a plane and our
# boundary to be a line. This implies that a logistic regression approach has no chance of
# capturing the non-linear nature of the true p(x_1,x_2). Therefore, we need other more flexible
# methods that permit other shapes. We're going to learn a few algorithms based on different ideas and
# concepts. But what they all have in common is that they permit more flexible approaches.

# Linear regression can be considered a machine learning algorithm. Although it can be
# too rigid to be useful, it works rather well for some challenges. It also serves as a baseline approach: if
# you can't beat it with a more complex approach, you probably want to stick to linear
# regression.

# We will start by describing nearest-neighbor approaches and kernel approaches, starting again with a one-dimensional
# example and describing the concept of smoothing.

# ..Code..
# load the dataset
library(tidyverse)
library(dslabs)
data("mnist_27")

# explore the data by plotting the two predictors
mnist_27$train %>% ggplot(aes(x_1, x_2, color = y)) + geom_point()

# smallest and largest values of x1 and x2
if(!exists("mnist")) mnist <- read_mnist()
is <- mnist_27$index_train[c(which.min(mnist_27$train$x_1), which.max(mnist_27$train$x_1))]
titles <- c("smallest","largest")
tmp <- lapply(1:2, function(i){
  expand.grid(Row=1:28, Column=1:28) %>%  
    mutate(label=titles[i],  
           value = mnist$train$images[is[i],])
})
tmp <- Reduce(rbind, tmp)
p1 <- tmp %>% ggplot(aes(Row, Column, fill=value)) + 
  geom_raster(show.legend = FALSE) + 
  scale_y_reverse() +
  scale_fill_gradient(low="white", high="black") +
  facet_grid(.~label) + 
  geom_vline(xintercept = 14.5) +
  geom_hline(yintercept = 14.5) +
  ggtitle("Largest and smallest x_1")

is <- mnist_27$index_train[c(which.min(mnist_27$train$x_2), which.max(mnist_27$train$x_2))]
titles <- c("smallest","largest")
tmp <- lapply(1:2, function(i){
  expand.grid(Row=1:28, Column=1:28) %>%  
    mutate(label=titles[i],  
           value = mnist$train$images[is[i],])
})
tmp <- Reduce(rbind, tmp)
p2 <- tmp %>% ggplot(aes(Row, Column, fill=value)) + 
  geom_raster(show.legend = FALSE) + 
  scale_y_reverse() +
  scale_fill_gradient(low="white", high="black") +
  facet_grid(.~label) + 
  geom_vline(xintercept = 14.5) +
  geom_hline(yintercept = 14.5) +
  ggtitle("Largest and smallest x_2")
gridExtra::grid.arrange(p1, p2, ncol = 2)

# fit the model
fit <- mnist_27$train %>%
  mutate(y = ifelse(y == 7, 1, 0)) %>%
  lm(y ~ x_1 + x_2, data = .)

# build a decision rule
library(caret)

p_hat < predict(fit, newdata = mnist_27$test, type = "response")
y_hat <- factor(ifelse(p_hat > 0.5, 7, 2))

confusionMatrix(y_hat, mnist_27$test$y)$overall[["Accuracy"]]

# plot the true values
mnist_27$true_p %>% ggplot(aes(x_1, x_2, z = p, fill = p)) +
  geom_raster() +
  scale_fill_gradientn(colors=c("#F8766D", "white", "#00BFC4")) +
  stat_contour(breaks=c(0.5), color="black")

# visual representation of p_hat
p_hat <- predict(fit, newdata = mnist_27$true_p)
p_hat <- scales::squish(p_hat, c(0, 1))
p1 <- mnist_27$true_p %>% mutate(p_hat = p_hat) %>%
  ggplot(aes(x_1, x_2,  z=p_hat, fill=p_hat)) +
  geom_raster() +
  scale_fill_gradientn(colors=c("#F8766D","white","#00BFC4")) +
  stat_contour(breaks=c(0.5), color="black") 

p2 <- mnist_27$true_p %>% mutate(p_hat = p_hat) %>%
  ggplot() +
  stat_contour(aes(x_1, x_2, z=p_hat), breaks=c(0.5), color="black") + 
  geom_point(mapping = aes(x_1, x_2, color=y), data = mnist_27$test) 
gridExtra::grid.arrange(p1, p2, ncol = 2)