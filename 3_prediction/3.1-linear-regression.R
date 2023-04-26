## Section 3.1 - Linear Regression for Prediction

# Case Study: 2 or 7
# Linear regression can be considered a machine learning algorithm. Although it can be too rigid
# to be useful, it works rather well for some challenges. It also serves as a baseline approach: if
# you can’t beat it with a more complex approach, you probably want to stick to linear
# regression.

# In this case study we apply logistic regression to classify whether a digit is two or seven.
# We are interested in estimating a conditional probability that depends on two variables:
# p(x_1,x_2) = Pr(Y = 1 | X_1 = x_1, X_2 = x)

# Through this case, we know that logistic regression forces our estimates to be a plane and our
# boundary to be a line. This implies that a logistic regression approach has no chance of
# capturing the non-linear nature of the true p(x_1,x_2). Therefore, we need other more flexible
# methods that permit other shapes.

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