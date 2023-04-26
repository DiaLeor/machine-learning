## Section 4.2 - Cross-Validation

# Choosing k --------------------------------------------------------------

# Due to overtraining, the accuracy estimates obtained with the test set will be generally
# lower than the estimates obtained with the training set.

# We prefer to minimize the expected loss rather than the loss we observe in just one dataset.
# Also, if we were to use the test set to pick k, we should not expect the accompanying accuracy
# estimate to extrapolate to the real world. This is because even here we broke a golden rule
# of machine learning: we selected the  using the test set. Cross validation provides an
# estimate that takes these into account.

# ..Code..
ks <- seq(3, 251, 2)

library(purrr)
accuracy <- map_df(ks, function(k){
  fit <- knn3(y ~ ., data = mnist_27$train, k = k)
  
  y_hat <- predict(fit, mnist_27$train, type = "class")
  cm_train <- confusionMatrix(y_hat, mnist_27$train$y)
  train_error <- cm_train$overall["Accuracy"]
  
  y_hat <- predict(fit, mnist_27$test, type = "class")
  cm_test <- confusionMatrix(y_hat, mnist_27$test$y)
  test_error <- cm_test$overall["Accuracy"]
  
  tibble(train = train_error, test = test_error)
})

accuracy %>% mutate(k = ks) %>%
  gather(set, accuracy, -k) %>%
  mutate(set = factor(set, levels = c("train", "test"))) %>%
  ggplot(aes(k, accuracy, color = set)) + 
  geom_line() +
  geom_point()

ks[which.max(accuracy$test)]
max(accuracy$test)

# Mathematical Description of  --------------------------------------------

# When we have just one dataset, we can estimate the MSE using the observed MSE. The theoretical
# MSE is often referred to as the true error, and the observed MSE as the apparent error.

# There are two important characteristics of the apparent error to keep in mind. First, the
# apparent error is a random variable. Second, if we train an algorithm on the same dataset
# we use the compute the apparent error, we might be overtraining. In these cases, the apparent
# error will be an underestimate of the true error. Cross-validation helps to alleviate both
# of these problems.

# Cross-validation helps imitate the theoretical set up of the true error as best as possible
# using the data we have. To do this, cross-validation randomly generates smaller datasets that
# are used to estimate the true error.

# k-fold Cross-Validation -------------------------------------------------

# To imitate an independent dataset, we divide the dataset into a training set and a test set.
# The training set is used for training our algorithms and the test set is used exclusively for
# evaluation purposes. Typically 10-20% of the data is set aside for testing.

# We need to optimize algorithm parameters without using the test set, but if we optimize and
# evaluate on the same dataset we will overtrain. This is where cross-validation is useful.

# To calculate MSE, we want to create several datasets that can be thought of as independent
# random samples. With k-fold cross-validation, we randomly split the data into k
# non-overlapping sets. We obtain k estimates of the MSE and then compute the average as a
# final estimate of our loss. Finally, we can pick the parameters that minimize this estimate
# of the MSE.

# For a final evaluation of our algorithm, we often just use the test set.

# In terms of picking k for cross-validation, larger values of  are preferable but they will
# also take much more computational time. For this reason, the choices of 5 and 10 are common.

# One way to improve the variance of our final estimate is to take more samples. We can do this
# by no longer requiring non-overlapping sets. The bootstrap can be thought of as a variation
# at which each fold observations are picked at random with replacement. This is the default
# approach of caret::train.

# Bootstrap ---------------------------------------------------------------

# Bootstrapping allows us to approximate a Monte Carlo simulation without access to the
# entire distribution. We act as if the observed sample is the population. Next, we sample
# datasets (with replacement) of the same sample size as the original dataset. Finally,
# we compute the summary statistic, in this case the median, on these bootstrap samples.

# Suppose the income distribution of your population is as follows:
set.seed(1995)
n <- 10^6
income <- 10^(rnorm(n, log10(45000), log10(3)))
qplot(log10(income), bins = 30, color = I("black"))
The population median is given by the following code:
m <- median(income)
# The population median is 44939.

# However, if we don't have access to the entire population but want to estimate the median,
# we can take a sample of 100 and estimate the population median m using the sample median M,
# like this:
N <- 100
X <- sample(income, N)
median(X)
# The sample median here is 38461.

# Now let's consider constructing a confidence interval and determining the distribution of M.
# Because we are simulating the data, we can use a Monte Carlo simulation to learn the
# distribution of M using the following code:
library(gridExtra)
B <- 10^4
M <- replicate(B, {
  X <- sample(income, N)
  median(X)
})
p1 <- qplot(M, bins = 30, color = I("black"))
p2 <- qplot(sample = scale(M), xlab = "theoretical", ylab = "sample") + 
  geom_abline()
grid.arrange(p1, p2, ncol = 2)

# Knowing the distribution allows us to construct a confidence interval. However, as we have
# discussed before, in practice, we do not have access to the distribution. In the past, we
# have used the Central Limit Theorem (CLT), but the CLT we studied applies to averages and
# here we are interested in the median. If we construct the 95% confidence interval based on
# the CLT using the code below, we see that it is quite different from the confidence
# interval we would generate if we knew the actual distribution of M.
median(X) + 1.96 * sd(X) / sqrt(N) * c(-1, 1)
# The 95% confidence interval based on the CLT is (21017, 55904).
quantile(M, c(0.025, 0.975))
# The confidence interval based on the actual distribution of M is (34438, 59050).

# The bootstrap permits us to approximate a Monte Carlo simulation without access to the
# entire distribution. The general idea is relatively simple. We act as if the observed
# sample is the population. We then sample (with replacement) datasets, of the same sample
# size as the original dataset. Then we compute the summary statistic, in this case the
# median, on these bootstrap samples.

# Theory tells us that, in many situations, the distribution of the statistics obtained
# with bootstrap samples approximate the distribution of our actual statistic. We can
# construct bootstrap samples and an approximate distribution using the following code:
B <- 10^4
M_star <- replicate(B, {
  X_star <- sample(X, N, replace = TRUE)
  median(X_star)
})

# The confidence interval constructed using the bootstrap is much closer to the one
# constructed with the theoretical distribution, as you can see by using this code:
quantile(M_star, c(0.025, 0.975))
# The confidence interval from the bootstrap is (30253, 56909).

# Note that we can use ideas similar to those used in the bootstrap in cross-validation:
# instead of dividing the data into equal partitions, we can simply bootstrap many times.

# In summary:
# When we don't have access to the entire population, we can use bootstrap to estimate
# the population median m.

# The bootstrap permits us to approximate a Monte Carlo simulation without access to the
# entire distribution. The general idea is relatively simple. We act as if the observed
# sample is the population. We then sample datasets (with replacement) of the same sample
# size as the original dataset. Then we compute the summary statistic, in this case the
# median, on this bootstrap sample.

# Note that we can use ideas similar to those used in the bootstrap in cross validation:
# instead of dividing the data into equal partitions, we simply bootstrap many times.

# ..Code..
# define the population distribution of income
set.seed(1995)
n <- 10^6
income <- 10^(rnorm(n, log10(45000), log10(3)))
qplot(log10(income), bins = 30, color = I("black"))

# calculate the population median
m <- median(income)
m

# estimate the population median
N <- 100
X <- sample(income, N)
M<- median(X)
M

# use a Monte Carlo simulation to learn the distribution of M
library(gridExtra)
B <- 10^4
M <- replicate(B, {
    X <- sample(income, N)
    median(X)
})
p1 <- qplot(M, bins = 30, color = I("black"))
p2 <- qplot(sample = scale(M), xlab = "theoretical", ylab = "sample") + geom_abline()
grid.arrange(p1, p2, ncol = 2)

# compare the 95% CI based on the CLT to the actual one
median(X) + 1.96 * sd(X) / sqrt(N) * c(-1, 1)
quantile(M, c(0.025, 0.975))

# bootstrap and approximate the distribution
B <- 10^4
M_star <- replicate(B, {
    X_star <- sample(X, N, replace = TRUE)
    median(X_star)
})

# look at the confidence interval from the bootstrap
quantile(M_star, c(0.025, 0.975))