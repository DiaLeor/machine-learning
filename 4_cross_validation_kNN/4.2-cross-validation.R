## Section 4.2 - Cross-Validation

# Choosing k --------------------------------------------------------------

# We need a special method to choose k. To understand why, we can repeat what we did earlier
# but with different values of k:
ks <- seq(3, 251, 2)

# The following code uses the function map_df() to apply the same model (the same code) for each
# one of these k's:
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
  # NOTE: we estimate accuracy by using boththe training set and the test set

# And now we can plot the accuracy estimates for each value of k, both for the test set and the
# training set.
accuracy %>% mutate(k = ks) %>%
  gather(set, accuracy, -k) %>%
  mutate(set = factor(set, levels = c("train", "test"))) %>%
  ggplot(aes(k, accuracy, color = set)) + 
  geom_line() +
  geom_point()

# Due to over-training, the accuracy estimates obtained with the test set will be generally
# lower than the estimates obtained with the training set. But the difference is larger for
# smaller values of k. Also note that the accuracy vs. k plot is quite jagged. We do not expect
# this to be jagged because small changes in k shouldn't affect the algorithm's performance too
# much. The jaggedness is explained by the fact htat the accuracy is computed on a sample, and,
# therefore, it's a random variable.

# This demonstrates why we prefer to minimize the expected loss rather than the loss we observe in
# just one dataset. If we were to use these estimates to pick the k that maximizes accuracy, we would
# be using the estimate built on the test data. And we should not expect the accompanying accuracy
# estimate to extrapolate to the real world. This is because even here we broke a golden rule of
# machine learning: we selected the k using the test set. Cross validation also provides an estimate
# that takes these into account.

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

# Mathematical Description of Cross-Validation --------------------------------------------

# We previously explained that a common goal of machine learning is to find an algorithm that
# produces predictors y_hat for an outcome y that minimizes the mean squared error (MSE).
# When we have just one dataset at our disposal, we can estimate the MSE using the observed MSE with
# this formula:
# MSE = E{(1/N)*the sum from i = 1 through i = N of (Y_hat_i - Y_i)^2}
  # This is the theoretical MSE or true error

# MSE_hat = (1/N)*the sum from i = 1 through i = N of (y_hat_i - y_i)^2  
  # This is the observes MSE or apparent error

# There are two important characteristics of the apparent error to keep in mind. First, the
# apparent error is a random variable because our data are random. For example, the dataset we have may
# be a random sample from a larger population. An algorithm may have a lower apparent error than another
# algorithm just due to luck. Second, if we train an algorithm on the same dataset  we use the compute
# the apparent error, we might be over-training. In these cases, the apparent error will be an
# underestimate of the true error. Cross-validation helps to alleviate both of these problems.

# To understand cross-validation, it helps to think of the true error (a theoretical quantity) as the
# average of many apparent errors obtained by applying the algorithm to B (a very large number) new
# random samples of the data, none of them used to train the algorithm. In this scenario, we think of
# the true error as the following formula, with B, a large number that can be thought of as practically
# infinite:
# (1/B)*the sum from b = 1 through b = B*(1/N)*the sum from i = 1 through i = N of 
    # (y_i_hat^b - y_i^b)^2
  # As already mentioned, this is a theoretical quantity because we only have one set of outcomes
  # available to us.

# Cross-validation helps imitate the theoretical setup of the true error as best as possible
# using the data we have. To do this, cross-validation randomly generates smaller datasets that
# are not used for training and instead used to estimate the true error.

# k-fold Cross-Validation -------------------------------------------------

# The first version of cross-validation we describe is k-fold cross-validation. Generally speaking,
# a machine learning challenge starts with a dataset. We need to build an algorithm using this dataset
# that will eventually be used in completely independent datasets. Bue we don't get to see these
# independent datasets. So to imitate an independent dataset, we divide the availalbe dataset into a
# training set and a test set. The training set is used exclusively for training our algorithms and
# the test set is used exclusively for evaluation purposes. We usually try to select a small piece of
# the dataset so that we have as much data as possible to train. However, you also want the test set to
# be large enough so that we obtain a stable estimate of the loss without fitting an impractical number
# of models. Typically 10-20% of the data is set aside for testing. Again, it is indispensable that we
# not use the test set at all - not for filtering out rows, not for selecting features, nothing!

# Now, this presents a new problem because for most machine learning algorithms, we need to select
# parameters, for example: the number of neighbors k and k-nearest neighbors. We refer to the set of
# parameters as lambda λ, as you will often see in textbooks.

# We need to optimize algorithm parameters without using the test set, but if we optimize and
# evaluate on the same dataset we will over-train. This is where cross-validation is most useful. For
# each set of algorithm parameters being considered, we want an estimate of the MSE:
# MSE(λ) = (1/B)*the sum from b = 1 through b = B*(1/N)*the sum from i = 1 through i = N of 
    # (y_i_hat^b(λ) - y_i^b)^2

# And then we will choose the parameters that minimizes estimate of the MSE. Cross-validation
# provides this estimate. But before we start the cross-validation procedure, it is important to fix
# all the algorithm parameters. Although we will train the algorithm on the set of training sets, the
# parameters λ will be the same across all training sets. We will use y_i_hat(λ) to denote the
# predictors obtained when we use the parameter λ.

# So if we're going to imitate the mathematical definition of MSE, we want to create several datasets
# that can be thought of as independent random samples. With k-fold cross-validation, we do it k times.
# Let's start by describing how we construct the first sample. We simply pick M = N/K observations
# at random. We'll round M if it's not round. And think of these as a random sample: y_1^b,...y_M^b,
# with b = 1. We call this the validation set. Now we can fit the model in the training set then
# compute the apparent error on the independent set. This is the formula:
# MSE_b_hat(λ) = (1/M)*the sum from i = 1 through i = M of (y_i_hat^b(λ) - y_i^b)^2
  # NOTE this is just one sample and will therefore return a noisy estimate of the true error. This
  # is why we take k samples, not just one.

# In k cross-validation, we randomly split the data into k non-overlapping sets. Then we repeat the
# calculation that we just did for each of these sets, b, going 1 through k, and we obtain k estimates
# of the MSE. Then, for our final estimate, we compute the average given by this formula:
# MSE_hat(λ) = (1/K)*the sum from k = 1 through k = K of MSE_k_hat(λ)
  # and this gives us an estimate of our loss.
# Finally, we can pick the parameters (select the λ) that minimizes this estimate of the MSE.

# Now that we have described how to use cross-validation to optimize parameters, we have to take
# into account the fact that the optimization occurred on the training data, and, therefore, we need
# an estimate for our final algorithm based on data that was not used to optimize the choice. Here is
# where we use the test set we separated early on. We can do cross-validation again, if we want, and
# obtain a final estimate of our expected loss. However, note that this means that our entire compute
# time gets multiplied by k. You will learn that performing this task takes time because we're 
# performing many complex computations. As a result, we're always looking for ways to reduce this
# time. Therefore, for the final evaluation, we often just use the one test set. Once we're satisfied
# with this model and want to make it available to others, we could refit the model on the entire
# dataset without changing the optimization parameters.

# In terms of picking k for cross-validation, larger values of k are preferable because the training
# data better estimates the original dataset. However, larger values of k will also have much slower
# computational time. For example, 100-fold cross-validation will be 10 times slower than 10-fold
# cross-validation. For this reason, the choices of k = 5 or 10 are popular.

# One way to improve the variance of our final estimate is to take more samples. We can do this
# by no longer requiring non-overlapping sets. Instead, we would just pick case sets of some size
# at random. One popular version of this technique (the bootstrap) can be thought of as
# a variation at which each fold observations are picked at random with replacement. This is the
# default approach of caret::train.

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

# The population median is given by the following code:
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