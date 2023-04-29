## Section 1.1 - Introduction to Machine Learning

# Notation ----------------------------------------------------------------

# Data comes in the form of features and outcomes.

# We want to build an algorithm that takes feature values as input and returns a prediction when we 
# don't know the actual outcome. We will train algorithms using datasets for which we do know the 
# actual outcome, and then apply these algorithms in the future to make predictions when we don’t 
# know the actual outcome. X_1,...,X_p denote the features/predictors/covariates, Y denotes the
# outcomes, and Y_hat denotes the predictions.

# For categorical outcomes, Y can be any of k classes. The number of classes can vary greatly
# across applications. For example, with digit readers, K = 10. The classes being the digits 0
# through 9. In speech recognition, the outcomes are all possible words or phrases. Spam
# detection has two outcomes: spam or not spam. In this course, we denote the k categories
# with indices. Little k equals 1 or 2, 3, all the way up to capital K. However, for binary
# data, we will use k = 0 or 1 for mathematical convenience.

# The general setup is as follows: we have a series of features and an unknown outcome we want to
# predict. To build a model that provides a prediction for any set of observable values x_1 through
# x_5, we collect data for which we know the outcome. This is also referred to as supervised
# learning. Supervised learning problems can be divided into those with categorical outcomes,
# which we refer to as classification; and those with continuous outcomes, which we refer to as
# prediction.

# In classification, the main output of the model will be a decision rule which prescribes which
# of the k classes we should predict. In this scenario, most models provide a function of the
# predictors for each class. These are used to make the decision:
# f_k(x_1,x_2,...,x_p)
# When the data is binary, a typical decision rule looks like this:
# f_1(x_1,x_2,...,x_p) > C.
# If f_1 of x_1 through x_p is bigger than some constant (some predefined constant C), then we
# predict category 1.
# # f_1(x_1,x_2,...,x_p) <= C.
# And if it's less than C, then we predict category 0.
# Because the outcomes are categorical, our predictions will be either right or wrong.

# In prediction, the main output of the model is a function f() that automatically produces a
# prediction denoted with a y_hat for any set of predictors x_1 through x_p:
# y_hat = f(x_1,x_2,...,x_p).

# We use the term actual outcome to denote what we end up observing! So we want the prediction
# y_hat to match the actual outcome y as closely as possible. Because our outcome is continuous,
# our prediction y_hat will not be either exactly right or wrong, but instead we'll determine an
# error defined as the difference between the prediction and the actual outcome.



# Example: Zip Code Reader ------------------------------------------------

# The first step in building an algorithm is to understand what are the outcomes and the features.
# In machine learning we want to build an algorithm that returns a prediction for any of the
# possible values of the features.

# Data which have already been read by a human and assigned an outcome Y are considered known
# and serve as a training set.

# Y_i = an outcome for observation or index i.

# We use boldface for bold_X_i to distinguish the vector of predictors from the individual predictors
# X_i,1,...X_i,784

# When referring to an arbitrary set of features and outcomes rather than a specific image in our
# data set, we drop the index i and use Y and bold_X.

# Uppercase is used to refer to variables because we think of predictors as random variables.
# Lowercase is used to denote observed values. However when we code in R, we stick to lowercase.

# Bold_X is an unobserved random variable and bold_x is an arbitrary value. Bold_X = bold_x means
# a realization of the random variable was observed and it was bold_x.