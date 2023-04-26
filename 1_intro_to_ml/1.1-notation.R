## Section 1.1 - Introduction to Machine Learning

# Notation ----------------------------------------------------------------

# Data comes in the form of features and outcomes.

# We want to build an algorithm that takes feature values as input and returns a prediction when we 
# don't know the actual outcome. We will train algorithms using datasets for which we do know the 
# actual outcome, and then apply these algorithms in the future to make predictions when we don’t 
# know the actual outcome. This is also referred to as supervised learning.

# Supervised learning problems can be divided into those with categorical outcomes, which we refer
# to as classification; and those with continuous outcomes, which we refer to as prediction.

# X_1,...,X_p denote the features/predictors/covariates, Y denotes the outcomes, and Y_hat denotes
# the predictions.

# Example: Zip Code Reader ------------------------------------------------

# In machine learning we want to build an algorithm that returns a prediction for any of the
# possible values of the features.

# Y_i = an outcome for observation or index i.

# We use boldface for bold_X_i to distinguish the vector of predictors from the individual predictors
# X_i,1,...X_i,784

# When referring to an arbitrary set of features and outcomes, we drop the index i and use Y and
# bold_X

# Uppercase is used to refer to variables because we think of predictors as random variables.
# Lowercase is used to denote observed values.

# Bold_X is an unobserved random variable and bold_x is an arbitrary value. Bold_X = bold_x means
# a realization of the random variable was observed and it was bold_x.