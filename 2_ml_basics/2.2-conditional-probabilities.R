## Section 2.2 - Conditional Probabilities

# Conditional Probabilities -----------------------------------------------

# Conditional probabilities for each class:
# p_k(x) = Pr(Y = k | X = x), for k = 1.

# In machine learning, this is referred to as Bayes' Rule. This is a theoretical rule because in
# practice we don't know p(x). Having a good estimate of the p(x) will suffice for us to build
# optimal prediction models, since we can control the balance between specificity and sensitivity
# however we wish. In fact, estimating these conditional probabilities can be thought of as the
# main challenge of machine learning.

# Conditional Expectations ------------------------------------------------

# Due to the connection between conditional probabilities and conditional expectations:
# p_k(x) = Pr(Y = k | X = x), for k = 1, we often only use the expectation to denote both
# the conditional probability and conditional expectation.

# For continuous outcomes, we define a loss function to evaluate the model. The most commonly
# used one is MSE (Mean Squared Error). The reason why we care about the conditional expectation
# in machine learning is that the expected value minimizes the MSE:
# Y_hat = E(Y | X = x) minimizes E{(Y_hat - Y)}
# Due to this property, a succinct description of the main task of machine learning is that we use
# data to estimate this conditional expectation for any set of features. The main way in which
# competing machine learning algorithms differ is in their approach to estimating this expectation.