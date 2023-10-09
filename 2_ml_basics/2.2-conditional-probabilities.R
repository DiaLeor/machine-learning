## Section 2.2 - Conditional Probabilities

# Conditional Probabilities -----------------------------------------------

# In machine learning (ml) applications, we rarely predict outcomes perfectly. For example, spam
# detectors often miss emails that are clearly spam. It's impossible to build a perfect algorithm.
# To see this, note that most data sets will include groups of observations with the same exact
# observed values for all predictors but with different outcomes. Because our prediction rules are
# functions, equal inputs implies equal outputs. Therefore, for a challenge in which the same predictors
# are associated with different outcomes across different individual observations, it is impossible to
# predict correctly for all these cases. We saw a simple example of this in a previous lesson in which
# for any given x you will have both males and females that are x inches tall. However, none of this
# means that we can't build useful algorithms that are much better than guessing and, in some cases,
# better than expert opinions. To achieve this in an optimal way, we make use of probabilistic
# representations of the problem based on the ideas presented in our regression course (see link
# in README).

# Observations with the same observed value for the predictions may not all be the same, but we can
# assume that they all have the same probability for each class. We will write this idea out
# mathematically for the case of categorical data. We use notation:
# (X_1 = x_1,...,X_p = x_p) to represent the fact that we observe values x_1,...,x_p for features
# X_1,...X_p.
# The probabilistic approach assumes all the observations with these features have the same probability
# distribution as opposed to outcome y always taking the same specific value. In particular, we denote
# the conditional probabilities for each class k like this:
# Pr(Y = k | X_1 = x_1,...,X_p = x_p), for k = 1,...,K
# To avoid writing out all the predictors, we will use bold like this:
# bold_X ≡ (X_1,...,X_p).
# We will also use the following notation for the conditional probability being in class K:
# p_k(bold_x) = Pr(Y = k | bold_X = bold_x), for k = 1,...,K
# Now going forward, we'll use the small p notation to represent conditional probabilities as a
# function of the class.
  ## NOTE don't confuse this p with the p representing the number of features.

# These probabilities guide the construction of an algorithm that makes the best possible prediction.
# For any given x, we will predict the class k with the largest probability among the p_1,...,p_k(x),
# or:
# Y_hat = max_k*p_k(bold_x)
# In machine learning, this is referred to as Bayes' Rule. This is a theoretical rule because in
# practice we don't know p(x) (these probabilities). In fact, estimating these conditional probabilities can be thought of as the
# main challenge of machine learning. The better the estimate of these probabilities, the better our
# predictor.

# So our predictions depend on two things:
  # 1) how close are the maximum probabilities (max_k*p_k(bold_x)) to 1 or 0 (perfect certainty)
  # 2) how close our estimates (p_hat_k(bold_x)) are to the actual probabilities (p_k(bold_x))

# We can't do anything about the first restrictions, as it is determined by the nature of the
# problem. So our energy goes into finding ways to best estimate the conditional probabilities. This
# first restriction does imply that we have limits as to how well even the best possible algorithm can
# perform. You should get used to the idea that, while in some challenges, we will be able to achieve
# almost perfect accuracy, in others, our success is restricted by the randomness of the process. We'll
# see this in movie recommendations.

# It is important to remember that defining our predictions by maximizing the probability is not
# always optimal in practice and depends on the context. As discussed previously, sensitivity and
# specificity may differ in importance. But even in these cases, having a good estimate of the
# conditional probabilities for each class will suffice for us to build the optimal prediction models
# since we can control the balance between specificity and sensitivity however we wish. For example,
# we can simply change the cutoffs used to predict one outcome or the other. In the plane example,
# we may ground the plane any time the probability of malfunction is higher than one in a million as
# opposed to the default one-half used when error-types are equally undesired.

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