## Section 3.2 - Smoothing

# Smoothing ---------------------------------------------------------------

# Smoothing is a very powerful technique used all across data analysis. Other names given to this
# technique are curve fitting and low pass filtering. It is designed to detect trends in the presence
# of noisy data in cases in which the shape of the trend is unknown. The name smoothing comes from the
# fact that to accomplish this feat, we assume that the trend is a smooth surface. In contrast, the
# noise/deviation from the trend is unpredictably wobbly. We explain the assumptions that permit us
# to extract the trend from the noise.
 
# The concepts behind smoothing techniques are extremely useful in machine learning because
# conditional expectations/probabilities can be thought of as trends of unknown shapes that we
# need to estimate in the presence of uncertainty. To explain these concepts, again, we will focus
# first on a problem with just one predictor. Specifically, we try to estimate the time trend in the
# 2008 US popular vote poll margin (the difference between Obama and McCain).
 
# For the purposes of this example, do not think of it as a forecasting problem. Instead, we're simply
# interested in learning the shape of the trend after the election is over and all polling data has
# been gathered. We assume that for any given day, x, there is a true preference among the electorate.
# We will represent this with f(x). But due to the uncertainty introduced by polling, each data point
# comes with an error, which we'll represent with an ∈ (epsilon). A mathematical model for the observed
# poll margin is therefore the following:
# Y = f(x) + Ε (where E is epsilon).
 
# To think of this as a machine learning problem, consider that we want to predict Y given day x.
# It would be helpful to know the conditional expectation:
# f(x) = E(Y | X = x).
# But since we don't know this conditional expectation, we'll have to estimate it. We start by using
# regression since it's the only method we have learning up until now. The line we see doesn't appear
# to describe the trend very well. For example, on Sept. 4, day -62, the Republican Convention was
# held, and the data suggests that it gave McCain a boost in the polls. However, the regression line
# doesn't capture this potential trend. To see the lack of fit more clearly, note that points above
# the fitted line (the blue ones) and those below the red line are not evenly distributed across the
# days. Therefore, we need an alternative, more flexible approach.

# ..Code..
# see that the trend is wobbly
library(tidyverse)
set.seed(1)
n <- 100
x <- seq(-pi*4, pi*4, len = n)
tmp <- data.frame(x = x , f = sin(x) + x/8, e = rnorm(n, 0, 0.5)) 
p1 <- qplot(x, f, main = "smooth trend", ylim = range(tmp$f+tmp$e), data = tmp, geom = "line")
p2 <- qplot(x, e, main = "noise", ylim = range(tmp$f+tmp$e), data = tmp, geom = "line")
p3 <- qplot(x, f+e, main = "data = smooth trend + noise", ylim = range(tmp$f+tmp$e), data = tmp, geom = "line")
gridExtra::grid.arrange(p1, p2, p3)

# estimate the time trend in the 2008 US popular vote poll margin
library(tidyverse)
library(dslabs)
data("polls_2008")
qplot(day, margin, data = polls_2008)

# use regression to estimate
resid <- ifelse(lm(margin~day, data = polls_2008)$resid > 0, "+", "-")
polls_2008 %>% 
  mutate(resid = resid) %>% 
  ggplot(aes(day, margin)) + 
  geom_smooth(method = "lm", se = FALSE, color = "black") +
  geom_point(aes(color = resid), size = 3)

# Bin Smoothing -----------------------------------------------------------

# The general idea of smoothing is to group data points into strata in which the value of f(x)
# can be assumed to be constant. We can make this assumption because we think f(x) changes slowly
# and, as a result, f(x) is almost constant in small windows of time. An example of this idea for the
# poll_2008 data is to assume that public opinion remained approximately the same within a week's
# time. With this assumption in place, we have several data points with the same expected value.
# If we fix a day to be the center of our week (let's call it x_0), then for any other day x such that
# |x - x_0| is within a week (<= 3.5) we assume f(x) is constant. So we say f(x) = µ.

# In mathematical terms, the assumption implies:
# E[Y_i | X_i = x_i] ≈ mu if |x_i - x_0| <= 3.5
# that the expected value of y given x is approximately µ if x is within 3.5 days of x_0.
# In smoothing, we call the size of the interval |x_i - x_0| satisfying the particular condition
# the window size, bandwidth, or span. These are used interchangeably.
 
# This assumption implies that a good estimate for f(x) is the average of the Y_i values in the
# window. If we define A_0 as a set of indices, i, such that |x_i - x_0| is within 3.5, and we define
# N_0 as the number of indices in A_0, then our estimate can be written like this:
# f_hat(x_0) = (1)/(N_0) and the sum from i∈A_0 of Y_i

# The idea behind bin smoothing is to make this calculation with each value of x as the center. In the
# poll example, for each day we would compute the average of the values within a week with that day in
# the center. By computing this mean for every point, we form an estimate of the underlying curve f(x).
   ## NOTE: The final result from the bin smoother is quite wiggly. One reason for this is that each
# time the window moves, two points change. We can attenuate this somewhat by taking weighted averages
# that give the center point more weight than far away points, with the two points at the edges
# receiving very little weight.

# The bin smoother approach can be thought of as a weighted averge - mathematically, it is this:
# f_hat(x_0) = the sum from 1 to N of w_0(x_i)Y_i

# In the code, we use the argument kernel - "box" in our call to the function ksmooth(). This is
# because the weight function looks like a box. The ksmooth() function provides a smoother option,
# which uses the normal density to assign weights. Using the normal kernel in the final estimate
# looks smoother.

# There are several funcitons in R that implement bin smoother. ksmooth() is one of these. In practice,
# however, we typically prefer methods that use slightly more complex models than fitting a constant.
# Thus the final result is still somewhat wiggly in parts we don't expect it to be (like between
# -125 and -75). Methods such as loess improve on this.

# ..Code..
# bin smoothers
span <- 3.5
tmp <- polls_2008 %>%
  crossing(center = polls_2008$day) %>%
  mutate(dist = abs(day - center)) %>%
  filter(dist <= span) 

tmp %>% filter(center %in% c(-125, -55)) %>%
  ggplot(aes(day, margin)) +   
  geom_point(data = polls_2008, size = 3, alpha = 0.5, color = "grey") +
  geom_point(size = 2) +    
  geom_smooth(aes(group = center), 
              method = "lm", formula=y~1, se = FALSE) +
  facet_wrap(~center)

# larger span
span <- 7 
fit <- with(polls_2008, 
            ksmooth(day, margin, kernel = "box", bandwidth = span))

polls_2008 %>% mutate(smooth = fit$y) %>%
  ggplot(aes(day, margin)) +
  geom_point(size = 3, alpha = .5, color = "grey") + 
  geom_line(aes(day, smooth), color="red")

# kernel
span <- 7
fit <- with(polls_2008, 
            ksmooth(day, margin, kernel = "normal", bandwidth = span))

polls_2008 %>% mutate(smooth = fit$y) %>%
  ggplot(aes(day, margin)) +
  geom_point(size = 3, alpha = .5, color = "grey") + 
  geom_line(aes(day, smooth), color="red")

# Local Weighted Regression (loess) ---------------------------------------

# According to Taylor's Theorem, if you look closely enough at any smoothing function f(x), it will
# look like a line. In local weighted regression (loess)... instead of assuming that the function
# is approximately constant in a window (like we do in bin smoother), we assume that the function
# is locally linear.


# A limitation of the bin smoothing approach is that we need small windows for the approximately
# constant assumptions to hold which may lead to imprecise estimates of f(x). Local weighted
# regression (loess) permits us to consider larger window sizes. Instead of the one-week window,
# we consider a larger one in which the trend is approximately linear. We start with a three-week
# window and later consider and evaluate other options:
# E[Y_i|X_i = x_i] = β_0 + β_1(x_i - x_0) if |x_i - x_0| <= 21
# Now for every point x_0, loess defines a window and fites a line in that window. The fitted value
# at x_0 becomes our estimate f_hat of x_0. The result of loess is a smoother fit than bin smoothing
# because we use larger sample sizes to estimate our local parameters.

# Now notice that different spans give us different estimates. We can see how different window sizes
# lead to different estimates just by applying them and looking at the final results.

# ..Code..
# The full code can be found in the .R file Handout at the beginning of the section.
polls_2008 %>% ggplot(aes(day, margin)) +
  geom_point() +
  geom_smooth(color="red", span = 0.15, method = "loess", method.args = list(degree=1))

# Beware of Default Smoothing Parameters ----------------------------------

# Local weighted regression (loess) permits us to fit parabola by considering a larger
# window size than the one considered while fitting a line.

# ..Code..
# The full code can be found in the .R file Handout at the beginning of the section.

total_days <- diff(range(polls_2008$day))
span <- 28/total_days
fit_1 <- loess(margin ~ day, degree=1, span = span, data=polls_2008)
fit_2 <- loess(margin ~ day, span = span, data=polls_2008)


polls_2008 %>% mutate(smooth_1 = fit_1$fitted, smooth_2 = fit_2$fitted) %>%
  ggplot(aes(day, margin)) +
  geom_point(size = 3, alpha = .5, color = "grey") +
  geom_line(aes(day, smooth_1), color="red", lty = 2) +
  geom_line(aes(day, smooth_2), color="orange", lty = 1)


polls_2008 %>% ggplot(aes(day, margin)) +
  geom_point() +
  geom_smooth()

# Connecting Smoothing to Machine Learning --------------------------------

# In the 2 vs 7 example, we saw that linear regression was not flexible enough to capture
# the non-linear nature of p(x_1,x_2). Smoothing approaches may provide an improvement in
# capturing the same.